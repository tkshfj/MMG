# handlers.py
from __future__ import annotations
import os
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, global_step_from_engine

from image_logger import make_image_logger
from metrics_utils import to_float_scalar, train_loss_output_transform
from engine_utils import is_ignite_engine

logger = logging.getLogger(__name__)


def configure_wandb_step_semantics() -> None:
    """
    Predeclare W&B step domains so metric groups always use the right axis.
    - trainer/epoch:     epoch-synchronous logs (validation, epoch aggregates)
    - trainer/iteration: iteration-synchronous logs (per-batch training)
    """
    import wandb
    try:
        if getattr(wandb, "run", None) is None:
            return  # nothing to configure yet
        wandb.define_metric("trainer/epoch")
        wandb.define_metric("trainer/iteration")
        wandb.define_metric("val/*", step_metric="trainer/epoch")
        wandb.define_metric("eval/*", step_metric="trainer/epoch")
        wandb.define_metric("train/*", step_metric="trainer/iteration")
        wandb.define_metric("opt/*", step_metric="trainer/epoch")
        # no seed log needed; epoch anchors will create the axes naturally
    except Exception:
        pass


_INT_PREFIXES = ("cls_confmat_", "seg_confmat_", "tp", "tn", "fp", "fn")


def _to_scalar(v: Any) -> Optional[float]:
    """
    Return a Python scalar (float) if v is scalar-like; else None.
    - Accepts: int/float, numpy scalars/0-d arrays, 1-element sequences, 1-element tensors.
    - Skips: longer lists/arrays/tensors, dicts, etc.
    """
    if isinstance(v, (int, float, np.number)):
        return float(v)
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return float(v.detach().cpu().item())
        return None
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return float(v.item())
        if v.size == 1:
            return float(v.reshape(()).item())
        return None
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return _to_scalar(v[0])
        return None
    # anything else: try a safe extraction or skip
    try:

        if hasattr(v, "item"):
            return float(v.item())  # numpy scalar-like
    except Exception:
        pass
    return None


def _cast_for_logging(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only scalar metrics. For count-like keys, cast to int; otherwise to float.
    """
    out: Dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        s = _to_scalar(v)
        if s is None:
            continue
        if any(k.startswith(pref) for pref in _INT_PREFIXES):
            out[k] = int(round(s))
        else:
            out[k] = float(s)
    return out


def _init_wandb_logger(project, run_name):
    """Initialize wandb if not already running; return the module."""
    import wandb
    try:
        if getattr(wandb, "run", None) is None:
            kwargs = {}
            if project:
                kwargs["project"] = project
            if run_name:
                kwargs["name"] = run_name
            wandb.init(**kwargs)
    except Exception as e:
        logger.warning("W&B init skipped: %s", e)
    return wandb


def _wb_log(wb_mod, payload: dict, *, commit: Optional[bool] = None) -> None:
    if wb_mod is None:
        return
    try:
        if commit is None:
            wb_mod.log(payload)
        else:
            wb_mod.log(payload, commit=commit)
    except Exception as e:
        logger.warning("W&B log failed: %s", e)


def register_handlers(
    *,
    trainer: Engine,
    evaluator: Optional[Engine],                  # may be None or non-Engine (two-pass case)
    evaluator_cal: Optional[Engine] = None,      # unused here, kept for signature compat
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    out_dir: str = "./outputs",
    last_model_prefix: str = "last",
    best_model_prefix: str = "best",
    save_last_n: int = 2,
    watch_metric: str = "val_auc",
    watch_mode: str = "max",
    metric_names: Optional[Iterable[str]] = None,
    log_images: bool = False,
    image_every_n_epochs: int = 1,
    image_max_items: int = 8,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_enable: bool = True,
    cfg: Optional[dict] = None,
    console_iter_log: Optional[bool] = None,
    console_epoch_log: Optional[bool] = None,
    run_tag: Optional[str] = None,
    seg_threshold: float = 0.5,
) -> None:
    """
    Attach training/eval logging, W&B, image logging, and checkpointing.
    Works with either an Ignite evaluator Engine or a non-Engine two-pass evaluator
    that publishes metrics into trainer.state.metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    run_tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")

    # W&B init (optional) + step semantics
    wb = None
    if wandb_enable:
        try:
            wb = _init_wandb_logger(wandb_project, wandb_run_name)
            configure_wandb_step_semantics()
        except Exception as e:
            wb = None
            print(f"[WARN] W&B init failed: {e}")

    # ========= Training logs (per-iteration; commit=False) =========
    @trainer.on(Events.ITERATION_COMPLETED)
    def _log_train_iter(engine: Engine):
        if wb is None:
            return
        it = int(engine.state.iteration)
        raw = train_loss_output_transform(engine.state.output) or {}
        scalars = _cast_for_logging(raw)
        payload: Dict[str, Any] = {"trainer/iteration": it}
        payload.update({f"train/{k}": v for k, v in scalars.items()})
        if optimizer is not None and optimizer.param_groups:
            payload["train/lr"] = float(optimizer.param_groups[0].get("lr", 0.0))
        _wb_log(wb, payload, commit=False)

    # Epoch anchor to flush per-iteration buffers
    @trainer.on(Events.EPOCH_COMPLETED)
    def _commit_epoch_anchor(engine: Engine):
        if wb is None:
            return
        _wb_log(wb, {"trainer/epoch": int(engine.state.epoch)})  # commit=True by default

    # ========= Eval logging + best checkpoint =========
    is_ev_engine = is_ignite_engine(evaluator)

    if is_ev_engine and evaluator is not None:
        # Console eval stats
        @evaluator.on(Events.COMPLETED)
        def _print_eval_metrics(engine: Engine):
            ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
            scalars = _cast_for_logging(engine.state.metrics or {})
            if scalars:
                parts = [f"{k}: {v}" if isinstance(v, int) else f"{k}: {v:.4f}" for k, v in scalars.items()]
                logger.info("Epoch[%d] Metrics -- %s", ep, " ".join(parts))

        # W&B eval logs on epoch axis
        if wb is not None:
            @evaluator.on(Events.COMPLETED)
            def _log_eval(engine: Engine):
                scalars = _cast_for_logging(engine.state.metrics or {})
                if not scalars:
                    return
                ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
                payload = {"trainer/epoch": ep}
                payload.update({f"val/{k}": v for k, v in scalars.items()})
                _wb_log(wb, payload)

        # Best checkpoint (per-run folder, tolerant to non-empty)
        if model is not None:
            best_dir = os.path.join(out_dir, "checkpoints_best", run_tag)
            os.makedirs(best_dir, exist_ok=True)

            sign = 1.0 if str(watch_mode).lower() == "max" else -1.0

            def _score_fn(e: Engine):
                m = e.state.metrics or {}
                v = to_float_scalar(m.get(watch_metric))
                return sign * float(v) if v is not None else -float("inf")

            ckpt_best = ModelCheckpoint(
                dirname=best_dir,
                filename_prefix=best_model_prefix,
                n_saved=max(1, int(os.getenv("BEST_MODEL_RETENTION", "3"))),
                score_function=_score_fn,
                score_name=watch_metric,
                global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
                create_dir=True,
                require_empty=False,
            )
            evaluator.add_event_handler(Events.COMPLETED, ckpt_best, {"model": model})

    # Two-pass path: metrics live on trainer.state.metrics
    if (not is_ev_engine) or (evaluator is None):
        @trainer.on(Events.EPOCH_COMPLETED)
        def _log_eval_from_trainer(engine: Engine):
            if wb is None:
                return
            scalars = _cast_for_logging(engine.state.metrics or {})
            if not scalars:
                return
            ep = int(engine.state.epoch)
            payload = {"trainer/epoch": ep}
            payload.update({f"val/{k}": v for k, v in scalars.items()})
            _wb_log(wb, payload)

    # ========= Last-N checkpoints (every epoch) =========
    if model is not None:
        last_dir = os.path.join(out_dir, "checkpoints_last", run_tag)
        os.makedirs(last_dir, exist_ok=True)
        ckpt_last = ModelCheckpoint(
            dirname=last_dir,
            filename_prefix=last_model_prefix,
            n_saved=int(save_last_n),
            global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
            create_dir=True,
            require_empty=False,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt_last, {"model": model})

    # ========= Image logger (epoch-anchored, no step=) =========
    if log_images:
        try:
            seg_thr = float(cfg.get("seg_threshold", 0.5)) if isinstance(cfg, dict) else 0.5
            img_logger = make_image_logger(max_items=image_max_items, threshold=seg_thr, namespace="val")
        except Exception as e:
            img_logger = None
            logger.warning("image logger disabled: %s", e)

        if img_logger is not None and is_ev_engine and evaluator is not None:
            @evaluator.on(Events.ITERATION_COMPLETED)
            def _cache_last_payload(engine: Engine):
                pay = engine.state.output
                if not isinstance(pay, dict) or "image" not in pay:
                    pay = getattr(engine.state, "batch", None)
                engine.state._last_payload = pay

            @evaluator.on(Events.COMPLETED)
            def _log_eval_images(engine: Engine):
                if wb is None:
                    return
                ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
                payload = getattr(engine.state, "_last_payload", None)
                if payload is not None:
                    img_logger(payload, wandb_module=wb, epoch_anchor=ep)

        elif img_logger is not None:
            @trainer.on(Events.EPOCH_COMPLETED)
            def _log_train_images(engine: Engine):
                if wb is None:
                    return
                if int(engine.state.epoch) % int(image_every_n_epochs) != 0:
                    return
                payload = getattr(engine.state, "preview_payload", None) or getattr(engine.state, "preview_batch", None)
                if payload is not None:
                    img_logger(payload, wandb_module=wb, epoch_anchor=int(engine.state.epoch))


# # handlers.py
# from __future__ import annotations
# import os
# import logging
# from datetime import datetime
# from typing import Any, Dict, Iterable, Optional

# import numpy as np
# import torch
# from ignite.engine import Events, Engine
# from ignite.handlers import ModelCheckpoint, global_step_from_engine

# from image_logger import make_image_logger
# from metrics_utils import to_float_scalar, train_loss_output_transform
# from engine_utils import is_ignite_engine

# logger = logging.getLogger(__name__)


# def configure_wandb_step_semantics() -> None:
#     """
#     Predeclare W&B step domains so metric groups always use the right axis.
#     - trainer/epoch:    epoch-synchronous logs (validation, epoch aggregates)
#     - trainer/iteration:iteration-synchronous logs (per-batch training)
#     """
#     import wandb
#     try:
#         if getattr(wandb, "run", None) is None:
#             return  # nothing to configure yet
#         # Canonical step axes
#         wandb.define_metric("trainer/epoch")
#         wandb.define_metric("trainer/iteration")
#         # Route namespaces to the correct axis
#         wandb.define_metric("val/*", step_metric="trainer/epoch")
#         wandb.define_metric("eval/*", step_metric="trainer/epoch")
#         wandb.define_metric("train/*", step_metric="trainer/iteration")
#         wandb.define_metric("opt/*", step_metric="trainer/epoch")
#         # Seed axes once so they exist in the run
#         # wandb.log({"trainer/epoch": 0, "trainer/iteration": 0})
#     except Exception:
#         pass


# def _cast_for_logging(metrics: Dict[str, Any],
#                       int_prefixes=("cls_confmat_", "seg_confmat_", "tp", "tn", "fp", "fn")) -> Dict[str, Any]:
#     out: Dict[str, Any] = {}
#     for k, v in metrics.items():
#         # -> python scalar
#         if isinstance(v, torch.Tensor):
#             if v.numel() != 1:
#                 continue
#             v = v.detach().cpu().item()
#         elif isinstance(v, np.generic):
#             v = v.item()
#         # -> choose int vs float
#         if any(k.startswith(pref) for pref in int_prefixes):
#             out[k] = int(round(v))
#         else:
#             out[k] = float(v)
#     return out


# def _init_wandb_logger(project, run_name):
#     """Call make_wandb_logger with whatever signature it supports."""
#     import wandb
#     try:
#         # Initialize only if not already in a run
#         if getattr(wandb, "run", None) is None:
#             kwargs = {}
#             if project:
#                 kwargs["project"] = project
#             if run_name:
#                 kwargs["name"] = run_name
#             wandb.init(**kwargs)
#     except Exception as e:
#         # Non-fatal: many setups init elsewhere
#         logger.warning("W&B init skipped: %s", e)
#     return wandb


# def _wb_log(wb_mod, payload: dict, *, commit: Optional[bool] = None) -> None:
#     if wb_mod is None:
#         return
#     try:
#         if commit is None:
#             wb_mod.log(payload)
#         else:
#             wb_mod.log(payload, commit=commit)
#     except Exception as e:
#         logger.warning("W&B log failed: %s", e)


# def register_handlers(
#     *,
#     trainer: Engine,
#     evaluator: Optional[Engine],                  # may be None or non-Engine (two-pass case)
#     evaluator_cal: Optional[Engine] = None,      # unused here, kept for signature compat
#     model: Optional[torch.nn.Module] = None,
#     optimizer: Optional[torch.optim.Optimizer] = None,
#     out_dir: str = "./outputs",
#     last_model_prefix: str = "last",
#     best_model_prefix: str = "best",
#     save_last_n: int = 2,
#     watch_metric: str = "val_auc",
#     watch_mode: str = "max",
#     metric_names: Optional[Iterable[str]] = None,
#     log_images: bool = False,
#     image_every_n_epochs: int = 1,
#     image_max_items: int = 8,
#     wandb_project: Optional[str] = None,
#     wandb_run_name: Optional[str] = None,
#     wandb_enable: bool = True,
#     cfg: Optional[dict] = None,
#     console_iter_log: Optional[bool] = None,
#     console_epoch_log: Optional[bool] = None,
#     run_tag: Optional[str] = None,
#     seg_threshold: float = 0.5,
# ) -> None:
#     """
#     Attach training/eval logging, W&B, image logging, and checkpointing.
#     Works with either an Ignite evaluator Engine or a non-Engine two-pass evaluator
#     that publishes metrics into trainer.state.metrics.
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     run_tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")

#     # Resolve console log flags
#     cfg = cfg or {}
#     # iter_log = bool(console_iter_log if console_iter_log is not None else cfg.get("console_iter_log", False))
#     # epoch_log = bool(console_epoch_log if console_epoch_log is not None else cfg.get("console_epoch_log", True))

#     # W&B init (optional) + step semantics
#     wb = None
#     if wandb_enable:
#         try:
#             wb = _init_wandb_logger(wandb_project, wandb_run_name)
#             # ensure W&B “step domains” exist (epoch vs iteration)
#             configure_wandb_step_semantics()
#         except Exception as e:
#             wb = None
#             # don't crash training for logging failures
#             print(f"[WARN] W&B init failed: {e}")

#     # TRAIN LOOP LOGGING (W&B, per-iteration, step axis = trainer/iteration)
#     @trainer.on(Events.ITERATION_COMPLETED)
#     def _log_train_iter(engine: Engine):
#         if wb is None:
#             return
#         it = int(engine.state.iteration)
#         raw = train_loss_output_transform(engine.state.output) or {}
#         scalars = _cast_for_logging(raw)
#         payload: Dict[str, Any] = {"trainer/iteration": it}
#         payload.update({f"train/{k}": v for k, v in scalars.items()})
#         if optimizer is not None and optimizer.param_groups:
#             payload["train/lr"] = float(optimizer.param_groups[0].get("lr", 0.0))
#         _wb_log(wb, payload, commit=False)

#     @trainer.on(Events.EPOCH_COMPLETED)
#     def _commit_epoch_anchor(engine: Engine):
#         if wb is None:
#             return
#         ep = int(engine.state.epoch)
#         _wb_log(wb, {"trainer/epoch": ep})  # commit=True by default -> flushes buffered iter logs

#     # EVAL LOGGING + BEST CKPT
#     is_ev_engine = is_ignite_engine(evaluator)

#     if is_ev_engine and evaluator is not None:
#         # console eval stats
#         sel_keys = set(metric_names) if metric_names else None

#         def _subset(engine):
#             m = engine.state.metrics or {}
#             return {k: m[k] for k in sel_keys if k in m} if sel_keys else m

#         # _attach_stats(evaluator, tag="val")
#         @evaluator.on(Events.COMPLETED)
#         def _print_eval_metrics(engine):
#             ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
#             parts = []
#             for k, v in _cast_for_logging(engine.state.metrics or {}).items():
#                 if isinstance(v, int):
#                     parts.append(f"{k}: {v}")
#                 else:
#                     parts.append(f"{k}: {v:.4f}")
#             if parts:
#                 logger.info("Epoch[%d] Metrics -- %s", ep, " ".join(parts))

#         # @evaluator.on(Events.COMPLETED)
#         # def _print_eval_metrics(engine):
#         #     metrics = engine.state.metrics or {}
#         #     ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
#         #     int_prefixes = ("cls_confmat_", "seg_confmat_", "tp", "tn", "fp", "fn")
#         #     parts = []
#         #     for k, v in metrics.items():
#         #         # → scalar
#         #         if isinstance(v, torch.Tensor):
#         #             if v.numel() != 1:
#         #                 continue
#         #             v = v.detach().cpu().item()
#         #         elif isinstance(v, np.generic):
#         #             v = v.item()
#         #         # → format
#         #         if any(k.startswith(p) for p in int_prefixes):
#         #             parts.append(f"{k}: {int(round(v))}")
#         #         else:
#         #             parts.append(f"{k}: {float(v):.4f}")
#         #     if parts:
#         #         logger.info("Epoch[%d] Metrics -- %s", ep, " ".join(parts))

#         if wb is not None:
#             @evaluator.on(Events.COMPLETED)
#             def _log_eval(engine: Engine):
#                 if wb is None:
#                     return
#                 ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
#                 scalars = _cast_for_logging(engine.state.metrics or {})
#                 payload = {"trainer/epoch": ep}
#                 payload.update({f"val/{k}": v for k, v in scalars.items()})
#                 if len(payload) > 1:
#                     _wb_log(wb, payload)  # no step=

#             # @evaluator.on(Events.COMPLETED)
#             # def _log_eval(engine):
#             #     if wb is None:
#             #         return
#             #     metrics = engine.state.metrics or {}
#             #     ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
#             #     int_prefixes = ("cls_confmat_", "seg_confmat_", "tp", "tn", "fp", "fn")
#             #     payload = {"trainer/epoch": ep}  # keep epoch anchor
#             #     for k, v in metrics.items():
#             #         # → scalar
#             #         if isinstance(v, torch.Tensor):
#             #             if v.numel() != 1:
#             #                 continue
#             #             v = v.detach().cpu().item()
#             #         elif isinstance(v, np.generic):
#             #             v = v.item()
#             #         # → type for W&B
#             #         if any(k.startswith(p) for p in int_prefixes):
#             #             payload[f"val/{k}"] = int(round(v))
#             #         else:
#             #             payload[f"val/{k}"] = float(v)
#             #     if len(payload) > 1:
#             #         _wb_log(wb, payload)  # no step; rely on epoch axis

#         # choose a per-run folder
#         run_tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")

#         # best model saver (per-run subfolder, no require_empty crash)
#         if model is not None:
#             best_dir = os.path.join(out_dir, "checkpoints_best", run_tag)
#             os.makedirs(best_dir, exist_ok=True)

#             sign = 1.0 if str(watch_mode).lower() == "max" else -1.0

#             def _score_fn(e):
#                 m = e.state.metrics or {}
#                 v = to_float_scalar(m.get(watch_metric))
#                 return sign * float(v) if v is not None else -float("inf")

#             ckpt_best = ModelCheckpoint(
#                 dirname=best_dir,                     # <- use dirname, not save_handler
#                 filename_prefix=best_model_prefix,    # "best"
#                 n_saved=max(1, int(os.getenv("BEST_MODEL_RETENTION", "3"))),
#                 score_function=_score_fn,
#                 score_name=watch_metric,
#                 global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
#                 create_dir=True,
#                 require_empty=False,                  # <- don't crash if folder isn’t empty
#             )

#             # If we have an Ignite evaluator Engine:
#             if is_ignite_engine(evaluator) and evaluator is not None:
#                 evaluator.add_event_handler(Events.COMPLETED, ckpt_best, {"model": model})
#             else:
#                 # two-pass: save on trainer epoch complete when metrics live on trainer.state.metrics
#                 @trainer.on(Events.EPOCH_COMPLETED)
#                 def _maybe_save_best(engine):
#                     # Let ModelCheckpoint handle retention each epoch; it will save if _score_fn improved
#                     ckpt_best(engine, {"model": model})

#     if not is_ev_engine or evaluator is None:
#         @trainer.on(Events.EPOCH_COMPLETED)
#         def _log_eval_from_trainer(engine: Engine):
#             if wb is None:
#                 return
#             metrics = engine.state.metrics or {}
#             ep = int(engine.state.epoch)

#             int_prefixes = ("cls_confmat_", "seg_confmat_", "tp", "tn", "fp", "fn")
#             payload = {"trainer/epoch": ep}
#             for k, v in metrics.items():
#                 if isinstance(v, torch.Tensor):
#                     if v.numel() != 1:
#                         continue
#                     v = v.detach().cpu().item()
#                 elif hasattr(v, "item"):  # numpy scalar
#                     v = v.item()
#                 payload[f"val/{k}"] = int(round(v)) if any(k.startswith(p) for p in int_prefixes) else float(v)

#             if len(payload) > 1:
#                 _wb_log(wb, payload)  # no step=

#     # LAST-N CHECKPOINTS (every epoch)
#     if model is not None:
#         last_dir = os.path.join(out_dir, "checkpoints_last", run_tag)
#         os.makedirs(last_dir, exist_ok=True)

#         ckpt_last = ModelCheckpoint(
#             dirname=last_dir,                     # <- use dirname, not save_handler
#             filename_prefix=last_model_prefix,    # "last"
#             n_saved=int(save_last_n),
#             global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
#             create_dir=True,
#             require_empty=False,                  # <- avoid “already present” crash
#         )
#         trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt_last, {"model": model})

#     # IMAGE LOGGER
#     if log_images:
#         try:
#             # pull seg threshold if present
#             seg_thr = float(cfg.get("seg_threshold", 0.5)) if isinstance(cfg, dict) else 0.5
#             img_logger = make_image_logger(max_items=image_max_items, threshold=seg_thr, namespace="val")
#         except Exception as e:
#             img_logger = None
#             logger.warning("image logger disabled: %s", e)

#         if img_logger is not None and is_ev_engine and evaluator is not None:
#             # (optional) cache the last payload per iter so we can log after the epoch
#             @evaluator.on(Events.ITERATION_COMPLETED)
#             def _cache_last_payload(engine):
#                 # Prefer a full output dict (with predictions). Fallback to batch.
#                 pay = engine.state.output
#                 if not isinstance(pay, dict) or "image" not in pay:
#                     pay = getattr(engine.state, "batch", None)
#                 engine.state._last_payload = pay

#             @evaluator.on(Events.COMPLETED)
#             def _log_eval_images(engine):
#                 if wb is None:
#                     return
#                 ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
#                 payload = getattr(engine.state, "_last_payload", None)
#                 if payload is not None:
#                     # epoch-anchored, no step=
#                     img_logger(payload, wandb_module=wb, epoch_anchor=ep)

#         elif img_logger is not None:
#             # Fallback: log a preview captured by training loop
#             @trainer.on(Events.EPOCH_COMPLETED)
#             def _log_train_images(engine):
#                 if wb is None:
#                     return
#                 if int(engine.state.epoch) % int(image_every_n_epochs) != 0:
#                     return
#                 # Expect a dict with keys like {"image", "mask", "seg_logits"} or a batch dict
#                 payload = getattr(engine.state, "preview_payload", None) or getattr(engine.state, "preview_batch", None)
#                 if payload is not None:
#                     img_logger(payload, wandb_module=wb, epoch_anchor=int(engine.state.epoch))

#     # IMAGE LOGGER
#     # if log_images:
#     #     try:
#     #         # img_logger = make_image_logger(max_items=image_max_items)
#     #         # thresh = float((cfg or {}).get("seg_threshold", 0.5))  # default 0.5
#     #         img_logger = make_image_logger(max_items=image_max_items, threshold=seg_threshold)
#     #     except Exception as e:
#     #         img_logger = None
#     #         print(f"[WARN] image logger disabled: {e}")

#     #     if img_logger is not None and is_ev_engine and evaluator is not None:
#     #         @evaluator.on(Events.COMPLETED)
#     #         def _log_eval_images(engine):
#     #             if wb is None:
#     #                 return
#     #             ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
#     #             # Prefer engine.state.output if it contains predictions; else pass state.batch
#     #             payload_like = engine.state.output if isinstance(engine.state.output, dict) else engine.state.batch
#     #             if isinstance(payload_like, dict):
#     #                 img_logger(payload_like, wandb_module=wb, epoch_anchor=ep)

#             # @evaluator.on(Events.COMPLETED)
#             # def _log_eval_images(engine: Engine):
#             #     batch = getattr(engine.state, "batch", None)
#             #     if batch is not None:
#             #         ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
#             #         img_logger(batch, wandb_module=wb, epoch_anchor=ep)  # no step=
#         # else:
#         #     @trainer.on(Events.EPOCH_COMPLETED)
#         #     def _log_train_images(engine: Engine):
#         #         if int(engine.state.epoch) % int(image_every_n_epochs) != 0:
#         #             return
#         #         preview = getattr(engine.state, "preview_batch", None)
#         #         if preview is not None:
#         #             img_logger(preview, wandb_module=wb, epoch_anchor=int(engine.state.epoch))

#         # if img_logger is not None and is_ev_engine and evaluator is not None:
#         #     @evaluator.on(Events.COMPLETED)
#         #     def _log_eval_images(engine: Engine):
#         #         batch = getattr(engine.state, "batch", None)
#         #         if batch is not None:
#         #             ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
#         #             img_logger(batch, step=ep)      # keep step on epoch axis
#         # elif img_logger is not None:
#         #     @trainer.on(Events.EPOCH_COMPLETED)
#         #     def _log_train_images(engine: Engine):
#         #         if int(engine.state.epoch) % int(image_every_n_epochs) != 0:
#         #             return
#         #         preview = getattr(engine.state, "preview_batch", None)
#         #         if preview is not None:
#         #             img_logger(preview, step=int(engine.state.epoch))
