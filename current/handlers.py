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

    # Training logs (per-iteration; commit=False)
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

    # Eval logging + best checkpoint
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

    # Last-N checkpoints (every epoch)
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

    # Image logger (epoch-anchored, no step=)
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
