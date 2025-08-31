# handlers.py
from __future__ import annotations
import os
import logging
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import torch
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, global_step_from_engine

from image_logger import make_image_logger
from metrics_utils import train_loss_output_transform
from utils.safe import to_float_scalar, is_finite_float

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
    """Unify scalar extraction with NaN guards."""
    f = to_float_scalar(v, strict=False)
    return f if is_finite_float(f) else None


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
    watch_metric: str = "val/auc",               # <- use val/auc by default
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
    def _flatten(d, prefix=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
        return out

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
            logger.warning("[WARN] W&B init failed: %s", e)

    # -------- train logging (iteration + epoch anchor) --------
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

    @trainer.on(Events.EPOCH_COMPLETED)
    def _commit_epoch_anchor(engine: Engine):
        if wb is None:
            return
        _wb_log(wb, {"trainer/epoch": int(engine.state.epoch)})

    # -------- eval logging (inline attach_wandb_eval_logging) --------
    def _with_val_prefix(d: Dict[str, Any]) -> Dict[str, Any]:
        if not d:
            return {}
        out: Dict[str, Any] = {}
        for k, v in d.items():
            out[k if (isinstance(k, str) and k.startswith("val/")) else f"val/{k}"] = v
        return out

    def _is_engine(x: Any) -> bool:
        try:
            return isinstance(x, Engine)
        except Exception:
            return False

    is_ev_engine = _is_engine(evaluator)

    if is_ev_engine and evaluator is not None:
        # idempotent guard so we don't attach twice
        if getattr(evaluator, "_wandb_eval_logging_attached", False):
            logger.debug("W&B eval logging already attached; skipping duplicate attach.")
        else:
            setattr(evaluator, "_wandb_eval_logging_attached", True)

            # console pretty-print
            @evaluator.on(Events.COMPLETED)
            def _print_eval_metrics(engine: Engine):
                ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
                scalars = _cast_for_logging(engine.state.metrics or {})
                if scalars:
                    parts = [f"{k}: {v}" if isinstance(v, int) else f"{k}: {v:.4f}" for k, v in scalars.items()]
                    logger.info("Epoch[%d] Metrics -- %s", ep, " ".join(parts))

            # wandb log on epoch axis + mirror into trainer.state.metrics
            if wb is not None:
                @evaluator.on(Events.COMPLETED)
                def _log_eval(engine: Engine):
                    prefixed = _with_val_prefix(engine.state.metrics or {})
                    flat = _flatten(prefixed)
                    scalars = _cast_for_logging(flat)
                    if not scalars:
                        return
                    ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
                    # mirror: make val/* visible to trainer-based schedulers/checkpoints
                    try:
                        if getattr(trainer.state, "metrics", None) is None:
                            trainer.state.metrics = {}
                        trainer.state.metrics.update(scalars)
                    except Exception:
                        pass
                    payload = {"trainer/epoch": ep}
                    payload.update(scalars)  # already val/* keys
                    _wb_log(wb, payload)

        # best checkpoint on evaluator completion
        if model is not None:
            best_dir = os.path.join(out_dir, "checkpoints_best", run_tag)
            os.makedirs(best_dir, exist_ok=True)

            sign = 1.0 if str(watch_mode).lower() == "max" else -1.0

            def _score_fn(e: Engine):
                m = e.state.metrics or {}
                v = m.get(watch_metric)
                if v is None and watch_metric.startswith("val/"):
                    v = m.get(watch_metric[4:])
                if v is None and not watch_metric.startswith("val/"):
                    v = m.get(f"val/{watch_metric}")
                v = to_float_scalar(v, strict=False)
                return sign * float(v) if is_finite_float(v) else -float("inf")

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

    # two-pass path: eval metrics live on trainer.state.metrics; log them at epoch end
    if (not is_ev_engine) or (evaluator is None):
        @trainer.on(Events.EPOCH_COMPLETED)
        def _log_eval_from_trainer(engine: Engine):
            if wb is None:
                return
            prefixed = _with_val_prefix(engine.state.metrics or {})
            flat = _flatten(prefixed)
            scalars = _cast_for_logging(flat)
            if not scalars:
                return
            ep = int(engine.state.epoch)
            payload = {"trainer/epoch": ep}
            payload.update(scalars)
            _wb_log(wb, payload)

    # last-N checkpoints each epoch
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

    # image logging (optional)
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
