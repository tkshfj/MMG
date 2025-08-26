# handlers.py
from __future__ import annotations
import os
import logging
from typing import Optional, Iterable, Callable, Any, Dict
import inspect

import numpy as np
import torch
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from monai.handlers import StatsHandler

from image_logger import make_image_logger
from metrics_utils import to_float_scalar, train_loss_output_transform
from engine_utils import is_ignite_engine

logger = logging.getLogger(__name__)


def _make_score_fn(metric_key: str, mode: str = "max") -> Callable[[Engine], float]:
    sign = 1.0 if str(mode).lower() == "max" else -1.0

    def _score(engine: Engine) -> float:
        val = engine.state.metrics.get(metric_key)
        f = to_float_scalar(val, strict=True)
        return sign * (f if f is not None else float("nan"))
    return _score


def _init_wandb_logger(project, run_name):
    """Call make_wandb_logger with whatever signature it supports."""
    import wandb
    try:
        # Initialize only if not already in a run
        if getattr(wandb, "run", None) is None:
            kwargs = {}
            if project:
                kwargs["project"] = project
            if run_name:
                kwargs["name"] = run_name
            wandb.init(**kwargs)
    except Exception as e:
        # Non-fatal: many setups init elsewhere
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


def _attach_stats(
    engine: Engine,
    *,
    tag: str,
    output_transform=None,
    iteration_log: bool = True,
    epoch_log: bool = True,
) -> None:
    """Attach MONAI StatsHandler with compatibility for sig changes (tag/name, iteration_log/epoch_log availability)."""
    try:
        sh_sig = inspect.signature(StatsHandler).parameters
        sh_kwargs: Dict[str, Any] = {}

        # Older MONAI used tag=, newer uses name=
        if "tag" in sh_sig:
            sh_kwargs["tag"] = tag
        elif "name" in sh_sig:
            sh_kwargs["name"] = tag

        if output_transform is not None:
            sh_kwargs["output_transform"] = output_transform

        # Respect iteration/epoch logging flags if supported
        if "iteration_log" in sh_sig:
            sh_kwargs["iteration_log"] = bool(iteration_log)
        if "epoch_log" in sh_sig:
            sh_kwargs["epoch_log"] = bool(epoch_log)

        StatsHandler(**sh_kwargs).attach(engine)
    except Exception as e:
        logger.warning("StatsHandler attach failed for %s: %s", tag, e)


# def _attach_stats(engine, tag: str, output_transform=None):
#     """Attach MONAI StatsHandler; handle name/tag arg across versions."""
#     try:
#         sh_kwargs = {}
#         # Older MONAI used tag=, newer uses name=
#         if "tag" in inspect.signature(StatsHandler).parameters:
#             sh_kwargs["tag"] = tag
#         elif "name" in inspect.signature(StatsHandler).parameters:
#             sh_kwargs["name"] = tag
#         if output_transform is not None:
#             sh_kwargs["output_transform"] = output_transform
#         StatsHandler(**sh_kwargs).attach(engine)
#     except Exception as e:
#         logger.warning("StatsHandler attach failed for %s: %s", tag, e)


def register_handlers(
    *,
    trainer: Engine,
    evaluator: Optional[Engine],  # may be None or non-Engine (two-pass case)
    evaluator_cal: Optional[Engine] = None,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    out_dir: str = "./outputs",
    last_model_prefix: str = "model",
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
) -> None:
    """
    Attach training/eval logging, W&B, image logging, and checkpointing.
    Works with either an Ignite evaluator Engine or a non-Engine two-pass evaluator
    that publishes metrics into trainer.state.metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Resolve console log flags
    cfg = cfg or {}
    iter_log = bool(console_iter_log if console_iter_log is not None else cfg.get("console_iter_log", False))
    epoch_log = bool(console_epoch_log if console_epoch_log is not None else cfg.get("console_epoch_log", True))

    # W&B logger (optional)
    wb = None
    if wandb_enable:
        try:
            # wb = make_wandb_logger(project=wandb_project, name=wandb_run_name)
            wb = _init_wandb_logger(wandb_project, wandb_run_name) if wandb_enable else None
        except Exception as e:
            logger.warning("W&B init failed: %s", e)

    # TRAIN LOOP LOGGING (console)
    _attach_stats(
        trainer,
        tag="train",
        output_transform=lambda out: out.get("loss", out),
        iteration_log=iter_log,
        epoch_log=epoch_log,
    )

    # TRAIN LOOP LOGGING
    # StatsHandler prints loss/metrics to stdout (nice for local dev)
    # _attach_stats(trainer, tag="train", output_transform=train_loss_output_transform)
    # _attach_stats(trainer, tag="train", output_transform=lambda out: out.get("loss", out))

    @trainer.on(Events.ITERATION_COMPLETED)
    def _log_train_iter(engine: Engine):
        if wb is None:
            return
        gs = int(engine.state.iteration)
        metrics = train_loss_output_transform(engine.state.output)
        payload: Dict[str, Any] = {
            "global_step": gs,
            **{f"train/{k}": v for k, v in metrics.items()},
        }
        if optimizer is not None and optimizer.param_groups:
            payload["train/lr"] = float(optimizer.param_groups[0].get("lr", 0.0))
        _wb_log(wb, payload, commit=False)

    # EVAL LOGGING + BEST CKPT
    is_ev_engine = is_ignite_engine(evaluator)

    # Ignite evaluator present: attach printing + W&B + best checkpoint to evaluator
    if is_ev_engine and evaluator is not None:
        # stdout metrics
        sel_keys = set(metric_names) if metric_names else None

        def _subset(engine):
            m = engine.state.metrics or {}
            return {k: m[k] for k in sel_keys if k in m} if sel_keys else m
        if evaluator is not None and is_ignite_engine(evaluator):
            # Let StatsHandler handle eval metrics printing on its default events
            _attach_stats(evaluator, tag="val")

        if wb is not None:
            @evaluator.on(Events.COMPLETED)
            def _log_eval(engine: Engine):
                metrics = engine.state.metrics or {}
                # step = int(getattr(engine.state, "trainer_iteration", 0))
                # Val logs: use epoch only (no step=)
                ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
                clean = {}
                for k, v in metrics.items():
                    f = to_float_scalar(v)
                    if f is not None:
                        clean[f"val/{k}"] = f
                if clean:
                    clean["epoch"] = ep
                    _wb_log(wb, clean)
                    # _wb_log(wb, clean, step=step)

        # Best checkpoint from evaluator metrics
        if model is not None:
            score_fn = _make_score_fn(watch_metric, watch_mode)
            best_dir = os.path.join(out_dir, "checkpoints_best")
            os.makedirs(best_dir, exist_ok=True)
            best_ckpt = ModelCheckpoint(
                best_dir,
                filename_prefix=best_model_prefix,
                n_saved=max(1, int(os.getenv("BEST_MODEL_RETENTION", "3"))),
                score_function=score_fn,
                score_name=watch_metric,
                global_step_transform=lambda e, *_: int(getattr(e.state, "trainer_epoch", e.state.epoch)),
                create_dir=True,
                require_empty=False,
            )
            evaluator.add_event_handler(Events.COMPLETED, best_ckpt, {"model": model})

    # No Ignite evaluator: read metrics injected into trainer.state.metrics
    else:
        # Two-pass case: metrics live on trainer.state.metrics
        @trainer.on(Events.EPOCH_COMPLETED)
        def _log_eval_from_trainer(engine: Engine):
            if wb is None:
                return
            metrics = engine.state.metrics or {}
            clean = {}
            for k, v in metrics.items():
                f = to_float_scalar(v)
                if f is not None:
                    clean[f"val/{k}"] = f
            if clean:
                # _wb_log(wb, clean, step=int(engine.state.iteration))
                # Val logs (two-pass case): use epoch only
                clean["epoch"] = int(engine.state.epoch)
                _wb_log(wb, clean)

        if model is not None:
            score_key = watch_metric
            sign = 1.0 if str(watch_mode).lower() == "max" else -1.0
            best_state = {"score": -np.inf if sign > 0 else np.inf}

            @trainer.on(Events.EPOCH_COMPLETED)
            def _maybe_save_best(engine: Engine):
                m = engine.state.metrics or {}
                val = to_float_scalar(m.get(score_key))
                if val is None:
                    return
                score = sign * val
                prev = best_state["score"]
                if (score > prev) if sign > 0 else (score < prev):
                    best_state["score"] = score
                    path = os.path.join(out_dir, "checkpoints_best")
                    os.makedirs(path, exist_ok=True)
                    saver = ModelCheckpoint(
                        dirname=path,
                        filename_prefix=best_model_prefix,
                        n_saved=max(1, int(os.getenv("BEST_MODEL_RETENTION", "3"))),
                        create_dir=True,
                        require_empty=False,
                    )
                    saver(engine, {"model": model})

    # LAST-N CHECKPOINTS
    if model is not None:
        last_dir = os.path.join(out_dir, "checkpoints_last")
        os.makedirs(last_dir, exist_ok=True)
        last_ckpt = ModelCheckpoint(
            last_dir,
            filename_prefix=last_model_prefix,
            n_saved=int(save_last_n),
            create_dir=True,
            require_empty=False,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, last_ckpt, {"model": model})

    # IMAGE LOGGER
    if log_images:
        try:
            img_logger = make_image_logger(max_items=image_max_items)
        except Exception as e:
            img_logger = None
            logger.warning("image logger disabled: %s", e)

        if img_logger is not None and is_ev_engine and evaluator is not None:
            @evaluator.on(Events.COMPLETED)
            def _log_eval_images(engine: Engine):
                # expects evaluator.state.batch or a cached preview
                batch = getattr(engine.state, "batch", None)
                if batch is not None:
                    # img_logger(batch, step=int(getattr(engine.state, "trainer_iteration", 0)))
                    ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
                    img_logger(batch, step=ep)
        elif img_logger is not None:
            # Fallback: sample every N epochs from trainer (requires a small hook in your loop to stash a preview)
            @trainer.on(Events.EPOCH_COMPLETED)
            def _log_train_images(engine: Engine):
                if int(engine.state.epoch) % int(image_every_n_epochs) != 0:
                    return
                preview = getattr(engine.state, "preview_batch", None)
                if preview is not None:
                    # img_logger(preview, step=int(engine.state.iteration))
                    img_logger(preview, step=int(engine.state.epoch))
