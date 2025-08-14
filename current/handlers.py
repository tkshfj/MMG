# handlers.py
import logging
logger = logging.getLogger(__name__)

import os
import time
import torch
import wandb
from monai.handlers import EarlyStopHandler, StatsHandler, from_engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from typing import Iterable, Optional, Dict, Any
import numpy as np

CHECKPOINT_DIR = "outputs/checkpoints"
CHECKPOINT_RETENTION = 3
CHECKPOINT_SAVE_EVERY = 1
BEST_MODEL_DIR = "outputs/best_model"
BEST_MODEL_RETENTION = 1


def _to_scalar_dict(metrics: Dict[str, Any], allow_vectors=True) -> Dict[str, float]:
    out = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                out[k] = float(v.item())
            elif allow_vectors and v.ndim == 1:
                for i, vi in enumerate(v.tolist()):
                    out[f"{k}/{i}"] = float(vi)
            else:
                out[k] = float(v.mean())
        elif isinstance(v, (float, int)):
            out[k] = float(v)
        else:
            try:
                out[k] = float(v.item())
            except Exception:
                continue
    return out


def register_handlers(
    trainer,
    evaluator,
    model,
    cfg: dict,
    *,
    train_loader=None,
    val_loader=None,
    wandb_log_handler=None,                 # optional external callback
    metric_names: Optional[Iterable[str]] = None,
    **kwargs,
):
    # ---- Debug flags --------------------------------------------------------
    global_debug = bool(os.environ.get("DEBUG", "")) or bool(cfg.get("debug", False))
    debug_handlers = bool(cfg.get("debug_handlers", global_debug))
    log_images = bool(cfg.get("log_images", True))
    # -------------------------------------------------------------------------

    arch_name = str(cfg.get("architecture", "model")).lower().replace("/", "_").replace(" ", "_")
    checkpoint_prefix = f"{arch_name}"
    best_model_prefix = f"{arch_name}_best"

    # Report what WE plan to log
    if debug_handlers:
        planned = list(metric_names) if metric_names else "(all evaluator metrics)"
        logger.info("[register_handlers] Will log metrics: %s", planned)

    # --- Stats (epoch-level) -------------------------------------------------
    StatsHandler(
        tag_name="train",
        output_transform=from_engine(["loss"], first=True),
        iteration_log=False,
        epoch_log=True,
    ).attach(trainer)

    val_keys = list(metric_names) if metric_names else ["val_acc", "val_auc", "val_loss"]
    StatsHandler(
        tag_name="val",
        output_transform=from_engine(val_keys, first=False),
        iteration_log=False,
    ).attach(evaluator)

    if debug_handlers:
        @evaluator.on(Events.EPOCH_COMPLETED)
        def _debug_metrics(engine):
            tepoch = getattr(engine.state, "trainer_epoch", engine.state.epoch)
            logger.info("[evaluator] epoch=%d metrics=%s", tepoch, engine.state.metrics)

    # --- Run evaluator once per training epoch ------------------------------
    @trainer.on(Events.EPOCH_COMPLETED)
    def _run_eval_after_epoch(engine):
        evaluator.state.trainer_epoch = engine.state.epoch  # used by loggers/ckpt
        evaluator.run()

    # --- Optional image logger ----------------------------------------------
    if log_images:
        attach_image_logger = make_image_logger(
            num_images=int(cfg.get("log_images_n", 4)),
            threshold=float(cfg.get("mask_threshold", 0.5)),
        )
        attach_image_logger(evaluator)

    # --- W&B metrics logger (single hook; single step convention) -----------
    @evaluator.on(Events.COMPLETED)
    def _log_eval(engine):
        # Use the *training* epoch as the global step for ALL eval logs
        step = int(getattr(evaluator.state, "trainer_epoch", getattr(trainer.state, "epoch", 0)))
        keys = list(metric_names) if metric_names else list(engine.state.metrics.keys())
        data = {k: engine.state.metrics[k] for k in keys if k in engine.state.metrics}
        flat = _to_scalar_dict(data, allow_vectors=True)

        # If user passed a custom handler, let it run (best effort)
        if callable(wandb_log_handler):
            try:
                wandb_log_handler(engine)
            except TypeError:
                try:
                    wandb_log_handler(epoch=step, metrics=flat, raw=engine.state.metrics)
                except TypeError:
                    try:
                        wandb_log_handler(metrics=flat, step=step, raw=engine.state.metrics)
                    except TypeError:
                        try:
                            wandb_log_handler({"epoch": step, **flat})
                        except TypeError:
                            wandb_log_handler(flat)

        # Always log a minimal, consistent payload ourselves
        wandb.log({"epoch": step, **flat}, step=step)

    # --- Early stopping ------------------------------------------------------
    es_cfg = cfg.get("early_stop", {}) or {}
    watch_metric = es_cfg.get("metric", "val_auc")
    mode = (es_cfg.get("mode", "max") or "max").lower()
    patience = int(es_cfg.get("patience", 10))
    min_delta = float(es_cfg.get("min_delta", 0.0))

    def score_fn(engine):
        val = engine.state.metrics.get(watch_metric, None)
        if val is None:
            return float("-inf") if mode == "max" else float("inf")
        return float(val) if mode == "max" else -float(val)

    early_stopper = EarlyStopHandler(
        patience=patience,
        min_delta=min_delta,
        score_function=score_fn,
        trainer=trainer,
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

    # --- Checkpoints ---------------------------------------------------------
    run_id = str(cfg.get("run_id") or (wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")))
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_id)
    best_dir = os.path.join(BEST_MODEL_DIR, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    periodic_ckpt = ModelCheckpoint(
        ckpt_dir,
        filename_prefix=checkpoint_prefix,
        n_saved=CHECKPOINT_RETENTION,
        create_dir=True,
        require_empty=False,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=CHECKPOINT_SAVE_EVERY), periodic_ckpt, {"model": model})
    logger.info(
        "Checkpoints -> %s (every %d epoch%s), keeping last %d.",
        ckpt_dir, CHECKPOINT_SAVE_EVERY, "s" if CHECKPOINT_SAVE_EVERY != 1 else "", CHECKPOINT_RETENTION,
    )

    best_ckpt = ModelCheckpoint(
        best_dir,
        filename_prefix=best_model_prefix,
        n_saved=BEST_MODEL_RETENTION,
        score_function=lambda e: float(e.state.metrics.get(watch_metric, float("-inf"))),
        score_name=watch_metric,
        global_step_transform=lambda e, _: int(getattr(e.state, "trainer_epoch", e.state.epoch)),
        create_dir=True,
        require_empty=False,
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_ckpt, {"model": model})


def wandb_log_handler(engine, debug: bool = False):
    """Optional external logger: logs engine.state.metrics → W&B with ‘epoch’ step."""
    epoch = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
    if debug:
        logger.info("wandb_log_handler: epoch=%s keys=%s", epoch, list(engine.state.metrics.keys()))

    log_data: Dict[str, Any] = {}
    for key, value in engine.state.metrics.items():
        try:
            # Handle tensors & confusion matrices nicely
            if hasattr(value, "as_tensor"):
                value = value.as_tensor()
            if isinstance(value, torch.Tensor):
                t = value.detach().cpu()
                if t.ndim == 2 and ("confmat" in key.lower() or "confusion" in key.lower()):
                    cm = t.to(torch.int64) if not t.dtype.is_floating_point else t.round().to(torch.int64)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            log_data[f"{key}_{i}{j}"] = int(cm[i, j])
                    log_data[key] = cm.tolist()
                elif t.numel() == 1:
                    log_data[key] = float(t.item())
                elif t.dtype.is_floating_point:
                    log_data[key] = float(t.mean().item())
                else:
                    log_data[key] = t.tolist()
            elif hasattr(value, "item"):
                log_data[key] = float(value.item())
            else:
                log_data[key] = float(value)
        except Exception as e:
            logging.warning("wandb_log_handler: could not log %s (%s): %s", key, type(value), e)

    log_data["epoch"] = epoch
    wandb.log(log_data, step=epoch)


def make_image_logger(num_images: int = 4, threshold: float = 0.5):
    """
    Collect (image, gt_mask, pred_mask) from evaluator.state.output and log to W&B.
    Uses the SAME epoch step as scalar metric logs to avoid step ordering issues.
    """
    import numpy as np
    from ignite.engine import Events
    from monai.data.meta_tensor import MetaTensor

    buf_imgs, buf_gt, buf_pred = [], [], []

    def _as_tensor(x):
        return x.as_tensor() if isinstance(x, MetaTensor) else x

    def _squeeze01(x):
        return x[:, 0] if (x.ndim == 4 and x.shape[1] == 1) else x

    def _to_numpy_image(x: torch.Tensor) -> np.ndarray:
        t = x.detach().cpu()
        if t.ndim == 3:
            c, h, w = t.shape
            arr = t[0] if c == 1 else t[:3].permute(1, 2, 0)
        elif t.ndim == 2:
            arr = t
        else:
            arr = t.view(-1, *t.shape[-2:])[0]
        arr = arr.float()
        mn, mx = float(arr.min()), float(arr.max())
        arr = (arr - mn) / (mx - mn) if mx > mn else arr * 0.0
        return arr.numpy()

    def _to_numpy_mask(x: torch.Tensor) -> np.ndarray:
        t = x.detach().cpu()
        if t.ndim == 3:
            t = t[0]
        t = (t > 0.5).to(torch.uint8)
        return t.numpy()

    def _collect_from_output(out):
        if not isinstance(out, dict):
            return None
        images = _as_tensor(out.get("image"))
        label = out.get("label", {})
        pred = out.get("pred", {})
        if not (isinstance(label, dict) and isinstance(pred, dict)):
            return None
        if "mask" not in label or "seg_out" not in pred:
            return None

        masks = _as_tensor(label["mask"])
        logits = _as_tensor(pred["seg_out"])

        if logits.ndim == 4 and logits.shape[1] == 1:
            pmask = (torch.sigmoid(logits) > threshold).float()[:, 0]
        elif logits.ndim == 4 and logits.shape[1] > 1:
            pmask = torch.softmax(logits, dim=1).argmax(dim=1).float()
        elif logits.ndim == 3:
            pmask = (torch.sigmoid(logits) > threshold).float()
        else:
            return None

        masks = _squeeze01(masks)
        return images, masks, pmask

    def on_iteration(engine):
        nonlocal buf_imgs, buf_gt, buf_pred
        if len(buf_imgs) >= num_images:
            return
        pack = _collect_from_output(engine.state.output)
        if pack is None:
            return
        images, masks, pmask = pack
        b = min(images.shape[0], num_images - len(buf_imgs))
        for i in range(b):
            buf_imgs.append(images[i])
            buf_gt.append(masks[i])
            buf_pred.append(pmask[i])

    def on_epoch_completed(engine):
        nonlocal buf_imgs, buf_gt, buf_pred
        if not buf_imgs:
            return
        # Use the same epoch step as scalar metrics
        step = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
        for i in range(len(buf_imgs)):
            wandb.log(
                {
                    "image": wandb.Image(_to_numpy_image(buf_imgs[i]), caption="input"),
                    "gt_mask": wandb.Image(_to_numpy_mask(buf_gt[i])),
                    "pred_mask": wandb.Image(_to_numpy_mask(buf_pred[i])),
                    "epoch": step,
                },
                step=step,
            )
        buf_imgs.clear()
        buf_gt.clear()
        buf_pred.clear()

    def attach(evaluator):
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, on_iteration)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, on_epoch_completed)
        return evaluator

    return attach
