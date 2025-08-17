# handlers.py
import logging
logger = logging.getLogger(__name__)

import os
import time
import torch
import wandb
from monai.handlers import EarlyStopHandler, StatsHandler, from_engine
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver, ModelCheckpoint
from typing import Iterable, Optional, Dict, Any
import numpy as np

CHECKPOINT_DIR = "outputs/checkpoints"
CHECKPOINT_RETENTION = 3
CHECKPOINT_SAVE_EVERY = 1
BEST_MODEL_DIR = "outputs/best_model"
BEST_MODEL_RETENTION = 1


# handlers.py (or wherever _to_scalar_dict lives)
def _to_scalar_dict(
    metrics: Dict[str, Any],
    allow_vectors: bool = True,
    *,
    vector_sep: str = "_",
    log_confmat: bool = True,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    def norm(name: str) -> str:
        # val_acc -> val/acc; val_dice -> val/dice; train_acc -> train/acc
        if name.startswith("val_"):
            return "val/" + name[4:]
        if name.startswith("train_"):
            return "train/" + name[6:]
        return name

    def is_confmat_key(name: str) -> bool:
        n = name.lower()
        return "confmat" in n or "confusion" in n

    for k, v in metrics.items():
        nk = norm(k)
        # tensors → numpy
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            # 2D confusion matrices → emit cells + matrix under val/... prefix
            if log_confmat and v.ndim == 2 and is_confmat_key(k):
                cm = v if np.issubdtype(v.dtype, np.integer) else np.rint(v).astype(np.int64)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        out[f"{nk}_{i}{j}"] = int(cm[i, j])
                out[f"{nk}_matrix"] = cm.tolist()
                continue
            # scalar
            if v.ndim == 0:
                val = float(v.item())
                if np.isfinite(val):
                    out[nk] = val
                continue
            # vector → val/dice/0, val/iou/1, ...
            if allow_vectors and v.ndim == 1:
                for i, vi in enumerate(v.tolist()):
                    val = float(vi)
                    if np.isfinite(val):
                        out[f"{nk}{vector_sep}{i}"] = val
                continue
            # fallback: mean
            val = float(np.mean(v))
            if np.isfinite(val):
                out[nk] = val
            continue
        # plain numbers
        if isinstance(v, (float, int, np.floating, np.integer)):
            val = float(v)
            if np.isfinite(val):
                out[nk] = val
            continue
        # things with .item()
        try:
            val = float(v.item())
            if np.isfinite(val):
                out[nk] = val
        except Exception:
            pass

    return out


def make_wandb_logger(
    *,
    trainer=None,                 # Ignite trainer (for global_step + epoch fallback)
    evaluator=None,               # Ignite evaluator (for trainer_epoch)
    metric_names=None,            # subset to log; None = all
    step_by: str = "epoch",      # "global" | "epoch" | "omit"
    debug: bool = True,
):
    """
    Curried Ignite handler: flattens engine.state.metrics and logs to W&B.
    Use step_by="epoch" for validation logs (avoids step-reset warnings).
    """

    # def _handler(engine):
    #     # epoch/global_step
    #     epoch = int(getattr(
    #         evaluator.state if evaluator is not None else engine.state,
    #         "trainer_epoch",
    #         getattr(trainer.state if trainer is not None else engine.state, "epoch", 0),
    #     ))
    #     global_step = int(getattr(
    #         trainer.state if trainer is not None else engine.state,
    #         "iteration",
    #         0,
    #     ))
    #     # pick metrics
    #     m = dict(engine.state.metrics or {})

    #     # synthesize macro scalar dice from vector if needed
    #     def _mean_of(prefix: str):
    #         # accepts both slash and underscore indexed forms
    #         vals = []
    #         for k in (f"{prefix}", f"{prefix}/0", f"{prefix}/1", f"{prefix}_0", f"{prefix}_1"):
    #             v = m.get(k)
    #             if v is None:
    #                 continue
    #             if torch.is_tensor(v): v = v.detach().cpu().numpy()
    #             try:
    #                 if isinstance(v, (list, tuple)): vals += [float(x) for x in v]
    #                 elif hasattr(v, "ndim") and v.ndim == 1: vals += [float(x) for x in v.tolist()]
    #                 else: vals.append(float(v))
    #             except Exception:
    #                 pass
    #         return (sum(vals) / len(vals)) if vals else None

    #     mw = float(getattr(wandb.config, "multi_weight", 0.65))
    #     auc  = m.get("val/auc", m.get("val_auc"))
    #     if auc is not None and dice is not None:
    #         # pick one namespace and stay consistent; here we use slash
    #         m["val/multi"] = mw * float(dice) + (1.0 - mw) * float(auc)

    #     dice = _mean_of("val/dice") or _mean_of("val_dice")
    #     if dice is not None:
    #         m["val/dice"] = float(dice)

    #     iou = _mean_of("val/iou") or _mean_of("val_iou")
    #     if iou is not None:
    #         m["val/iou"] = float(iou)

    #     prec = _mean_of("val/prec") or _mean_of("val_prec")
    #     if prec is not None:
    #         m["val/prec"] = float(prec)

    #     recall = _mean_of("val/recall") or _mean_of("val_recall")
    #     if recall is not None:
    #         m["val/recall"] = float(recall)

    #     if metric_names:
    #         m = {k: m[k] for k in metric_names if k in m}
    #     flat = _to_scalar_dict(m, allow_vectors=True)

    #     # create scalar sweep target from the per-class values
    #     def add_macro_mean(prefix: str):
    #         a, b = f"{prefix}/0", f"{prefix}/1"
    #         if a in flat and b in flat and prefix not in flat:
    #             flat[prefix] = 0.5 * (flat[a] + flat[b])

    #     for k in ("val/multi", "val/dice", "val/iou", "val/prec", "val/recall"):
    #         add_macro_mean(k)

    #     payload = {"epoch": epoch, "global_step": global_step, **flat}
    #     if step_by == "omit":
    #         step_arg = {}
    #     else:
    #         step = global_step if step_by == "global" else epoch
    #         step_arg = {"step": step}
    #     if debug:
    #         logger.info("wandb log: step_by=%s, step=%s, keys=%s",
    #                     step_by, step_arg.get("step"), list(flat.keys()))
    #     try:
    #         wandb.log(payload, **step_arg)
    #     except Exception as e:
    #         logger.warning("wandb.log failed: %s", e)

    # return _handler

    def _handler(engine):
        # epoch/global_step
        epoch = int(getattr(
            evaluator.state if evaluator is not None else engine.state,
            "trainer_epoch",
            getattr(trainer.state if trainer is not None else engine.state, "epoch", 0),
        ))
        global_step = int(getattr(
            trainer.state if trainer is not None else engine.state,
            "iteration",
            0,
        ))

        # copy metrics
        m = dict(engine.state.metrics or {})

        # helper: mean of vector metric supporting both namespaces + shapes
        def _mean_of(prefix: str):
            # accepts both slash and underscore indexed forms and vector tensors/arrays
            vals = []
            for k in (f"{prefix}", f"{prefix}/0", f"{prefix}/1", f"{prefix}_0", f"{prefix}_1"):
                v = m.get(k)
                if v is None:
                    continue
                if torch.is_tensor(v):
                    v = v.detach().cpu().numpy()
                try:
                    # vector-like
                    if isinstance(v, (list, tuple)):
                        vals += [float(x) for x in v]
                    elif hasattr(v, "ndim") and getattr(v, "ndim", 0) == 1:
                        vals += [float(x) for x in v.tolist()]
                    else:
                        vals.append(float(v))
                except Exception:
                    pass
            return (sum(vals) / len(vals)) if vals else None

        # synthesize scalar aggregates
        dice_mean = _mean_of("val/dice") or _mean_of("val_dice")
        iou_mean = _mean_of("val/iou") or _mean_of("val_iou")
        prec_mean = _mean_of("val/prec") or _mean_of("val_prec")
        recall_mean = _mean_of("val/recall") or _mean_of("val_recall")

        if dice_mean is not None:
            m["val/dice"] = float(dice_mean)
        if iou_mean is not None:
            m["val/iou"] = float(iou_mean)
        if prec_mean is not None:
            m["val/prec"] = float(prec_mean)
        if recall_mean is not None:
            m["val/recall"] = float(recall_mean)

        # synthesize scalar val/multi from dice + auc (prefer slash namespace if present)
        mw = float(getattr(wandb.config, "multi_weight", 0.65))
        auc = m.get("val/auc", m.get("val_auc"))
        if auc is not None and dice_mean is not None:
            m["val/multi"] = mw * float(dice_mean) + (1.0 - mw) * float(auc)

        # optional subset
        if metric_names:
            m = {k: m[k] for k in metric_names if k in m}

        # flatten (emits per-class as val/dice/0 etc., keeps scalar keys above)
        flat = _to_scalar_dict(m, allow_vectors=True)

        # payload + step control
        payload = {"epoch": epoch, "global_step": global_step, **flat}
        if step_by == "omit":
            step_arg = {}
        else:
            step_arg = {"step": (global_step if step_by == "global" else epoch)}

        if debug:
            logger.info("wandb log: step_by=%s, step=%s, keys=%s",
                        step_by, step_arg.get("step"), list(flat.keys()))
        try:
            wandb.log(payload, **step_arg)
        except Exception as e:
            logger.warning("wandb.log failed: %s", e)
        return _handler


def register_handlers(
    trainer,
    evaluator,
    model,
    cfg: dict,
    *,
    optimizer=None,
    scheduler=None,
    train_loader=None,
    val_loader=None,
    wandb_log_handler=None,
    metric_names: Optional[Iterable[str]] = None,
    **kwargs,
):
    # Debug flags
    global_debug = bool(os.environ.get("DEBUG", "")) or bool(cfg.get("debug", False))
    debug_handlers = bool(cfg.get("debug_handlers", global_debug))
    log_images = bool(cfg.get("log_images", True))

    arch_name = str(cfg.get("architecture", "model")).lower().replace("/", "_").replace(" ", "_")
    checkpoint_prefix = f"{arch_name}"
    best_model_prefix = f"{arch_name}_best"

    # Report metrics to log
    if debug_handlers:
        planned = list(metric_names) if metric_names else "(all evaluator metrics)"
        logger.info("[register_handlers] Will log metrics: %s", planned)

    # Stats (epoch-level)
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

    # Run evaluator once per training epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def _run_eval_after_epoch(engine):
        evaluator.state.trainer_epoch = engine.state.epoch  # used by loggers/ckpt
        evaluator.state.trainer_iteration = engine.state.iteration
        evaluator.run()

    # Image logger
    if log_images:
        attach_image_logger = make_image_logger(
            num_images=int(cfg.get("log_images_n", 4)),
            threshold=float(cfg.get("mask_threshold", 0.5)),
        )
        attach_image_logger(evaluator)

    # W&B metrics logger
    @trainer.on(Events.ITERATION_COMPLETED)
    def _log_train_iter(engine):
        out = getattr(engine.state, "output", None)

        def _to_float(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                t = x.detach().float()
                return float(t.mean().item()) if t.numel() > 0 else None
            if isinstance(x, (float, int)):
                return float(x)
            return None

        loss_val = None
        if isinstance(out, dict):
            v = out.get("loss")
            if v is None:
                # try any key containing "loss"
                for k, vv in out.items():
                    if isinstance(k, str) and "loss" in k.lower():
                        v = vv
                        break
            loss_val = _to_float(v)
        elif isinstance(out, (list, tuple)):
            for v in out:  # pick the first floatable item
                loss_val = _to_float(v)
                if loss_val is not None:
                    break
        else:
            loss_val = _to_float(out)

        payload = {"global_step": engine.state.iteration}
        if loss_val is not None:
            payload["train/loss"] = loss_val
        if optimizer is not None and optimizer.param_groups:
            payload["train/lr"] = float(optimizer.param_groups[0]["lr"])

        if payload:
            wandb.log(payload, step=engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _log_epoch_lr(engine):
        try:
            lr = float(optimizer.param_groups[0]["lr"])
            gs = int(engine.state.iteration)
            wandb.log({"val/epoch_lr": lr, "epoch": engine.state.epoch, "global_step": gs}, step=gs)
        except Exception:
            pass

    evaluator.add_event_handler(
        Events.COMPLETED,
        make_wandb_logger(trainer=trainer, evaluator=evaluator, metric_names=metric_names, step_by="global")
    )

    # Early stopping
    es_cfg = cfg.get("early_stop", {}) or {}
    watch_metric = es_cfg.get("metric", "val_auc")
    mode = (es_cfg.get("mode", "max") or "max").lower()
    patience = int(es_cfg.get("patience", 10))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    min_epochs_stop = int(es_cfg.get("min_epochs", 3))

    def score_fn(engine):
        val = engine.state.metrics.get(watch_metric, None)
        if val is None:
            return float("-inf") if mode == "max" else float("inf")
        return float(val) if mode == "max" else -float(val)
    if patience > 0:
        early_stopper = EarlyStopHandler(
            patience=patience,
            min_delta=min_delta,
            score_function=score_fn,
            trainer=trainer,
        )

        @evaluator.on(Events.EPOCH_COMPLETED)
        def _gated_early_stop(engine):
            epoch = int(getattr(trainer.state, "epoch", 0))
            if epoch >= min_epochs_stop:
                early_stopper(engine)

    # Checkpoints
    run_id = str(cfg.get("run_id") or (wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")))
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_id)
    best_dir = os.path.join(BEST_MODEL_DIR, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    to_save_periodic = {"model": model}
    if optimizer is not None:
        to_save_periodic["optimizer"] = optimizer
    if scheduler is not None:
        to_save_periodic["lr_scheduler"] = scheduler
    # Trainer’s Engine also has state_dict/load_state_dict – great for resuming epoch/iter
    to_save_periodic["trainer"] = trainer
    disk_saver = DiskSaver(ckpt_dir, create_dir=True, require_empty=False)
    periodic = Checkpoint(
        to_save=to_save_periodic,
        save_handler=disk_saver,
        n_saved=CHECKPOINT_RETENTION,
        filename_prefix=checkpoint_prefix,
        global_step_transform=lambda e, _: int(e.state.epoch),
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=CHECKPOINT_SAVE_EVERY), periodic)

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
    """Log evaluator metrics to W&B using epoch as the step; include global_step field."""
    epoch = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
    # epoch = int(getattr(engine.state, "trainer_epoch", 0) or getattr(engine.state, "epoch", 0))
    gstep = int(getattr(engine.state, "trainer_iteration", 0))
    flat = _to_scalar_dict(engine.state.metrics, allow_vectors=True)
    if debug:
        logger.info("wandb_log_handler: epoch=%s keys=%s", epoch, list(flat.keys()))
    wandb.log({"epoch": epoch, "global_step": gstep, **flat}, step=epoch)


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
        step = int(
            getattr(engine.state, "trainer_iteration",
                    getattr(engine, "state", None) and getattr(engine.state, "iteration", 0))
        )
        for i in range(len(buf_imgs)):
            wandb.log(
                {
                    "image": wandb.Image(_to_numpy_image(buf_imgs[i]), caption="input"),
                    "gt_mask": wandb.Image(_to_numpy_mask(buf_gt[i])),
                    "pred_mask": wandb.Image(_to_numpy_mask(buf_pred[i])),
                    # "epoch": step,
                    "epoch": int(getattr(engine.state, "trainer_epoch", engine.state.epoch)),
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
