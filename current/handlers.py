# handlers.py
import logging
import torch
import wandb
from monai.handlers import EarlyStopHandler, StatsHandler, from_engine
# from monai.data.meta_tensor import MetaTensor
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, DiskSaver
# from metrics_utils import attach_metrics

CHECKPOINT_DIR = "outputs/checkpoints"
CHECKPOINT_RETENTION = 3
CHECKPOINT_SAVE_EVERY = 1
BEST_MODEL_DIR = "outputs/best_model"
BEST_MODEL_RETENTION = 1


class SafeDiskSaver(DiskSaver):
    def remove(self, filename):
        try:
            super().remove(filename)
        except FileNotFoundError:
            print(f"WARNING: Checkpoint file not found for deletion: {filename}")


def register_handlers(
    trainer,
    evaluator,
    model,
    config,
    train_loader=None,
    val_loader=None,
    wandb_log_handler=None,
    prepare_batch=None,
    **handler_kwargs
):
    arch_name = str(config.get("architecture", "model")).lower().replace("/", "_").replace(" ", "_")
    checkpoint_prefix = f"{arch_name}"
    best_model_prefix = f"{arch_name}_best"

    # DEBUG: What metrics are attached?
    print("\n[DEBUG register_handlers] Metrics registered on evaluator at setup:")
    for k in getattr(evaluator, "_metrics", {}).keys():
        print(f"  {k}")
    print("------")

    # Training loss stats
    StatsHandler(
        tag_name="train",
        output_transform=from_engine(["loss"], first=True),
        iteration_log=False,
        epoch_log=True
    ).attach(trainer)

    # Validation metrics stats (always keys from config)
    val_metrics = handler_kwargs.get("val_metrics", ["val_acc", "val_auc", "val_loss"])  # fallback to common metrics
    StatsHandler(
        tag_name="val",
        output_transform=from_engine(val_metrics, first=False),
        iteration_log=False
    ).attach(evaluator)

    # DEBUG: Print metrics every eval epoch
    def debug_metrics_printer(engine):
        print(f"[DEBUG @EPOCH_COMPLETED] engine.state.metrics: {engine.state.metrics}")
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, debug_metrics_printer)

    # Validation after each epoch
    def run_evaluator_with_epoch(engine):
        # Run evaluator and pass the trainer's epoch as an argument
        print(f"[DEBUG] Calling evaluator.run() after epoch {engine.state.epoch}")
        evaluator.state.trainer_epoch = engine.state.epoch  # synchronize evaluator with global epoch
        evaluator.run()
    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator_with_epoch)

    attach_image_logger = make_image_logger(num_images=4, threshold=0.5)
    attach_image_logger(evaluator)
    if wandb_log_handler:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: wandb_log_handler(engine))

    def get_score_function():
        metric = config.get("early_stop", {}).get("metric", "val_auc")
        mode = config.get("early_stop", {}).get("mode", "max")
        if mode == "max":
            return lambda engine: engine.state.metrics.get(metric, 0.0)
        else:
            return lambda engine: -engine.state.metrics.get(metric, 0.0)

    early_stopper = EarlyStopHandler(
        patience=config["early_stop"]["patience"],
        min_delta=config["early_stop"]["min_delta"],
        score_function=get_score_function(),
        trainer=trainer
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

    # Checkpoint Saving (every N epochs)
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_handler = ModelCheckpoint(
        dirname=CHECKPOINT_DIR,
        filename_prefix=checkpoint_prefix,
        n_saved=CHECKPOINT_RETENTION,
        create_dir=True,
        require_empty=False,
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=CHECKPOINT_SAVE_EVERY))
    def save_model(engine):
        checkpoint_handler(engine, {"model": model})

    print(f"[INFO] Checkpoints will be saved to '{CHECKPOINT_DIR}' every {CHECKPOINT_SAVE_EVERY} epoch(s), keeping the last {CHECKPOINT_RETENTION}.")

    # Best Model Saving (by val_auc or handler kwarg key)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    best_score_metric = config.get("early_stop", {}).get("metric", "val_auc")
    best_ckpt_handler = ModelCheckpoint(
        dirname=BEST_MODEL_DIR,
        filename_prefix=best_model_prefix,
        n_saved=BEST_MODEL_RETENTION,
        score_function=lambda engine: engine.state.metrics.get(best_score_metric, 0.0),
        score_name=best_score_metric,
        global_step_transform=lambda e, _: e.state.epoch,
        require_empty=False
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_ckpt_handler, {"model": model})


def wandb_log_handler(engine):
    """Log all available metrics from engine.state.metrics to wandb, handling types robustly."""
    print("wandb_log_handler metrics (before logging):", engine.state.metrics)
    # Just in case, print type and content for further diagnosis
    print("wandb_log_handler engine.state.metrics.keys():", list(engine.state.metrics.keys()))

    # Try to get trainer's epoch if present, else fallback
    epoch = getattr(engine.state, "trainer_epoch", engine.state.epoch)
    log_data = {}
    print("\n[wandb_log_handler] metrics dict:")
    for key, value in engine.state.metrics.items():
        print(f"[wandb_log_handler] key: {key}, value: {value}  (type={type(value)})")
        try:
            if hasattr(value, "as_tensor"):
                value = value.as_tensor()
            if isinstance(value, torch.Tensor):
                value = value.cpu().detach()
                if value.ndim == 2 and value.dtype in (torch.int32, torch.int64):  # Confusion matrix
                    for i in range(value.shape[0]):
                        for j in range(value.shape[1]):
                            log_data[f"{key}_{i}{j}"] = int(value[i, j])
                    log_data[key] = value.tolist()
                elif value.numel() > 1:
                    log_data[key] = float(value.mean().item()) if value.dtype.is_floating_point else value.tolist()
                elif value.numel() == 1:
                    log_data[key] = float(value.item())
                else:
                    log_data[key] = value.tolist()
            elif hasattr(value, "item"):
                log_data[key] = float(value.item())
            else:
                log_data[key] = float(value)
        except Exception as e:
            logging.warning(f"[wandb_log_handler] Could not log {key}: {value} - {e}")

    val_confmat = engine.state.metrics.get("val_cls_confmat", None)
    if val_confmat is not None:
        print("[DEBUG] val_cls_confmat value:", val_confmat)
        print("[DEBUG] sum of confmat entries:", sum([sum(row) for row in val_confmat.tolist()]))

    log_data["epoch"] = epoch
    print(f"[wandb_log_handler] Final log_data dict: {log_data}")
    wandb.log(log_data)


def make_image_logger(num_images: int = 4, threshold: float = 0.5):
    """Collect (image, gt_mask, pred_mask) from engine.state.output and log to W&B."""
    import torch
    import numpy as np
    import wandb
    from ignite.engine import Events
    from monai.data.meta_tensor import MetaTensor

    buf_imgs, buf_gt, buf_pred = [], [], []

    def _as_tensor(x):
        return x.as_tensor() if isinstance(x, MetaTensor) else x

    def _squeeze01(x):
        # [B,1,H,W] -> [B,H,W]
        return x[:, 0] if (x.ndim == 4 and x.shape[1] == 1) else x

    def _to_numpy_image(x: torch.Tensor) -> np.ndarray:
        """
        Accepts [C,H,W] or [H,W] torch tensor, returns HxW or HxWx3 numpy float in [0,1].
        We convert to numpy FIRST so wandb.Image won't try to permute a 2D torch tensor.
        """
        t = x.detach().cpu()
        if t.ndim == 3:  # CHW
            c, h, w = t.shape
            if c == 1:
                arr = t[0]  # HxW (grayscale)
            else:
                arr = t[:3].permute(1, 2, 0)  # HWC (RGB-ish)
        elif t.ndim == 2:  # HxW
            arr = t
        else:
            # Fallback: pick first channel then treat as HxW
            arr = t.view(-1, *t.shape[-2:])[0]

        arr = arr.float()
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = arr * 0.0
        return arr.numpy()

    def _to_numpy_mask(x: torch.Tensor) -> np.ndarray:
        """Accepts [H,W] (float 0/1 or logits). Returns uint8 HxW in {0,1}."""
        t = x.detach().cpu()
        if t.ndim == 3:
            t = t[0]  # CHW->HW if needed
        # ensure binary 0/1 for logging; keep as numpy 2D so W&B won't permute
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

        # Pred mask: binary or multiclass -> [B,H,W] float 0/1 or class ids
        if logits.ndim == 4 and logits.shape[1] == 1:
            pmask = (torch.sigmoid(logits) > threshold).float()[:, 0]
        elif logits.ndim == 4 and logits.shape[1] > 1:
            pmask = torch.softmax(logits, dim=1).argmax(dim=1).float()
        elif logits.ndim == 3:
            pmask = (torch.sigmoid(logits) > threshold).float()
        else:
            return None

        masks = _squeeze01(masks)  # [B,H,W]
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

        for i in range(len(buf_imgs)):
            img_np = _to_numpy_image(buf_imgs[i])
            gt_np = _to_numpy_mask(buf_gt[i])
            pr_np = _to_numpy_mask(buf_pred[i])

            wandb.log(
                {
                    "image": wandb.Image(img_np, caption="input"),
                    "gt_mask": wandb.Image(gt_np),
                    "pred_mask": wandb.Image(pr_np),
                },
                step=engine.state.iteration,
            )

        buf_imgs.clear()
        buf_gt.clear()
        buf_pred.clear()

    def attach(evaluator):
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, on_iteration)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, on_epoch_completed)
        return evaluator

    return attach
