# handlers.py
import logging
import torch
import wandb
from monai.handlers import EarlyStopHandler, StatsHandler, from_engine  # CheckpointSaver
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, DiskSaver

# Configurable save interval and retention
CHECKPOINT_DIR = "outputs/checkpoints"
CHECKPOINT_PREFIX = ""
CHECKPOINT_RETENTION = 3         # Number of checkpoints to keep
CHECKPOINT_SAVE_EVERY = 1        # Save every N epochs (set to 1 for every epoch)

class SafeDiskSaver(DiskSaver):
    def remove(self, filename):
        try:
            super().remove(filename)
        except FileNotFoundError:
            print(f"WARNING: Checkpoint file not found for deletion: {filename}")


# Attach Dice and Jaccard metrics via ConfusionMatrix if requested
def attach_segmentation_metrics(evaluator, num_classes=2, output_transform=None, dice_name="val_dice", iou_name="val_iou"):
    from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
    # evaluator, num_classes, output_transform, dice_name, iou_name
    cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=output_transform)
    DiceCoefficient(cm=cm_metric).attach(evaluator, dice_name)
    JaccardIndex(cm=cm_metric).attach(evaluator, iou_name)


def register_handlers(
    trainer,
    evaluator,
    model,
    config,
    train_loader=None,
    val_loader=None,
    manual_dice_handler=None,
    image_log_handler=None,
    wandb_log_handler=None,
    add_segmentation_metrics=False,
    num_classes=2,
    seg_output_transform=None,
    dice_name="val_dice",
    iou_name="val_iou",
    prepare_batch=None
):
    # Attach all event handlers to trainer/evaluator: logging, checkpoint, early stop, etc.
    if add_segmentation_metrics and seg_output_transform is not None:
        attach_segmentation_metrics(
            evaluator,
            num_classes=num_classes,
            output_transform=seg_output_transform,
            dice_name=dice_name,
            iou_name=iou_name
        )

    # Training loss stats (per-iteration and per-epoch)
    StatsHandler(
        tag_name="train",
        output_transform=from_engine(["loss"], first=True),
        iteration_log=False,
        epoch_log=True
    ).attach(trainer)

    def get_val_metric_keys(config, add_segmentation_metrics=False, dice_name="val_dice", iou_name="val_iou"):
        """Return a list of metric keys for StatsHandler output_transform, based on config['task'] and metrics attached."""
        keys = []
        # Classification metrics
        if config.get("task") in ("classification", "multitask"):
            keys += ["val_acc", "val_auc", "val_loss", "val_cls_confmat"]
        # Segmentation metrics
        if config.get("task") in ("segmentation", "multitask") and add_segmentation_metrics:
            keys += [dice_name, iou_name]
        return keys

    # Validation metrics at epoch end
    val_metrics = get_val_metric_keys(config, add_segmentation_metrics, dice_name, iou_name)

    StatsHandler(
        tag_name="val",
        output_transform=from_engine(val_metrics, first=False),
        iteration_log=False
    ).attach(evaluator)

    # Run validation after each training epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: evaluator.run())

    # Log manual dice and sample images after each validation epoch (handlers must be provided)
    if manual_dice_handler and prepare_batch is not None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, manual_dice_handler(model, prepare_batch))
    if image_log_handler and prepare_batch is not None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, image_log_handler(model, prepare_batch))

    # Log metrics to wandb
    if wandb_log_handler:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log_handler)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, wandb_log_handler)

    # evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda e: print("Eval metrics at epoch end:", list(e.state.metrics.keys())))

    def get_score_function(config):
        key = config["early_stop"]["metric"]
        mode = config["early_stop"].get("mode", "max")
        if mode == "max":
            return lambda engine: engine.state.metrics.get(key, 0.0)
        else:
            # For 'min' metrics (like loss), invert value so improvement is still detected correctly
            return lambda engine: -engine.state.metrics.get(key, 0.0)

    # Early Stopping
    early_stopper = EarlyStopHandler(
        patience=config["early_stop"]["patience"],
        min_delta=config["early_stop"]["min_delta"],
        score_function=get_score_function(config),
        trainer=trainer
        # patience=10,
        # min_delta=0.001,
        # score_function=lambda engine: engine.state.metrics.get("val_auc", 0.0)
    )
    # early_stopper.set_trainer(trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

    # Checkpoint Saving (Every N Epochs)
    import os
    os.makedirs('outputs/checkpoints', exist_ok=True)

    checkpoint_handler = ModelCheckpoint(
        dirname=CHECKPOINT_DIR,
        filename_prefix=CHECKPOINT_PREFIX,
        n_saved=CHECKPOINT_RETENTION,
        create_dir=True,
        require_empty=False,
        save_as_state_dict=True,
    )

    # Register handler to save every N epochs
    @trainer.on(Events.EPOCH_COMPLETED(every=CHECKPOINT_SAVE_EVERY))
    def save_model(engine):
        checkpoint_handler(engine, {"model": model})

    print(f"[INFO] Checkpoints will be saved to '{CHECKPOINT_DIR}' every {CHECKPOINT_SAVE_EVERY} epoch(s), keeping the last {CHECKPOINT_RETENTION}.")

    # @trainer.on(Events.EPOCH_COMPLETED(every=1))
    # def save_model(engine):
    #     checkpoint_handler(engine, {"model": model})

    # Best Model Saving (by val_auc)
    os.makedirs('outputs/best_model', exist_ok=True)
    best_ckpt_handler = ModelCheckpoint(
        dirname="outputs/best_model",
        filename_prefix="best_val_auc",
        n_saved=1,
        score_function=lambda engine: engine.state.metrics.get("val_auc", 0.0),
        score_name="val_auc",
        global_step_transform=lambda e, _: e.state.epoch,
        require_empty=False
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_ckpt_handler, {"model": model})


def wandb_log_handler(engine):
    """Log all available metrics from engine.state.metrics to wandb, handling types robustly."""
    log_data = {}
    for key, value in engine.state.metrics.items():
        try:
            # Handle MetaTensor (MONAI) to Tensor
            if hasattr(value, "as_tensor"):
                value = value.as_tensor()
            # Handle torch.Tensor
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
            # Handle numpy scalars
            elif hasattr(value, "item"):
                log_data[key] = float(value.item())
            else:
                log_data[key] = float(value)
        except Exception as e:
            logging.warning(f"[wandb_log_handler] Could not log {key}: {value} - {e}")
    log_data["epoch"] = engine.state.epoch
    wandb.log(log_data)


def image_log_handler(model, prepare_batch_fn, num_images=4):
    def _handler(engine):
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            all_images = []
            all_gt_masks = []
            all_pred_masks = []
            for batch in engine.state.dataloader:
                images, targets = prepare_batch_fn(batch, device)

                # Only proceed if "mask" exists in targets (segmentation or multitask)
                if "mask" not in targets:
                    continue  # skip this batch

                # Model may be multitask or just segmentation
                if isinstance(model(images), (tuple, list)) and len(model(images)) > 1:
                    _, seg_logits = model(images)
                else:
                    seg_logits = model(images)

                pred_mask = (torch.sigmoid(seg_logits) > 0.5).float()
                gt_mask = targets["mask"]

                for i in range(len(images)):
                    all_images.append(images[i].cpu())
                    all_gt_masks.append(gt_mask[i].cpu())
                    all_pred_masks.append(pred_mask[i].cpu())
                    if len(all_images) >= num_images:
                        break
                if len(all_images) >= num_images:
                    break

            if len(all_images) == 0:
                # Optionally log or warn that no images/masks were available for logging
                return

            for i in range(min(len(all_images), num_images)):
                wandb.log({
                    "image": wandb.Image(all_images[i], caption="GT vs Pred"),
                    "gt_mask": wandb.Image(all_gt_masks[i]),
                    "pred_mask": wandb.Image(all_pred_masks[i]),
                })
    return _handler


def manual_dice_handler(model, prepare_batch_fn):
    """
    Ignite handler to manually compute and log Dice score for segmentation.
    Logs as 'val_dice_manual' to W&B at the end of each evaluation epoch.
    """
    logger = logging.getLogger(__name__)

    def handler(engine):
        val_loader = engine.state.dataloader
        device = engine.state.device
        model.eval()
        dice_scores = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = prepare_batch_fn(batch, device=device, non_blocking=False)
                label = batch.get("label", {})
                if not isinstance(label, dict) or "mask" not in label:
                    logger.warning("Skipping batch without 'mask' in nested label dict (classification-only batch).")
                    continue

                outputs = model(inputs)
                _, seg_output = outputs
                pred_mask = (torch.sigmoid(seg_output) > 0.5).float()
                true_mask = targets["mask"]

                intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
                union = pred_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3))
                dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
                dice_scores.extend(dice.cpu().tolist())

        if dice_scores:
            avg_dice = sum(dice_scores) / len(dice_scores)
            # Log using desired key and associate with validation epoch
            wandb.log({"val_dice_manual": avg_dice}, step=engine.state.epoch)
        else:
            logger.warning("No masks found for manual dice calculation.")

    return handler
