# handlers.py
import torch
import wandb
from monai.handlers import CheckpointSaver, EarlyStopHandler, StatsHandler, from_engine
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events


# Attach Dice and Jaccard metrics via ConfusionMatrix if requested
def attach_segmentation_metrics(evaluator, num_classes=2, output_transform=None, dice_name="val_dice", iou_name="val_iou"):
    from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
    # evaluator, num_classes, output_transform, dice_name, iou_name
    cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=output_transform)
    DiceCoefficient(cm=cm_metric).attach(evaluator, dice_name)
    JaccardIndex(cm=cm_metric).attach(evaluator, iou_name)


def register_handlers(
    trainer, evaluator, model, config,
    train_loader=None, val_loader=None,
    manual_dice_handler=None, image_log_handler=None, wandb_log_handler=None,
    add_segmentation_metrics=False,
    num_classes=2, seg_output_transform=None, dice_name="val_dice", iou_name="val_iou",
    prepare_batch=None
):
    # Attach all event handlers to trainer/evaluator: logging, checkpoint, early stop, etc.
    if add_segmentation_metrics and seg_output_transform is not None:
        attach_segmentation_metrics(evaluator, num_classes=num_classes, output_transform=seg_output_transform, dice_name=dice_name, iou_name=iou_name)

    # Training loss stats (per-iteration and per-epoch)
    StatsHandler(
        tag_name="train",
        output_transform=from_engine(["loss"], first=True)
    ).attach(trainer)

    # Validation metrics at epoch end
    val_metrics = ["val_acc", "val_auc", "val_loss"]
    if add_segmentation_metrics:
        val_metrics += [dice_name, iou_name]

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

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda e: print("Eval metrics at epoch end:", list(e.state.metrics.keys())))

    # Early Stopping
    early_stopper = EarlyStopHandler(
        patience=10,
        min_delta=0.001,
        score_function=lambda engine: engine.state.metrics.get("val_auc", 0.0)
    )
    early_stopper.set_trainer(trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

    # Checkpoint handler (periodic)
    ckpt_saver = CheckpointSaver(
        save_dir="outputs/checkpoints",
        save_dict={"model": model},
        save_interval=1,
        n_saved=2,
    )
    ckpt_saver.attach(trainer)

    # Save best model by AUC
    model_ckpt = ModelCheckpoint(
        dirname="outputs/best_model",
        filename_prefix="best_val_auc",
        n_saved=1,
        score_function=lambda engine: engine.state.metrics.get("val_auc", 0.0),
        score_name="val_auc",
        global_step_transform=lambda e, _: e.state.epoch,
        require_empty=False
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, model_ckpt, {"model": model})


# Handlers for logging to Weights & Biases and image logging
def wandb_log_handler(engine):
    """Log metrics from engine.state.metrics to Weights & Biases."""
    log_data = {}
    for k, v in engine.state.metrics.items():
        try:
            # Handle torch tensors or numpy values
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach()
                if v.numel() == 1:
                    log_data[k] = v.item()
                else:
                    # Log classwise: val_mean_dice_c0, val_mean_dice_c1, etc.
                    for i, val in enumerate(v.flatten()):
                        log_data[f"{k}_c{i}"] = float(val)
                    # Log mean over classes as the main key (e.g., val_mean_dice, val_iou)
                    log_data[k] = float(v.mean().item())
            else:
                log_data[k] = float(v)
        except Exception as e:
            print(f"[wandb_log_handler] Warning: Could not log {k}: {v} - {e}")
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
                class_logits, seg_logits = model(images)
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

            for i in range(min(len(all_images), num_images)):
                wandb.log({
                    "image": wandb.Image(all_images[i], caption="GT vs Pred"),
                    "gt_mask": wandb.Image(all_gt_masks[i]),
                    "pred_mask": wandb.Image(all_pred_masks[i]),
                })
    return _handler


def manual_dice_handler(model, prepare_batch_fn):
    """
    Returns an Ignite handler that manually computes the Dice score
    for segmentation outputs at the end of each evaluation epoch.
    """
    def _handler(engine):
        model.eval()
        device = next(model.parameters()).device

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in engine.state.dataloader:
                images, targets = prepare_batch_fn(batch, device)
                class_logits, seg_out = model(images)

                # Apply sigmoid → binary thresholding
                pred_mask = (torch.sigmoid(seg_out) > 0.5).float()
                if pred_mask.ndim == 3:
                    pred_mask = pred_mask.unsqueeze(1)

                gt_mask = targets["mask"]
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask.unsqueeze(1)

                all_preds.append(pred_mask.cpu())
                all_targets.append(gt_mask.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute Dice coefficient
        intersection = (all_preds * all_targets).sum(dim=(1, 2, 3))
        union = all_preds.sum(dim=(1, 2, 3)) + all_targets.sum(dim=(1, 2, 3))
        dice_scores = (2. * intersection / (union + 1e-8)).numpy()
        mean_dice = dice_scores.mean()

        print(f"[Manual Dice] Mean Dice over validation set: {mean_dice:.4f}")

        engine.state.metrics["val_dice_manual"] = mean_dice
        if wandb.run is not None:
            # Do **not** pass step=engine.state.epoch here, let wandb auto-increment step
            wandb.log({"val_dice_manual": mean_dice})

    return _handler
