# handlers.py
# import logging
# import torch
# import wandb
# from monai.handlers import EarlyStopHandler, StatsHandler, from_engine
# from ignite.engine import Events
# from ignite.handlers import ModelCheckpoint, DiskSaver

# CHECKPOINT_DIR = "outputs/checkpoints"
# CHECKPOINT_RETENTION = 3
# CHECKPOINT_SAVE_EVERY = 1
# BEST_MODEL_DIR = "outputs/best_model"
# BEST_MODEL_RETENTION = 1


# class SafeDiskSaver(DiskSaver):
#     def remove(self, filename):
#         try:
#             super().remove(filename)
#         except FileNotFoundError:
#             print(f"WARNING: Checkpoint file not found for deletion: {filename}")


# def attach_segmentation_metrics(evaluator, num_classes=2, output_transform=None, dice_name="val_dice", iou_name="val_iou"):
#     from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
#     cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=output_transform)
#     DiceCoefficient(cm=cm_metric).attach(evaluator, dice_name)
#     JaccardIndex(cm=cm_metric).attach(evaluator, iou_name)


# def register_handlers(
#     trainer,
#     evaluator,
#     model,
#     config,
#     train_loader=None,
#     val_loader=None,
#     manual_dice_handler=None,
#     image_log_handler=None,
#     wandb_log_handler=None,
#     prepare_batch=None,
#     **handler_kwargs  # Everything from model_class.get_handler_kwargs()
# ):
#     arch_name = str(config.get("architecture", "model")).lower().replace("/", "_").replace(" ", "_")
#     checkpoint_prefix = f"{arch_name}"
#     best_model_prefix = f"{arch_name}_best"

#     # Attach segmentation metrics if requested
#     if handler_kwargs.get("add_segmentation_metrics", False) and handler_kwargs.get("seg_output_transform", None) is not None:
#         attach_segmentation_metrics(
#             evaluator,
#             num_classes=handler_kwargs.get("num_classes", 2),
#             output_transform=handler_kwargs["seg_output_transform"],
#             dice_name=handler_kwargs.get("dice_name", "val_dice"),
#             iou_name=handler_kwargs.get("iou_name", "val_iou"),
#         )

#     # Training stats
#     StatsHandler(
#         tag_name="train",
#         output_transform=from_engine(["loss"], first=True),
#         iteration_log=False,
#         epoch_log=True
#     ).attach(trainer)

#     # Validation stats (metric keys from handler kwargs or default)
#     val_metrics = handler_kwargs.get("val_metrics", ["val_acc", "val_auc", "val_loss"])
#     StatsHandler(
#         tag_name="val",
#         output_transform=from_engine(val_metrics, first=False),
#         iteration_log=False
#     ).attach(evaluator)

#     # Ensure evaluator runs after every epoch, and that it knows the trainer's epoch
#     def run_evaluator_with_epoch(engine):
#         evaluator.state.trainer_epoch = engine.state.epoch  # Pass trainer's epoch for wandb step sync
#         evaluator.run()

#     trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator_with_epoch)

#     # Manual dice/image handlers
#     if manual_dice_handler and prepare_batch is not None:
#         evaluator.add_event_handler(Events.EPOCH_COMPLETED, manual_dice_handler(model, prepare_batch))
#     if image_log_handler and prepare_batch is not None:
#         evaluator.add_event_handler(Events.EPOCH_COMPLETED, image_log_handler(model, prepare_batch))
#     if wandb_log_handler:
#         # Use trainer's epoch for WandB logging
#         evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: wandb_log_handler(engine, epoch=getattr(engine.state, "trainer_epoch", engine.state.epoch)))  # noqa: E501

#     # Early stopping
#     def get_score_function():
#         metric = config.get("early_stop", {}).get("metric", "val_auc")
#         mode = config.get("early_stop", {}).get("mode", "max")
#         if mode == "max":
#             return lambda engine: engine.state.metrics.get(metric, 0.0)
#         else:
#             return lambda engine: -engine.state.metrics.get(metric, 0.0)

#     early_stopper = EarlyStopHandler(
#         patience=config["early_stop"]["patience"],
#         min_delta=config["early_stop"]["min_delta"],
#         score_function=get_score_function(),
#         trainer=trainer
#     )
#     evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

#     # Checkpoint saving
#     import os
#     os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#     checkpoint_handler = ModelCheckpoint(
#         dirname=CHECKPOINT_DIR,
#         filename_prefix=checkpoint_prefix,
#         n_saved=CHECKPOINT_RETENTION,
#         create_dir=True,
#         require_empty=False,
#     )

#     @trainer.on(Events.EPOCH_COMPLETED(every=CHECKPOINT_SAVE_EVERY))
#     def save_model(engine):
#         checkpoint_handler(engine, {"model": model})

#     print(f"[INFO] Checkpoints will be saved to '{CHECKPOINT_DIR}' every {CHECKPOINT_SAVE_EVERY} epoch(s), keeping the last {CHECKPOINT_RETENTION}.")

#     # Best model saving
#     os.makedirs(BEST_MODEL_DIR, exist_ok=True)
#     best_score_metric = config.get("early_stop", {}).get("metric", "val_auc")
#     best_ckpt_handler = ModelCheckpoint(
#         dirname=BEST_MODEL_DIR,
#         filename_prefix=best_model_prefix,
#         n_saved=BEST_MODEL_RETENTION,
#         score_function=lambda engine: engine.state.metrics.get(best_score_metric, 0.0),
#         score_name=best_score_metric,
#         global_step_transform=lambda e, _: getattr(e.state, "trainer_epoch", e.state.epoch),
#         require_empty=False
#     )
#     evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_ckpt_handler, {"model": model})


# def wandb_log_handler(engine, epoch=None):
#     """Log all available metrics from engine.state.metrics to wandb, handling types robustly."""
#     print("wandb_log_handler metrics:", engine.state.metrics)
#     epoch = epoch if epoch is not None else getattr(engine.state, "trainer_epoch", engine.state.epoch)
#     log_data = {}
#     for key, value in engine.state.metrics.items():
#         print(f"[wandb_log_handler] key: {key}, value: {value}")
#         try:
#             if hasattr(value, "as_tensor"):
#                 value = value.as_tensor()
#             if isinstance(value, torch.Tensor):
#                 value = value.cpu().detach()
#                 if value.ndim == 2 and value.dtype in (torch.int32, torch.int64):
#                     for i in range(value.shape[0]):
#                         for j in range(value.shape[1]):
#                             log_data[f"{key}_{i}{j}"] = int(value[i, j])
#                     log_data[key] = value.tolist()
#                 elif value.numel() > 1:
#                     log_data[key] = float(value.mean().item()) if value.dtype.is_floating_point else value.tolist()
#                 elif value.numel() == 1:
#                     log_data[key] = float(value.item())
#                 else:
#                     log_data[key] = value.tolist()
#             elif hasattr(value, "item"):
#                 log_data[key] = float(value.item())
#             else:
#                 log_data[key] = float(value)
#         except Exception as e:
#             logging.warning(f"[wandb_log_handler] Could not log {key}: {value} - {e}")
#     log_data["epoch"] = epoch
#     wandb.log(log_data, step=log_data["epoch"])


# def image_log_handler(model, prepare_batch_fn, num_images=4):
#     def _handler(engine):
#         model.eval()
#         device = next(model.parameters()).device
#         with torch.no_grad():
#             all_images = []
#             all_gt_masks = []
#             all_pred_masks = []
#             for batch in engine.state.dataloader:
#                 images, targets = prepare_batch_fn(batch, device)
#                 if "mask" not in targets:
#                     continue
#                 if isinstance(model(images), (tuple, list)) and len(model(images)) > 1:
#                     _, seg_logits = model(images)
#                 else:
#                     seg_logits = model(images)
#                 pred_mask = (torch.sigmoid(seg_logits) > 0.5).float()
#                 gt_mask = targets["mask"]
#                 for i in range(len(images)):
#                     all_images.append(images[i].cpu())
#                     all_gt_masks.append(gt_mask[i].cpu())
#                     all_pred_masks.append(pred_mask[i].cpu())
#                     if len(all_images) >= num_images:
#                         break
#                 if len(all_images) >= num_images:
#                     break
#             if len(all_images) == 0:
#                 return
#             for i in range(min(len(all_images), num_images)):
#                 wandb.log({
#                     "image": wandb.Image(all_images[i], caption="GT vs Pred"),
#                     "gt_mask": wandb.Image(all_gt_masks[i]),
#                     "pred_mask": wandb.Image(all_pred_masks[i]),
#                 })
#     return _handler


# def manual_dice_handler(model, prepare_batch_fn):
#     logger = logging.getLogger(__name__)

#     def handler(engine):
#         val_loader = engine.state.dataloader
#         device = engine.state.device
#         model.eval()
#         dice_scores = []
#         with torch.no_grad():
#             for batch in val_loader:
#                 inputs, targets = prepare_batch_fn(batch, device=device, non_blocking=False)
#                 label = batch.get("label", {})
#                 if not isinstance(label, dict) or "mask" not in label:
#                     logger.warning("Skipping batch without 'mask' in nested label dict (classification-only batch).")
#                     continue
#                 outputs = model(inputs)
#                 _, seg_output = outputs
#                 pred_mask = (torch.sigmoid(seg_output) > 0.5).float()
#                 true_mask = targets["mask"]
#                 intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
#                 union = pred_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3))
#                 dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
#                 dice_scores.extend(dice.cpu().tolist())
#         if dice_scores:
#             avg_dice = sum(dice_scores) / len(dice_scores)
#             wandb.log({"val_dice_manual": avg_dice}, step=getattr(engine.state, "trainer_epoch", engine.state.epoch))
#         else:
#             logger.warning("No masks found for manual dice calculation.")
#     return handler


import logging
import torch
import wandb
from monai.handlers import EarlyStopHandler, StatsHandler, from_engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, DiskSaver

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


def attach_segmentation_metrics(evaluator, num_classes=2, output_transform=None, dice_name="val_dice", iou_name="val_iou"):
    from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
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
    prepare_batch=None,
    **handler_kwargs  # Everything from model_class.get_handler_kwargs()
):
    arch_name = str(config.get("architecture", "model")).lower().replace("/", "_").replace(" ", "_")
    checkpoint_prefix = f"{arch_name}"
    best_model_prefix = f"{arch_name}_best"

    # Segmentation metrics (if requested by MCP handler kwargs)
    if handler_kwargs.get("add_segmentation_metrics", False) and handler_kwargs.get("seg_output_transform", None) is not None:
        attach_segmentation_metrics(
            evaluator,
            num_classes=handler_kwargs.get("num_classes", 2),
            output_transform=handler_kwargs["seg_output_transform"],
            dice_name=handler_kwargs.get("dice_name", "val_dice"),
            iou_name=handler_kwargs.get("iou_name", "val_iou"),
        )

    # Training loss stats
    StatsHandler(
        tag_name="train",
        output_transform=from_engine(["loss"], first=True),
        iteration_log=False,
        epoch_log=True
    ).attach(trainer)

    # Validation metrics stats (always keys from MCP or config)
    val_metrics = handler_kwargs.get("val_metrics", ["val_acc", "val_auc", "val_loss"])  # fallback to common metrics
    StatsHandler(
        tag_name="val",
        output_transform=from_engine(val_metrics, first=False),
        iteration_log=False
    ).attach(evaluator)

    # # Validation after each epoch
    # # trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: evaluator.run())
    # def run_evaluator_with_epoch(engine):
    #     # Run evaluator and pass the trainer's epoch as an argument
    #     evaluator.state.trainer_epoch = engine.state.epoch  # Save for use in logging
    #     evaluator.run()
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator_with_epoch)

    # Manual dice/image handlers (if provided)
    if manual_dice_handler and prepare_batch is not None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, manual_dice_handler(model, prepare_batch))
    if image_log_handler and prepare_batch is not None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, image_log_handler(model, prepare_batch))
    if wandb_log_handler:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, wandb_log_handler)
        # evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: wandb_log_handler(engine, epoch=engine.state.epoch))

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


def wandb_log_handler(engine, epoch=None):
    """Log all available metrics from engine.state.metrics to wandb, handling types robustly."""
    print("wandb_log_handler metrics:", engine.state.metrics)
    # Try to get trainer's epoch if present, else fallback
    epoch = getattr(engine.state, "trainer_epoch", engine.state.epoch)
    log_data = {}
    for key, value in engine.state.metrics.items():
        print(f"[wandb_log_handler] key: {key}, value: {value}")
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

    log_data["epoch"] = epoch
    wandb.log(log_data, step=epoch)

    # log_data["epoch"] = epoch if epoch is not None else engine.state.epoch
    # wandb.log(log_data, step=log_data["epoch"])

    # log_data["epoch"] = engine.state.epoch
    # wandb.log(log_data, step=log_data["epoch"])

    # # log_data["epoch"] = engine.state.epoch
    # # wandb.log(log_data)

    # epoch = getattr(engine.state, "trainer_epoch", engine.state.epoch)
    # log_data["epoch"] = epoch
    # wandb.log(log_data, step=epoch)

    # log_data["epoch"] = epoch if epoch is not None else engine.state.epoch
    # wandb.log(log_data, step=log_data["epoch"])
    # log_data["epoch"] = epoch
    # wandb.log(log_data, step=log_data["epoch"])


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
                if "mask" not in targets:
                    continue
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
                return
            for i in range(min(len(all_images), num_images)):
                wandb.log({
                    "image": wandb.Image(all_images[i], caption="GT vs Pred"),
                    "gt_mask": wandb.Image(all_gt_masks[i]),
                    "pred_mask": wandb.Image(all_pred_masks[i]),
                })
    return _handler


def manual_dice_handler(model, prepare_batch_fn):
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
            wandb.log({"val_dice_manual": avg_dice}, step=engine.state.epoch)
        else:
            logger.warning("No masks found for manual dice calculation.")
    return handler
