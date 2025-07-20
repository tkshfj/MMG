# multitask_modular/train.py
import os
import torch
import wandb
from ignite.handlers import ModelCheckpoint
from ignite.metrics import (
    Accuracy,
    ConfusionMatrix,
    DiceCoefficient,
    JaccardIndex,
    Loss,
    ROC_AUC,
)
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import CheckpointSaver, EarlyStopHandler, StatsHandler, from_engine
from monai.utils import set_determinism
from data_utils import build_dataloaders
from eval_utils import get_classification_metrics, get_config, get_segmentation_metrics  # log_sample_images
from model_utils import build_model, get_optimizer


# Multitask loss function (segmentation + classification)
def multitask_loss(y_pred, y_true):
    """Compute combined segmentation + classification loss."""
    class_logits, seg_out = y_pred            # model outputs
    labels = y_true['label']                  # ground truth class labels
    masks = y_true['mask']                    # ground truth segmentation masks
    # Individual losses
    loss_seg = get_segmentation_metrics()["loss"](seg_out, masks)
    loss_cls = get_classification_metrics()["loss"](class_logits, labels)
    return loss_seg + loss_cls


# Batch preparation function for engines
def prepare_batch(batch, device=None, non_blocking=False):
    """Move batch data to device and prepare inputs/targets for the model."""
    images = batch['image'].to(device, non_blocking=non_blocking)
    masks = batch['mask'].to(device, non_blocking=non_blocking)
    labels = batch['label'].long().to(device, non_blocking=non_blocking)
    return images, {'label': labels, 'mask': masks}


# Output transforms for multi-task per-sample list output
def seg_output_transform(output):
    # For segmentation (Dice/IoU/Jaccard): returns (pred_mask, true_mask), both batched
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        pred_masks, true_masks = [], []
        # Get device from first sample's seg_logits
        device = output[0]['pred'][1].device
        for sample in output:
            seg_logits = sample['pred'][1]
            true_mask = sample['label']['mask']
            # Ensure (1, H, W)
            pred_mask = (torch.sigmoid(seg_logits).unsqueeze(0) > 0.5).float().to(device)
            true_mask = true_mask.unsqueeze(0).float().to(device)
            pred_masks.append(pred_mask)
            true_masks.append(true_mask)
        pred_masks = torch.cat(pred_masks, dim=0)
        true_masks = torch.cat(true_masks, dim=0)
        return pred_masks, true_masks
    else:
        raise ValueError(f"seg_output_transform expected list of dicts, got: {output}")


def cls_output_transform(output):
    # For classification (Accuracy, AUROC): returns (pred_label, true_label), both batched
    # output is list of dicts per sample: [{'pred': (class_logits, seg_out), 'label': {...}}, ...]
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        # Should return [batch, num_classes], NOT [batch]
        pred_logits = torch.stack([s['pred'][0] for s in output])  # s['pred'][0] should be [num_classes]
        device = pred_logits.device
        true_labels = torch.tensor([s['label']['label'] for s in output], dtype=torch.long, device=device)
        assert pred_logits.shape[0] == true_labels.shape[0], "Batch size mismatch in classification output"
        print("DEBUG: pred_logits.shape:", pred_logits.shape)
        print("CLS METRIC TRANS:", pred_logits.shape, true_labels.shape)
        return pred_logits, true_labels
    else:
        raise ValueError("cls_output_transform expects a list of dicts")


def auc_output_transform(output):
    """
    For AUROC: returns (probs, true_label), batched.
    Handles both single-sample ([num_classes]) and batched ([batch, num_classes]) logits.
    """
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        probs_list, true_labels = [], []
        # Get device from first sample's logits
        device = torch.as_tensor(output[0]['pred'][0]).device
        for sample in output:
            class_logits = torch.as_tensor(sample['pred'][0]).to(device)
            label = sample['label']['label']
            # If class_logits is shape [num_classes] (standard case), softmax on dim=0
            if class_logits.ndim == 1:
                prob = torch.softmax(class_logits, dim=0)[1]
            # If shape is [1, num_classes], squeeze then softmax on dim=0
            elif class_logits.ndim == 2 and class_logits.shape[0] == 1:
                prob = torch.softmax(class_logits.squeeze(0), dim=0)[1]
            # If shape is [batch, num_classes], use dim=1, pick first sample
            elif class_logits.ndim == 2:
                prob = torch.softmax(class_logits, dim=1)[0, 1]
            else:
                raise ValueError(f"class_logits shape unexpected: {class_logits.shape}")
            probs_list.append(prob.unsqueeze(0))
            # true_labels.append(torch.tensor([label], dtype=torch.long))
            true_labels.append(torch.tensor([label], dtype=torch.long, device=device))
        return torch.cat(probs_list, dim=0), torch.cat(true_labels, dim=0)
    else:
        raise ValueError(f"auc_output_transform expected list of dicts, got: {output}")


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

        engine.state.metrics["manual_val_dice"] = mean_dice
        if wandb.run is not None:
            # Do **not** pass step=engine.state.epoch here, let wandb auto-increment step
            wandb.log({"manual_val_dice": mean_dice})

    return _handler


# Attach metrics to evaluator so StatsHandler and wandb can access them
def attach_metrics(evaluator):
    Accuracy(
        output_transform=lambda output: (
            torch.stack([s['pred'][0] for s in output]),
            torch.tensor([int(s['label']['label']) for s in output], dtype=torch.long, device=torch.stack([s['pred'][0] for s in output]).device)
        )
    ).attach(evaluator, "val_accuracy")
    # ROCAUCMetric(
    #     output_transform=lambda output: (
    #         torch.stack([s['pred'][0] for s in output]),
    #         torch.tensor([int(s['label']['label']) for s in output], dtype=torch.long)
    #     )
    # ).attach(evaluator, "val_auc")
    ROC_AUC(
        output_transform=auc_output_transform
    ).attach(evaluator, "val_auc")
    Loss(
        loss_fn=get_classification_metrics()["loss"],
        output_transform=lambda output: (
            torch.stack([s["pred"][0] for s in output]),
            torch.tensor([int(s["label"]["label"]) for s in output], dtype=torch.long, device=torch.stack([s['pred'][0] for s in output]).device)
        )
    ).attach(evaluator, "val_loss")


# Handlers for logging to Weights & Biases and image logging
def wandb_log_handler(engine):
    """Log metrics from engine.state.metrics to Weights & Biases."""
    log_data = {}
    # Convert metrics to plain float for logging
    for k, v in engine.state.metrics.items():
        try:
            # Handle torch tensors or numpy values
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach()
                if v.numel() == 1:
                    log_data[k] = v.item()
                else:
                    # Log as separate entries: val_dice_class0, val_dice_class1
                    for i, val in enumerate(v.flatten()):
                        log_data[f"{k}_c{i}"] = float(val)
            else:
                log_data[k] = float(v)
        except Exception as e:
            print(f"[wandb_log_handler] Warning: Could not log {k}: {v} - {e}")
    # Also log current epoch
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


# Main training function using MONAI engines
def train(config=None):
    # Initialize W&B run (config passed from sweep or CLI)
    with wandb.init(config=config):
        config = get_config(wandb.config)  # retrieve config object
        set_determinism(seed=42)           # for reproducibility
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.device = device  # store device in config
        run_id = wandb.run.id

        # Prepare data loaders
        train_loader, val_loader, test_loader = build_dataloaders(
            metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
            input_shape=(256, 256),
            batch_size=config.batch_size,
            task="multitask",
            split=(0.7, 0.15, 0.15),
            num_workers=32
        )

        # Build model and optimizer
        model = build_model(config).to(device)
        optimizer = get_optimizer(config.optimizer, model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        from ignite.engine import Events

        # MONAI Trainer Setup
        trainer = SupervisedTrainer(
            device=device,
            max_epochs=config.epochs,
            train_data_loader=train_loader,
            network=model,
            optimizer=optimizer,
            loss_function=multitask_loss,
            prepare_batch=prepare_batch
        )

        # Attach StatsHandler for training loss (per-iteration and per-epoch)
        StatsHandler(
            tag_name="train",
            output_transform=from_engine(["loss"], first=True)
        ).attach(trainer)

        # MONAI Evaluator Setup
        cm_metric = ConfusionMatrix(num_classes=2, output_transform=cls_output_transform)

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=val_loader,
            network=model,
            prepare_batch=prepare_batch,
            key_val_metric={
                "val_auc": ROC_AUC(output_transform=auc_output_transform)
            }
        )
        # Explicitly attach metrics to evaluator so StatsHandler and wandb can access them
        attach_metrics(evaluator)

        # Attach Dice and Jaccard using the cm object
        DiceCoefficient(cm=cm_metric).attach(evaluator, "val_mean_dice")
        JaccardIndex(cm=cm_metric).attach(evaluator, "val_iou")

        # Attach StatsHandler for validation metrics at epoch end
        StatsHandler(
            tag_name="val",
            output_transform=from_engine(["val_accuracy", "val_auc", "val_loss", "val_mean_dice", "val_iou"], first=False),
            iteration_log=False
        ).attach(evaluator)

        # Event Handlers
        # Run validation (evaluator) at the end of every training epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: evaluator.run())
        # Log manual Dice score at the end of each validation epoch
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, manual_dice_handler(model, prepare_batch))
        # Log sample images at end of each validation epoch (using evaluator)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, image_log_handler(model, prepare_batch))
        # Log metrics to Weights & Biases after each epoch (for both trainer and evaluator)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, wandb_log_handler)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, wandb_log_handler)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda e: print("Eval metrics at epoch end:", list(e.state.metrics.keys())))

        # Early Stopping & Checkpointing
        early_stopper = EarlyStopHandler(
            patience=10,
            min_delta=0.001,
            score_function=lambda engine: engine.state.metrics.get("val_auc", 0.0)
        )
        early_stopper.set_trainer(trainer)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

        # Checkpoint handler
        ckpt_saver = CheckpointSaver(
            save_dir="outputs/checkpoints",
            save_dict={"model": model},
            save_interval=1,
            n_saved=2,
        )
        ckpt_saver.attach(trainer)

        # Save the best model by AUC
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

        # Training loop
        trainer.run()

        # Save final model
        os.makedirs("outputs/models", exist_ok=True)
        final_path = f"outputs/models/multitask_unet_{run_id}.pth"
        torch.save(model.state_dict(), final_path)
        print("Training complete. Final model saved to:", final_path)


if __name__ == "__main__":
    train()
