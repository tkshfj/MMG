import torch
import os
import wandb
from monai.utils import set_determinism
from model_utils import build_model, get_optimizer
from data_utils import build_dataloaders
from eval_utils import (
    get_segmentation_metrics, get_classification_metrics, get_config,
    compute_segmentation_metrics, compute_classification_metrics, log_wandb,
    log_sample_images
)


def train(config=None):
    with wandb.init(config=config):
        config = get_config(wandb.config)
        set_determinism(seed=42)
        run_id = wandb.run.id
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, test_loader = build_dataloaders(
            metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
            input_shape=(256, 256),
            batch_size=config.batch_size,
            task="multitask",
            split=(0.7, 0.15, 0.15),
            num_workers=12
        )

        model = build_model(config).to(config.device)
        optimizer = get_optimizer(config.optimizer, model.parameters(), config.learning_rate, config.weight_decay)
        criterion_seg, criterion_cls = get_segmentation_metrics()["loss"], get_classification_metrics()["loss"]
        dice_metric, iou_metric = get_segmentation_metrics()["dice"], get_segmentation_metrics()["iou"]
        # MONAI/W&B handlers (prepared for integration or future Trainer/Evaluator)
        # Note: These handlers are designed for Ignite Engine, not direct calls
        # train_handlers = get_handlers("train")
        # val_handlers = get_handlers("val", model=model, save_dir="models/")

        # Training Loops
        for epoch in range(config.epochs):
            model.train()                                   # Set model to train mode
            running_loss = 0.0

            for batch in train_loader:                      # Iterate over mini-batches
                images = batch['image'].to(config.device)
                masks = batch['mask'].to(config.device)
                labels = batch['label'].long().to(config.device)   # Ensure correct type for classification

                optimizer.zero_grad()                       # Clear gradients
                class_logits, seg_out = model(images)       # Forward pass

                # Calculate losses
                loss_seg = criterion_seg(seg_out, masks)
                loss_cls = criterion_cls(class_logits, labels)
                loss = loss_seg + loss_cls                  # Simple sum, or use weights

                loss.backward()                             # Backpropagation
                optimizer.step()                            # Update weights

                running_loss += loss.item()

                # Handlers are designed for Ignite Engine, not direct calls
                # for h in train_handlers:
                #     h({"loss": loss.item()})

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {avg_train_loss:.4f}")

            # Validation Loop
            model.eval()
            val_loss = 0.0
            manual_dices = []
            cls_preds, cls_probs, cls_targets = [], [], []
            dice_metric.reset()
            iou_metric.reset()
            images_for_logging, masks_for_logging, preds_for_logging = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(config.device)
                    masks = batch['mask'].to(config.device)
                    labels = batch['label'].long().to(config.device)
                    # Ensure shapes are (B, 1, H, W)
                    if masks.ndim == 3:
                        masks = masks.unsqueeze(1)
                    class_logits, seg_out = model(images)
                    pred_mask = (torch.sigmoid(seg_out) > 0.5).float()
                    if pred_mask.ndim == 3:
                        pred_mask = pred_mask.unsqueeze(1)
                    # Make sure types and config.devices match
                    masks = masks.float().to(config.device)
                    pred_mask = pred_mask.float().to(config.device)

                    loss_seg = criterion_seg(seg_out, masks)
                    loss_cls = criterion_cls(class_logits, labels)
                    loss = loss_seg + loss_cls
                    val_loss += loss.item()

                    # Classification metrics
                    probs = torch.softmax(class_logits, dim=1)
                    pred_labels = torch.argmax(probs, dim=1)
                    cls_preds.extend(pred_labels.cpu().numpy())
                    cls_probs.extend(probs[:, 1].cpu().numpy())
                    cls_targets.extend(labels.cpu().numpy())

                    # Debugging
                    # print(
                    #     "pred_mask shape:", pred_mask.shape, "dtype:", pred_mask.dtype, "config.device:", pred_mask.config.device,
                    #     "masks shape:", masks.shape, "dtype:", masks.dtype, "config.device:", masks.config.device
                    # )
                    # print(
                    #     f"[DEBUG] mask unique: {torch.unique(masks)}, pred_mask unique: {torch.unique(pred_mask)}, "
                    #     f"mask mean: {masks.mean().item():.4f}, pred_mask mean: {pred_mask.mean().item():.4f}"
                    # )
                    # if torch.any(torch.isnan(pred_mask)) or torch.any(torch.isnan(masks)):
                    #     print("[DEBUG] Found nan in pred_mask or masks at validation step!")

                    # MONAI metrics
                    dice_metric(pred_mask, masks)
                    iou_metric(pred_mask, masks)

                    # Manual Dice (debug)
                    intersection = (pred_mask * masks).sum(dim=(1, 2, 3))
                    union = pred_mask.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                    manual_dice = (2. * intersection / (union + 1e-8)).mean().item()
                    manual_dices.append(manual_dice)
                    # print(f"[DEBUG] Manual Dice: {manual_dice:.4f}")

                    # For image logging, store first batch (or every N batches)
                    if len(images_for_logging) < 2:
                        images_for_logging.append(images.cpu())
                        masks_for_logging.append(masks.cpu())
                        preds_for_logging.append(pred_mask.cpu())

            avg_val_loss = val_loss / len(val_loader)
            avg_dice, avg_iou, avg_manual_dice = compute_segmentation_metrics(dice_metric, iou_metric, manual_dices, task=config.task)
            val_acc, val_rocauc = compute_classification_metrics(cls_targets, cls_preds, cls_probs)
            # val_f1 = f1_score(cls_targets, cls_preds, task=config.task)  # Commented out since not used
            # cm = confusion_matrix(cls_targets, cls_preds)  # Commented out since not used

            print(
                f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Dice: {avg_dice:.4f} | Manual Dice: {avg_manual_dice:.4f} | Val IoU: {avg_iou:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val AUC: {val_rocauc:.4f}"
            )

            # Custom epoch dictionary for handlers (commented out since handlers are disabled)
            # epoch_metrics = {
            #     "val_acc": val_acc,
            #     "val_auc": val_rocauc,
            #     "val_f1": val_f1,
            #     "val_cm_00": cm[0, 0] if cm.shape == (2, 2) else 0,
            #     "val_cm_01": cm[0, 1] if cm.shape == (2, 2) else 0,
            #     "val_cm_10": cm[1, 0] if cm.shape == (2, 2) else 0,
            #     "val_cm_11": cm[1, 1] if cm.shape == (2, 2) else 0,
            # }
            # for h in val_handlers:
            #     h(epoch_metrics)

            # Logging
            log_wandb(epoch, avg_train_loss, avg_val_loss, avg_dice, avg_manual_dice, avg_iou, val_acc, val_rocauc, task=config.task)

            # For segmentation-only
            # log_wandb(epoch, avg_train_loss, avg_val_loss, avg_dice, avg_manual_dice, avg_iou, task="segmentation")

            # For classification-only
            # log_wandb(epoch, avg_train_loss, avg_val_loss, val_acc=val_acc, val_rocauc=val_rocauc, task="classification")

            # Log sample images (first batch only)
            if images_for_logging:
                log_sample_images(images_for_logging[0], masks_for_logging[0], preds_for_logging[0], epoch, task=config.task)

        # Save final model
        print("Training complete.")
        os.makedirs("outputs/models", exist_ok=True)
        save_path = f"outputs/models/multitask_unet_{run_id}.pth"
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()
