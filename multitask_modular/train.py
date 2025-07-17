import torch
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score  # For classification metrics
import numpy as np                              # For numerical operations
from monai.utils import set_determinism
from multitask_model_utils import build_model, get_optimizer
from data_utils_monai import build_dataloaders
from multitask_eval_utils import (
    get_segmentation_metrics, get_classification_metrics,
    get_handlers, get_config
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
            num_workers=32
        )

        model = build_model(config).to(config.device)
        optimizer = get_optimizer(config.optimizer, model.parameters(), config.learning_rate, config.weight_decay)
        criterion_seg, criterion_cls = get_segmentation_metrics()["loss"], get_classification_metrics()["loss"]
        dice_metric, iou_metric = get_segmentation_metrics()["dice"], get_segmentation_metrics()["iou"]

        # for epoch in range(config.config.epochs):
        #     # --- Training loop ---
        #     # [move your training loop logic here, using model, optimizer, criterions]
        #     pass

        #     # --- Validation loop ---
        #     # [move your validation logic here, using metrics, handlers, wandb logging]
        #     pass

        #     # --- wandb logging, model saving, etc. ---

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

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {avg_train_loss:.4f}")

            # Validation Loop
            model.eval()
            val_loss = 0.0
            manual_dices = []
            cls_preds, cls_probs, cls_targets = [], [], []
            dice_metric.reset()
            iou_metric.reset()

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

            avg_val_loss = val_loss / len(val_loader)
            avg_dice = dice_metric.aggregate().item()
            avg_iou = iou_metric.aggregate().item()
            avg_manual_dice = np.mean(manual_dices) if manual_dices else float("nan")
            val_acc = accuracy_score(cls_targets, cls_preds)
            try:
                val_rocauc = roc_auc_score(cls_targets, cls_probs)
            except ValueError:
                val_rocauc = float("nan")

            print(
                f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Dice: {avg_dice:.4f} | Manual Dice: {avg_manual_dice:.4f} | Val IoU: {avg_iou:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val AUC: {val_rocauc:.4f}"
            )

            # Logging
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_dice": avg_dice,
                "val_dice_manual": avg_manual_dice,
                "val_iou": avg_iou,
                "val_acc": val_acc,
                "val_auc": val_rocauc
            })

        # Save final model
        print("Training complete.")
        save_path = f"models/multitask_unet_{run_id}.pth"
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()
