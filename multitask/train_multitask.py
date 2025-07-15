import os                                 # For file and path handling
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
import torch                              # PyTorch core
import torch.nn as nn                     # Neural network components
from torch.optim import Adam              # Adam optimizer
# from monai.data import set_determinism    # Reproducibility
from monai.utils import set_determinism
from monai.metrics import DiceMetric, MeanIoU  # Dice metric for segmentation
import wandb                              # Weights & Biases for logging
from data_utils_monai import build_dataloaders  # Custom dataset utility
from multitask_unet import MultiTaskUNet  # Model
from sklearn.metrics import accuracy_score, roc_auc_score  # For classification metrics
import numpy as np  # For numerical operations


# def to_channel_first(t):
#     # Ensures (B, H, W) -> (B, 1, H, W)
#     if t.ndim == 3:
#         t = t.unsqueeze(1)
#     return t


def main():
    # =======================
    # Configuration
    # =======================
    default_config = dict(
        batch_size=16,
        dropout=0.2,
        base_learning_rate=0.0002,
        lr_multiplier=1.0,
        l2_reg=1e-4,
        epochs=40,
        input_shape=[256, 256, 1],
        task="multitask"
    )
    wandb.init(project="multitask_exp", config=default_config)
    config = wandb.config
    # learning_rate = config.base_learning_rate * config.lr_multiplier
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.base_learning_rate * config.lr_multiplier
    EPOCHS = config.epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================
    # Data Preparation
    # =======================
    # Set deterministic training for reproducibility
    set_determinism(seed=42)

    train_loader, val_loader, test_loader = build_dataloaders(
        metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
        input_shape=(256, 256),
        batch_size=BATCH_SIZE,
        task="multitask",  # or "segmentation"
        split=(0.7, 0.15, 0.15)
    )

    # =======================
    # Model, Optimizer, Loss
    # =======================
    model = MultiTaskUNet(                              # Instantiate model
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_class_labels=2,                             # E.g. binary classification
        features=(32, 64, 128, 256, 512)
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer

    # Losses
    criterion_seg = nn.BCEWithLogitsLoss()                   # For binary segmentation
    criterion_cls = nn.CrossEntropyLoss()                    # For classification

    # Initialize metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")

    # =======================
    # Training Loop
    # =======================
    for epoch in range(EPOCHS):
        model.train()                                   # Set model to train mode
        running_loss = 0.0

        for batch in train_loader:                      # Iterate over mini-batches
            images = batch['image'].to(DEVICE)
            masks = batch['mask'].to(DEVICE)
            labels = batch['label'].long().to(DEVICE)   # Ensure correct type for classification

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
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

        # =======================
        # Validation Loop
        # =======================
        model.eval()
        val_loss = 0.0
        manual_dices = []
        cls_preds, cls_probs, cls_targets = [], [], []
        dice_metric.reset()
        iou_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                masks = batch['mask'].to(DEVICE)
                labels = batch['label'].long().to(DEVICE)
                # Ensure shapes are (B, 1, H, W)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                class_logits, seg_out = model(images)
                pred_mask = (torch.sigmoid(seg_out) > 0.5).float()
                if pred_mask.ndim == 3:
                    pred_mask = pred_mask.unsqueeze(1)
                # Make sure types and devices match
                masks = masks.float().to(DEVICE)
                pred_mask = pred_mask.float().to(DEVICE)

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
                #     "pred_mask shape:", pred_mask.shape, "dtype:", pred_mask.dtype, "device:", pred_mask.device,
                #     "masks shape:", masks.shape, "dtype:", masks.dtype, "device:", masks.device
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

        # =======================
        # Logging
        # =======================
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_dice,
            "manual_dice": avg_manual_dice,
            "val_iou": avg_iou,
            "val_acc": val_acc,
            "val_auc": val_rocauc
        })

    # Save final model
    print("Training complete.")
    torch.save(model.state_dict(), "models/multitask_unet_final.pth")


if __name__ == "__main__":
    main()
