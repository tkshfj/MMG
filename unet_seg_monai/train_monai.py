# train_monai.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow info/warning logs

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import random
import wandb
from monai.utils import set_determinism         # MONAI utility for reproducibility
from monai.networks.nets import UNet           # Classic U-Net implementation
from monai.losses import DiceLoss              # Dice loss for segmentation
from monai.metrics import DiceMetric, MeanIoU  # Metrics for evaluating segmentation
from data_utils_monai import build_dataloaders # Custom dataloader builder


def main():
    # ======================= Configuration =======================
    default_config = dict(
        batch_size=16,              # How many samples per batch to load
        dropout=0.2,                # Dropout probability in U-Net
        base_learning_rate=0.0002,  # Base learning rate before applying multiplier
        lr_multiplier=1.0,          # Multiplier for learning rate (for sweeps)
        l2_reg=1e-4,                # L2 weight decay for optimizer regularization
        epochs=40,                  # Number of training epochs
        input_shape=[256, 256, 1],  # Shape of input images (H, W, C)
        task="segmentation"         # Task type (should remain "segmentation" here)
    )
    wandb.init(project="classic_unet_seg_monai", config=default_config)  # Initialize Weights & Biases
    config = wandb.config          # W&B config object (hyperparameters)
    run_id = wandb.run.id          # Unique ID for this W&B run

    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.base_learning_rate * config.lr_multiplier
    EPOCHS = config.epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # ======================= Reproducibility =======================
    seed = 42
    random.seed(seed)              # Python random seed
    np.random.seed(seed)           # Numpy random seed
    torch.manual_seed(seed)        # PyTorch random seed
    set_determinism(seed=seed)     # MONAI utility for full determinism

    # ======================= Data Loading =======================
    train_loader, val_loader, test_loader = build_dataloaders(
        metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",  # Path to CSV with image/mask metadata
        input_shape=tuple(config.input_shape[:2]),                     # Input image shape (H, W)
        batch_size=BATCH_SIZE,                                         # Batch size for DataLoader
        task="segmentation",                                           # Segmentation task mode
        split=(0.7, 0.15, 0.15),                                       # Split: 70% train, 15% val, 15% test
        num_workers=16                                                 # Multiprocessing workers for fast I/O
    )

    # ======================= Model, Optimizer, Loss =======================
    model = UNet(
        spatial_dims=2,                            # 2D segmentation
        in_channels=1,                             # Single channel (grayscale) input
        out_channels=1,                            # Single channel output mask
        channels=(32, 64, 128, 256, 512),          # U-Net encoder/decoder channels
        strides=(2, 2, 2, 2),                      # Downsampling strides
        num_res_units=2,                           # Residual units per stage
        norm='batch',                              # Batch normalization
        dropout=config.dropout                     # Dropout rate from config
    ).to(DEVICE)                                   # Move model to device (CPU/GPU)

    optimizer = Adam(model.parameters(),
                    lr=LEARNING_RATE,
                    weight_decay=config.l2_reg)    # Adam optimizer with weight decay

    criterion = DiceLoss(sigmoid=True)              # Dice loss with internal sigmoid for logits
    dice_metric = DiceMetric(include_background=True, reduction="mean")  # MONAI Dice metric
    iou_metric = MeanIoU(include_background=True, reduction="mean")      # MONAI IoU metric

    # ======================= Training Loop =======================
    for epoch in range(EPOCHS):
        model.train()                               # Set model to training mode
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)      # Input images, shape: (B, 1, H, W)
            masks = batch["mask"].to(DEVICE)        # Ground truth masks, shape: (B, 1, H, W)

            optimizer.zero_grad()                   # Zero gradients before backward pass
            outputs = model(images)                 # Forward pass through U-Net
            loss = criterion(outputs, masks)        # Compute Dice loss
            loss.backward()                         # Backpropagate
            optimizer.step()                        # Update parameters

            running_loss += loss.item()             # Accumulate batch loss

        avg_train_loss = running_loss / len(train_loader)    # Average train loss for epoch
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

        # ======================= Validation =======================
        model.eval()                                # Set model to evaluation mode
        val_loss = 0.0
        dice_metric.reset()                         # Reset MONAI Dice metric state
        iou_metric.reset()                          # Reset MONAI IoU metric state
        manual_dices = []                           # For storing manual Dice scores
        manual_ious = []                            # For storing manual IoU scores

        with torch.no_grad():                       # No gradients needed during validation
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)    # Validation Dice loss
                val_loss += loss.item()

                outputs_sigmoid = torch.sigmoid(outputs)         # Convert logits to [0,1]
                pred_mask = (outputs_sigmoid > 0.5).float()      # Binarize predictions

                # Update MONAI metrics (aggregate over the whole epoch)
                dice_metric(pred_mask, masks)
                iou_metric(pred_mask, masks)

                # Manual Dice & IoU (calculated per-batch for reference/debug)
                intersection = (pred_mask * masks).sum(dim=(1, 2, 3))
                union = pred_mask.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                manual_dice = (2. * intersection / (union + 1e-8)).mean().item()
                manual_dices.append(manual_dice)

                iou_union = ((pred_mask + masks) > 0).float().sum(dim=(1, 2, 3))
                manual_iou = (intersection / (iou_union + 1e-8)).mean().item()
                manual_ious.append(manual_iou)

        avg_val_loss = val_loss / len(val_loader)                       # Average val loss
        avg_dice = dice_metric.aggregate().item()                       # Average MONAI Dice
        avg_iou = iou_metric.aggregate().item()                         # Average MONAI IoU
        avg_manual_dice = np.mean(manual_dices) if manual_dices else float("nan")  # Manual Dice
        avg_manual_iou = np.mean(manual_ious) if manual_ious else float("nan")     # Manual IoU

        print(
            f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_dice:.4f} | Manual Dice: {avg_manual_dice:.4f} | "
            f"Val IoU: {avg_iou:.4f} | Manual IoU: {avg_manual_iou:.4f}"
        )

        # ======================= Logging to Weights & Biases =======================
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_dice,
            "val_dice_manual": avg_manual_dice,
            "val_iou": avg_iou,
            "val_iou_manual": avg_manual_iou
        })

    # ======================= Model Saving =======================
    os.makedirs("models", exist_ok=True)                # Ensure model output directory exists
    save_path = f"models/unet_monai_{run_id}.pth"       # File path with W&B run ID
    torch.save(model.state_dict(), save_path)           # Save trained model weights
    wandb.save(save_path)                               # Save model as W&B artifact
    wandb.finish()                                      # Mark run as finished in W&B

    print("Training complete. Model saved to:", save_path)


if __name__ == "__main__":
    main()

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import random
# import numpy as np
# import wandb
# import torch
# from monai.networks.nets import UNet
# from monai.losses import DiceLoss
# from monai.metrics import DiceMetric, MeanIoU
# from data_utils_monai import build_dataloaders

# # Set seeds for reproducibility
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# default_config = dict(
#     batch_size=16,
#     dropout=0.2,
#     base_learning_rate=0.0002,
#     lr_multiplier=1.0,
#     l2_reg=1e-4,
#     epochs=40,
#     input_shape=[256, 256, 1],
#     task="segmentation"
# )
# wandb.init(project="classic_unet_seg_monai", config=default_config)
# config = wandb.config
# batch_size = getattr(config, "batch_size", 16)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Compute learning rate for sweeps
# if hasattr(config, "base_learning_rate") and hasattr(config, "lr_multiplier"):
#     learning_rate = config.base_learning_rate * config.lr_multiplier
# elif hasattr(config, "learning_rate"):
#     learning_rate = config.learning_rate
# else:
#     learning_rate = 1e-4  # default fallback

# # Data preparation
# train_loader, val_loader, test_loader = build_dataloaders(
#     metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
#     input_shape=tuple(config.input_shape[:2]),
#     batch_size=config.batch_size,
#     task="segmentation"  # or "multitask", "classification"
# )

# # Model
# model = UNet(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=1,
#     channels=(32, 64, 128, 256, 512),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm='batch',
#     dropout=config.dropout
# ).to(device)

# loss_function = DiceLoss(sigmoid=True)
# dice_metric = DiceMetric(include_background=True, reduction="mean")
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Training Loop
# for epoch in range(config.epochs):
#     model.train()
#     epoch_loss = 0
#     for batch in train_loader:
#         images = batch["image"].to(device)
#         masks = batch["mask"].to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = loss_function(outputs, masks)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     avg_loss = epoch_loss / len(train_loader)
#     print(f"Epoch {epoch} train_loss: {avg_loss}")
#     wandb.log({"train_loss": avg_loss, "epoch": epoch})

#     # Validation
#     model.eval()
#     all_outputs = []
#     all_masks = []
#     manual_dice_vals = []
#     manual_iou_vals = []
#     with torch.no_grad():
#         try:
#             val_batches = 0
#             for batch in val_loader:
#                 val_batches += 1
#                 images = batch["image"].to(device)
#                 masks = batch["mask"].to(device)
#                 outputs = model(images)
#                 outputs = torch.sigmoid(outputs)
#                 all_outputs.append(outputs.cpu())
#                 all_masks.append(masks.cpu())

#                 # Manual Dice/IoU for each batch (thresholded)
#                 outputs_bin = (outputs > 0.5).float()
#                 intersection = (outputs_bin * masks).sum()
#                 dice_manual = 2. * intersection / (outputs_bin.sum() + masks.sum() + 1e-8)
#                 iou_manual = intersection / (((outputs_bin + masks) > 0).float().sum() + 1e-8)
#                 manual_dice_vals.append(dice_manual.item())
#                 manual_iou_vals.append(iou_manual.item())

#                 if val_batches == 1:
#                     print(f"[DEBUG] outputs shape: {outputs.shape}, masks shape: {masks.shape}")
#                     print("outputs stats:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
#                     print("masks stats:", masks.min().item(), masks.max().item(), masks.mean().item())
#                     print("[DEBUG] Manual Dice (thresholded):", dice_manual.item())
#                     print("[DEBUG] Manual IoU (thresholded):", iou_manual.item())

#             print(f"Epoch {epoch}: {val_batches} validation batches processed.")

#             # Concatenate outputs and masks: [N, 1, H, W]
#             all_outputs = torch.cat(all_outputs, dim=0)
#             all_masks = torch.cat(all_masks, dim=0)
#             outputs_sq = all_outputs.squeeze(1)
#             masks_sq = all_masks.squeeze(1)
#             outputs_bin = (outputs_sq > 0.5).float()

#             # MONAI metrics: Dice and MeanIoU (per image, ignore nans)
#             dice_metric = DiceMetric(include_background=True, reduction="none")
#             iou_metric = MeanIoU(include_background=True, reduction="none")
#             dice_scores = dice_metric(outputs_bin, masks_sq)
#             iou_scores = iou_metric(outputs_bin, masks_sq)

#             valid_dice = dice_scores[~torch.isnan(dice_scores)]
#             valid_iou = iou_scores[~torch.isnan(iou_scores)]
#             val_dice = valid_dice.mean().item() if valid_dice.numel() > 0 else float('nan')
#             val_iou = valid_iou.mean().item() if valid_iou.numel() > 0 else float('nan')
#             manual_val_dice = sum(manual_dice_vals) / len(manual_dice_vals) if manual_dice_vals else float('nan')
#             manual_val_iou = sum(manual_iou_vals) / len(manual_iou_vals) if manual_iou_vals else float('nan')

#             print(f"Epoch {epoch}: Logging val_dice_coefficient = {val_dice}")
#             print(f"Epoch {epoch}: Logging manual_val_dice_coefficient = {manual_val_dice}")
#             print(f"Epoch {epoch}: Logging val_iou_coefficient = {val_iou}")
#             print(f"Epoch {epoch}: Logging manual_val_iou_coefficient = {manual_val_iou}")

#             wandb.log({
#                 "val_dice_coefficient": val_dice,
#                 "manual_val_dice_coefficient": manual_val_dice,
#                 "val_iou_coefficient": val_iou,
#                 "manual_val_iou_coefficient": manual_val_iou,
#                 "epoch": epoch
#             })
#         except Exception as e:
#             print(f"Exception during validation at epoch {epoch}: {e}")
#             import traceback
#             traceback.print_exc()
#             wandb.log({"val_dice_exception": str(e), "epoch": epoch})

# # Save model with unique name, finish W&B
# model_path = f"models/unet_monai_{wandb.run.id}.pth"
# torch.save(model.state_dict(), model_path)
# wandb.save(model_path)
# wandb.finish()
