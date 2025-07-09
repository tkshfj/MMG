# train_monai.py
import random
import numpy as np
import wandb
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from data_utils_monai import build_dataloaders

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

default_config = dict(
    batch_size=16,
    dropout=0.2,
    base_learning_rate=0.0002,
    lr_multiplier=1.0,
    l2_reg=1e-4,
    epochs=40,
    input_shape=[256, 256, 1],
    task="segmentation"
)
wandb.init(project="classic_unet_seg_monai", config=default_config)
config = wandb.config
batch_size = getattr(config, "batch_size", 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute learning rate for sweeps
if hasattr(config, "base_learning_rate") and hasattr(config, "lr_multiplier"):
    learning_rate = config.base_learning_rate * config.lr_multiplier
elif hasattr(config, "learning_rate"):
    learning_rate = config.learning_rate
else:
    learning_rate = 1e-4  # default fallback

# Data preparation
train_loader, val_loader, test_loader = build_dataloaders(
    metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
    input_shape=tuple(config.input_shape[:2]),
    batch_size=config.batch_size,
    task="segmentation"  # or "multitask", "classification"
)

# Model
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm='batch',
    dropout=config.dropout
).to(device)

loss_function = DiceLoss(sigmoid=True)
dice_metric = DiceMetric(include_background=True, reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} train_loss: {avg_loss}")
    wandb.log({"train_loss": avg_loss, "epoch": epoch})

    # Validation
    model.eval()
    all_outputs = []
    all_masks = []
    manual_dice_vals = []
    manual_iou_vals = []
    with torch.no_grad():
        try:
            val_batches = 0
            for batch in val_loader:
                val_batches += 1
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                all_outputs.append(outputs.cpu())
                all_masks.append(masks.cpu())

                # Manual Dice/IoU for each batch (thresholded)
                outputs_bin = (outputs > 0.5).float()
                intersection = (outputs_bin * masks).sum()
                dice_manual = 2. * intersection / (outputs_bin.sum() + masks.sum() + 1e-8)
                iou_manual = intersection / (((outputs_bin + masks) > 0).float().sum() + 1e-8)
                manual_dice_vals.append(dice_manual.item())
                manual_iou_vals.append(iou_manual.item())

                if val_batches == 1:
                    print(f"[DEBUG] outputs shape: {outputs.shape}, masks shape: {masks.shape}")
                    print("outputs stats:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
                    print("masks stats:", masks.min().item(), masks.max().item(), masks.mean().item())
                    print("[DEBUG] Manual Dice (thresholded):", dice_manual.item())
                    print("[DEBUG] Manual IoU (thresholded):", iou_manual.item())

            print(f"Epoch {epoch}: {val_batches} validation batches processed.")

            # Concatenate outputs and masks: [N, 1, H, W]
            all_outputs = torch.cat(all_outputs, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            outputs_sq = all_outputs.squeeze(1)
            masks_sq = all_masks.squeeze(1)
            outputs_bin = (outputs_sq > 0.5).float()

            # MONAI metrics: Dice and MeanIoU (per image, ignore nans)
            dice_metric = DiceMetric(include_background=True, reduction="none")
            iou_metric = MeanIoU(include_background=True, reduction="none")
            dice_scores = dice_metric(outputs_bin, masks_sq)
            iou_scores = iou_metric(outputs_bin, masks_sq)

            valid_dice = dice_scores[~torch.isnan(dice_scores)]
            valid_iou = iou_scores[~torch.isnan(iou_scores)]
            val_dice = valid_dice.mean().item() if valid_dice.numel() > 0 else float('nan')
            val_iou = valid_iou.mean().item() if valid_iou.numel() > 0 else float('nan')
            manual_val_dice = sum(manual_dice_vals) / len(manual_dice_vals) if manual_dice_vals else float('nan')
            manual_val_iou = sum(manual_iou_vals) / len(manual_iou_vals) if manual_iou_vals else float('nan')

            print(f"Epoch {epoch}: Logging val_dice_coefficient = {val_dice}")
            print(f"Epoch {epoch}: Logging manual_val_dice_coefficient = {manual_val_dice}")
            print(f"Epoch {epoch}: Logging val_iou_coefficient = {val_iou}")
            print(f"Epoch {epoch}: Logging manual_val_iou_coefficient = {manual_val_iou}")

            wandb.log({
                "val_dice_coefficient": val_dice,
                "manual_val_dice_coefficient": manual_val_dice,
                "val_iou_coefficient": val_iou,
                "manual_val_iou_coefficient": manual_val_iou,
                "epoch": epoch
            })
        except Exception as e:
            print(f"Exception during validation at epoch {epoch}: {e}")
            import traceback; traceback.print_exc()
            wandb.log({"val_dice_exception": str(e), "epoch": epoch})

# Save model with unique name, finish W&B
model_path = f"models/unet_monai_{wandb.run.id}.pth"
torch.save(model.state_dict(), model_path)
wandb.save(model_path)
wandb.finish()
