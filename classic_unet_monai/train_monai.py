# train_monai.py
import random
import numpy as np
import wandb
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
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
    with torch.no_grad():
        try:
            val_batches = 0
            for batch in val_loader:
                val_batches += 1
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)

                # Collect for epoch-level metric computation
                all_outputs.append(outputs)
                all_masks.append(masks)

                # Manual Dice for each batch
                outputs_bin = (outputs > 0.5).float()
                intersection = (outputs_bin * masks).sum()
                dice_manual = 2. * intersection / (outputs_bin.sum() + masks.sum() + 1e-8)
                manual_dice_vals.append(dice_manual.item())

                if val_batches == 1:
                    print(f"[DEBUG] outputs shape: {outputs.shape}, masks shape: {masks.shape}")
                    print("outputs stats:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
                    print("masks stats:", masks.min().item(), masks.max().item(), masks.mean().item())
                    print("[DEBUG] Manual Dice (thresholded):", dice_manual.item())

            print(f"Epoch {epoch}: {val_batches} validation batches processed.")

            # Concatenate all for epoch-level DiceMetric
            all_outputs = torch.cat(all_outputs, dim=0)
            all_masks = torch.cat(all_masks, dim=0)

            # Squeeze channel dimension for DiceMetric if needed
            val_dice = dice_metric(all_outputs.squeeze(1), all_masks.squeeze(1)).item()
            manual_val_dice = sum(manual_dice_vals) / len(manual_dice_vals) if manual_dice_vals else 0.0

            print(f"Epoch {epoch}: Logging val_dice_coefficient = {val_dice}")
            print(f"Epoch {epoch}: Logging manual_val_dice_coefficient = {manual_val_dice}")

            wandb.log({
                "val_dice_coefficient": val_dice,
                "manual_val_dice_coefficient": manual_val_dice,
                "epoch": epoch
            })
        except Exception as e:
            print(f"Exception during validation at epoch {epoch}: {e}")
            import traceback; traceback.print_exc()
            wandb.log({"val_dice_exception": str(e), "epoch": epoch})

    # model.eval()
    # dice_metric.reset()
    # manual_dice_vals = []
    # with torch.no_grad():
    #     try:
    #         val_batches = 0
    #         for batch in val_loader:
    #             val_batches += 1
    #             images = batch["image"].to(device)
    #             masks = batch["mask"].to(device)
    #             outputs = model(images)
    #             outputs = torch.sigmoid(outputs)

    #             if val_batches == 1:
    #                 print(f"[DEBUG] outputs shape: {outputs.shape}, masks shape: {masks.shape}")
    #                 print("outputs stats:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
    #                 print("masks stats:", masks.min().item(), masks.max().item(), masks.mean().item())

    #             # Accumulate in DiceMetric
    #             dice_metric(outputs.squeeze(1), masks.squeeze(1))

    #             # Manual Dice for each batch
    #             outputs_bin = (outputs > 0.5).float()
    #             intersection = (outputs_bin * masks).sum()
    #             dice_manual = 2. * intersection / (outputs_bin.sum() + masks.sum() + 1e-8)
    #             manual_dice_vals.append(dice_manual.item())
    #             if val_batches == 1:
    #                 print("[DEBUG] Manual Dice (thresholded):", dice_manual.item())

    #         print(f"Epoch {epoch}: {val_batches} validation batches processed.")

    #         val_dice = dice_metric.aggregate().item()
    #         manual_val_dice = sum(manual_dice_vals) / len(manual_dice_vals) if manual_dice_vals else 0.0
    #         print(f"Epoch {epoch}: Logging val_dice_coefficient = {val_dice}")
    #         print(f"Epoch {epoch}: Logging manual_val_dice_coefficient = {manual_val_dice}")

    #         wandb.log({
    #             "val_dice_coefficient": val_dice,
    #             "manual_val_dice_coefficient": manual_val_dice,
    #             "epoch": epoch
    #         })
    #     except Exception as e:
    #         print(f"Exception during validation at epoch {epoch}: {e}")
    #         import traceback; traceback.print_exc()
    #         wandb.log({"val_dice_exception": str(e), "epoch": epoch})

# Save model with unique name, finish W&B
model_path = f"unet_monai_{wandb.run.id}.pth"
torch.save(model.state_dict(), model_path)
wandb.save(model_path)
wandb.finish()
