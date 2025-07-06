import sys
import monai
# print("PYTHON EXECUTABLE:", sys.executable)
# print("MONAI VERSION:", monai.__version__)
# print("MONAI UNet LOCATION:", monai.networks.nets.UNet)
# print("MONAI UNet DOC:", monai.networks.nets.UNet.__doc__)

import os
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

# ---- Compute learning rate for sweeps ----
if hasattr(config, "base_learning_rate") and hasattr(config, "lr_multiplier"):
    learning_rate = config.base_learning_rate * config.lr_multiplier
elif hasattr(config, "learning_rate"):
    learning_rate = config.learning_rate
else:
    learning_rate = 1e-4  # default fallback

# ---- Data preparation ----
train_loader, val_loader, test_loader = build_dataloaders(
    metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
    input_shape=tuple(config.input_shape[:2]),
    batch_size=config.batch_size,
    task="segmentation"  # or "multitask", "classification"
)

# ---- Model ----
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

# ---- Training Loop ----
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
    wandb.log({"train_loss": avg_loss, "epoch": epoch})

    # --- Validation ---
    model.eval()
    dice_vals = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            dice = dice_metric(outputs, masks)
            dice_vals.append(dice.mean().item())
    val_dice = sum(dice_vals) / len(dice_vals)
    wandb.log({"val_dice_coefficient": val_dice, "epoch": epoch})

# ---- Save model with unique name, finish W&B ----
model_path = f"unet_monai_{wandb.run.id}.pth"
torch.save(model.state_dict(), model_path)
wandb.save(model_path)
wandb.finish()
