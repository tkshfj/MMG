import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import wandb
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from data_utils_monai import build_dataloaders
from multitask_unet import MultiTaskUNet
from sklearn.metrics import roc_auc_score, confusion_matrix

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Config
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
wandb.init(project="multitask_unet_monai", config=default_config)
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = config.base_learning_rate * config.lr_multiplier

# Data preparation
train_loader, val_loader, test_loader = build_dataloaders(
    metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
    input_shape=tuple(config.input_shape[:2]),
    batch_size=config.batch_size,
    task="multitask"
)

# Model
model = MultiTaskUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_class_labels=1,
    features=(32, 64, 128, 256, 512)
).to(device)

# Losses
segmentation_loss_fn = DiceLoss(sigmoid=True)
classification_loss_fn = torch.nn.BCEWithLogitsLoss()
alpha, beta = 1.0, 1.0

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.l2_reg)
dice_metric = DiceMetric(include_background=False, reduction="mean")
iou_metric = MeanIoU(include_background=False, reduction="none")

for epoch in range(config.epochs):
    # Training
    model.train()
    train_epoch_loss = 0
    train_class_preds, train_class_targets = [], []
    train_dice_scores_manual = []

    for batch in train_loader:
        images = batch["image"].to(device, dtype=torch.float)
        masks = batch["mask"].to(device, dtype=torch.float)
        labels = batch["label"].to(device, dtype=torch.float)
        # Sanity check mask/seg shape
        assert masks.shape == (images.shape[0], 1, images.shape[2], images.shape[3]), f"Mask shape {masks.shape}"
        optimizer.zero_grad()
        class_logits, seg_out = model(images)

        # Debug: print stats
        if epoch == 0:  # Only print for first epoch to reduce log spam
            print("Seg out stats:", seg_out.min().item(), seg_out.max().item(), seg_out.mean().item())
            print("Mask stats:", masks.min().item(), masks.max().item(), masks.mean().item())
            pred_probs = torch.sigmoid(seg_out)
            print("Pred mask mean:", pred_probs.mean().item(), 
                  "min:", pred_probs.min().item(), 
                  "max:", pred_probs.max().item())
            pred_bin = (pred_probs > 0.5).float()
            print("Pred binary unique:", torch.unique(pred_bin))

        loss_seg = segmentation_loss_fn(seg_out, masks)
        loss_class = classification_loss_fn(class_logits.view(-1), labels.view(-1))
        loss = alpha * loss_class + beta * loss_seg
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()

        with torch.no_grad():
            # Classification metrics
            class_probs = torch.sigmoid(class_logits.view(-1)).cpu().numpy()
            preds = (class_probs > 0.5).astype(np.float32)
            train_class_preds.extend(preds.tolist())
            train_class_targets.extend(labels.view(-1).cpu().numpy().tolist())

            # Segmentation metrics: MONAI and manual
            pred_probs = torch.sigmoid(seg_out)

            # Ensure shape, type, device match
            if pred_probs.shape != masks.shape:
                print(f"[WARNING] Shape mismatch: pred_probs {pred_probs.shape}, masks {masks.shape}")
                pred_probs = pred_probs.view_as(masks)
            pred_probs = pred_probs.float()
            masks = masks.float()
            if pred_probs.device != masks.device:
                pred_probs = pred_probs.to(masks.device)
            # Optional: Convert to vanilla torch.Tensor (not MetaTensor) to avoid MONAI issues
            pred_probs = pred_probs.clone().detach()
            masks = masks.clone().detach()
            # print(f"Calling dice_metric: pred_probs {pred_probs.shape} {pred_probs.dtype}, masks {masks.shape} {masks.dtype}")
            # print(f"Pred min/max: {pred_probs.min().item():.4f}, {pred_probs.max().item():.4f}")
            # print(f"Mask min/max: {masks.min().item():.4f}, {masks.max().item():.4f}")
            dice_metric(pred_probs, masks)
            pred_bin = (pred_probs > 0.5).float()
            intersection = (pred_bin * masks).sum().item()
            union = pred_bin.sum().item() + masks.sum().item()
            dice_manual = (2. * intersection) / (union + 1e-8)
            train_dice_scores_manual.append(dice_manual)

    train_avg_loss = train_epoch_loss / len(train_loader)
    train_dice_score = dice_metric.aggregate().item()
    train_dice_score_manual = np.mean(train_dice_scores_manual)
    dice_metric.reset()
    train_acc = np.mean(np.array(train_class_preds) == np.array(train_class_targets))
    try:
        train_rocauc = roc_auc_score(train_class_targets, train_class_preds)
    except Exception:
        train_rocauc = 0
    try:
        tn, fp, fn, tp = confusion_matrix(train_class_targets, train_class_preds).ravel()
        train_sens = tp / (tp + fn + 1e-8)
        train_spec = tn / (tn + fp + 1e-8)
    except Exception:
        train_sens = train_spec = 0

    # Validation
    model.eval()
    val_epoch_loss = 0
    val_class_preds, val_class_targets = [], []
    manual_dice_vals = []
    manual_iou_vals = []
    all_outputs = []
    all_masks = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, dtype=torch.float)
            masks = batch["mask"].to(device, dtype=torch.float)
            labels = batch["label"].to(device, dtype=torch.float)
            class_logits, seg_out = model(images)

            # Loss
            loss_seg = segmentation_loss_fn(seg_out, masks)
            loss_class = classification_loss_fn(class_logits.view(-1), labels.view(-1))
            loss = alpha * loss_class + beta * loss_seg
            val_epoch_loss += loss.item()

            # Classification Metrics
            class_probs = torch.sigmoid(class_logits.view(-1)).cpu().numpy()
            preds = (class_probs > 0.5).astype(np.float32)
            val_class_preds.extend(preds.tolist())
            val_class_targets.extend(labels.view(-1).cpu().numpy().tolist())

            # Segmentation Metrics
            outputs = torch.sigmoid(seg_out)
            all_outputs.append(outputs.cpu())
            all_masks.append(masks.cpu())

            # Manual Dice & IoU (batch)
            outputs_bin = (outputs > 0.5).float()
            intersection = (outputs_bin * masks).sum()
            dice_manual = 2. * intersection / (outputs_bin.sum() + masks.sum() + 1e-8)
            iou_manual = intersection / (((outputs_bin + masks) > 0).float().sum() + 1e-8)
            manual_dice_vals.append(dice_manual.item())
            manual_iou_vals.append(iou_manual.item())

    # Aggregate all outputs for MONAI metric
    all_outputs = torch.cat(all_outputs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    outputs_sq = all_outputs.squeeze(1)
    masks_sq = all_masks.squeeze(1)
    outputs_bin = (outputs_sq > 0.5).float()

    # MONAI metrics
    dice_scores = dice_metric(outputs_bin, masks_sq)
    iou_scores = iou_metric(outputs_bin, masks_sq)

    valid_dice = dice_scores[~torch.isnan(dice_scores)]
    valid_iou = iou_scores[~torch.isnan(iou_scores)]
    val_dice = valid_dice.mean().item() if valid_dice.numel() > 0 else float('nan')
    val_iou = valid_iou.mean().item() if valid_iou.numel() > 0 else float('nan')
    manual_val_dice = np.mean(manual_dice_vals) if manual_dice_vals else float('nan')
    manual_val_iou = np.mean(manual_iou_vals) if manual_iou_vals else float('nan')

    val_avg_loss = val_epoch_loss / len(val_loader)
    val_acc = np.mean(np.array(val_class_preds) == np.array(val_class_targets))
    try:
        val_rocauc = roc_auc_score(val_class_targets, val_class_preds)
    except Exception:
        val_rocauc = 0
    try:
        tn, fp, fn, tp = confusion_matrix(val_class_targets, val_class_preds).ravel()
        val_sens = tp / (tp + fn + 1e-8)
        val_spec = tn / (tn + fp + 1e-8)
    except Exception:
        val_sens = val_spec = 0

    print(
        f"[Epoch {epoch}] Validation Dice (MONAI): {val_dice:.4f}, Manual Dice: {manual_val_dice:.4f}, "
        f"MONAI IoU: {val_iou:.4f}, Manual IoU: {manual_val_iou:.4f}"
    )
    print(
        f"Epoch {epoch} | "
        f"val_loss: {val_avg_loss:.4f}, val_acc: {val_acc:.4f}, val_auc: {val_rocauc:.4f}, "
        f"val_dice: {val_dice:.4f}, val_iou: {val_iou:.4f}"
    )

    wandb.log({
        "val_loss": val_avg_loss,
        "val_acc": val_acc,
        "val_auc": val_rocauc,
        "val_dice": val_dice,
        "val_dice_manual": manual_val_dice,
        "val_iou": val_iou,
        "val_iou_manual": manual_val_iou,
        "val_sens": val_sens,
        "val_spec": val_spec,
        "epoch": epoch,
    })

    dice_metric.reset()
    iou_metric.reset()

print("Training completed.")
wandb.finish()
