import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import wandb
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
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
    num_class_labels=1,  # 1 for binary, 2 for two logits, adjust as needed
    features=(32, 64, 128, 256, 512)
).to(device)

# Losses
segmentation_loss_fn = DiceLoss(sigmoid=True)
classification_loss_fn = torch.nn.BCEWithLogitsLoss()
alpha, beta = 1.0, 1.0  # loss weights

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.l2_reg)
dice_metric = DiceMetric(include_background=True, reduction="mean")

# from sklearn.metrics import roc_auc_score, confusion_matrix

for epoch in range(config.epochs):
    # Training
    model.train()
    train_epoch_loss = 0
    train_class_preds = []
    train_class_targets = []
    train_dice_scores = []

    for batch in train_loader:
        images = batch["image"].to(device, dtype=torch.float)
        masks = batch["mask"].to(device, dtype=torch.float)
        labels = batch["label"].to(device, dtype=torch.float)

        optimizer.zero_grad()
        class_logits, seg_out = model(images)
        loss_seg = segmentation_loss_fn(seg_out, masks)
        # loss_class = classification_loss_fn(class_logits.squeeze(), labels)
        loss_class = classification_loss_fn(class_logits.view(-1), labels.view(-1))
        # loss_class = classification_loss_fn(class_logits.flatten(), labels.flatten())
        loss = alpha * loss_class + beta * loss_seg
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()

        # Store predictions for metrics
        with torch.no_grad():
            class_probs = torch.sigmoid(class_logits.squeeze()).cpu().numpy()
            preds = (class_probs > 0.5).astype(np.float32)
            # train_class_preds.extend(preds.tolist())
            preds = np.array(preds)
            train_class_preds.extend(preds.flatten().tolist())
            train_class_targets.extend(labels.cpu().numpy().tolist())
            # Segmentation
            dice_metric(seg_out, masks)
            # (do NOT append .item() inside the loop: DiceMetric accumulates batches)

    train_avg_loss = train_epoch_loss / len(train_loader)
    train_dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    train_acc = np.mean(np.array(train_class_preds) == np.array(train_class_targets))
    train_rocauc = roc_auc_score(train_class_targets, train_class_preds)
    tn, fp, fn, tp = confusion_matrix(train_class_targets, train_class_preds).ravel()
    train_sens = tp / (tp + fn + 1e-8)
    train_spec = tn / (tn + fp + 1e-8)

    # Validation
    model.eval()
    val_epoch_loss = 0
    val_class_preds = []
    val_class_targets = []
    val_dice_scores = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, dtype=torch.float)
            masks = batch["mask"].to(device, dtype=torch.float)
            labels = batch["label"].to(device, dtype=torch.float)

            class_logits, seg_out = model(images)
            loss_seg = segmentation_loss_fn(seg_out, masks)
            loss_class = classification_loss_fn(class_logits.squeeze(), labels)
            loss = alpha * loss_class + beta * loss_seg
            val_epoch_loss += loss.item()

            # Store predictions for metrics
            class_probs = torch.sigmoid(class_logits.squeeze()).cpu().numpy()
            preds = (class_probs > 0.5).astype(np.float32)
            val_class_preds.extend(preds.tolist())
            val_class_targets.extend(labels.cpu().numpy().tolist())
            # Segmentation
            dice_metric(seg_out, masks)

    val_avg_loss = val_epoch_loss / len(val_loader)
    val_dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    val_acc = np.mean(np.array(val_class_preds) == np.array(val_class_targets))
    val_rocauc = roc_auc_score(val_class_targets, val_class_preds)
    tn, fp, fn, tp = confusion_matrix(val_class_targets, val_class_preds).ravel()
    val_sens = tp / (tp + fn + 1e-8)
    val_spec = tn / (tn + fp + 1e-8)

    print(
        f"Epoch {epoch} | "
        f"train_loss: {train_avg_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_rocauc:.4f}, train_dice: {train_dice_score:.4f} | "
        f"val_loss: {val_avg_loss:.4f}, val_acc: {val_acc:.4f}, val_auc: {val_rocauc:.4f}, val_dice: {val_dice_score:.4f}"
    )

    wandb.log({
        "train_loss": train_avg_loss,
        "train_acc": train_acc,
        "train_auc": train_rocauc,
        "train_dice": train_dice_score,
        "train_sens": train_sens,
        "train_spec": train_spec,
        "val_loss": val_avg_loss,
        "val_acc": val_acc,
        "val_auc": val_rocauc,
        "val_dice": val_dice_score,
        "val_sens": val_sens,
        "val_spec": val_spec,
        "epoch": epoch,
    })


# Training Loop
# for epoch in range(config.epochs):
#     model.train()
#     epoch_loss = 0
#     class_correct = 0
#     class_total = 0
#     dice_scores = []

#     for batch in train_loader:
#         images = batch["image"].to(device, dtype=torch.float)
#         masks = batch["mask"].to(device, dtype=torch.float)
#         labels = batch["label"].to(device, dtype=torch.float)

#         optimizer.zero_grad()
#         class_logits, seg_out = model(images)
#         loss_seg = segmentation_loss_fn(seg_out, masks)
#         loss_class = classification_loss_fn(class_logits.squeeze(), labels)
#         loss = alpha * loss_class + beta * loss_seg
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#         # Metrics: classification
#         preds = (torch.sigmoid(class_logits.squeeze()) > 0.5).float()
#         class_correct += (preds == labels).sum().item()
#         class_total += labels.numel()
#         # Metrics: segmentation Dice
#         dice = dice_metric(seg_out, masks)
#         dice_scores.append(dice.item())

#     avg_loss = epoch_loss / len(train_loader)
#     avg_acc = class_correct / class_total if class_total else 0
#     avg_dice = np.mean(dice_scores)

#     print(f"Epoch {epoch} train_loss: {avg_loss:.4f}  acc: {avg_acc:.4f}  dice: {avg_dice:.4f}")
#     wandb.log({
#         "train_loss": avg_loss,
#         "train_acc": avg_acc,
#         "train_dice": avg_dice,
#         "epoch": epoch
#     })

#     # Optionally add validation here

print("Training completed.")
