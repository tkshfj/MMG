# eval_utils.py
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

import wandb


# Metric/Handler/Config Factories
def get_segmentation_metrics():
    return {"loss": nn.BCEWithLogitsLoss()}


def get_classification_metrics():
    return {"loss": nn.CrossEntropyLoss()}


# Batch preparation function for engines
def prepare_batch(batch, device=None, non_blocking=False):
    """
    Prepare inputs and nested labels for multitask learning with MONAI/Ignite.
    Supports nested label dictionaries: batch["label"]["mask"], batch["label"]["label"]
    """
    images = batch["image"].to(device, non_blocking=non_blocking)
    label_dict = batch.get("label", {})
    if not isinstance(label_dict, dict):
        raise ValueError(f"Expected 'label' to be a dict, got {type(label_dict)} (batch keys: {list(batch.keys())})")
    # Prepare target dictionary
    target = {}
    mask = label_dict.get("mask")
    class_label = label_dict.get("label")
    # Ensure mask and class_label are moved to the correct device
    if mask is not None:
        target["mask"] = mask.to(device, non_blocking=non_blocking)
    if class_label is not None:
        target["label"] = class_label.to(device, non_blocking=non_blocking).long()  # force integer class labels
    # Ensure at least one target is present
    if not target:
        raise ValueError(
            f"Batch label dictionary contains neither 'mask' nor 'label': keys={list(label_dict.keys())}"
        )
    return images, target


# Metric Computation Functions
def compute_segmentation_metrics(dice_metric=None, iou_metric=None, manual_dices=None, task="multitask"):
    if task not in ("multitask", "segmentation"):
        return None, None, None
    avg_dice = dice_metric.aggregate().item() if dice_metric is not None else None
    avg_iou = iou_metric.aggregate().item() if iou_metric is not None else None
    avg_manual_dice = np.mean(manual_dices) if manual_dices else None
    return avg_dice, avg_iou, avg_manual_dice


def compute_classification_metrics(cls_targets=None, cls_preds=None, cls_probs=None, task="multitask"):
    if task not in ("multitask", "classification"):
        return None, None
    val_acc = None
    val_rocauc = None
    if cls_targets is not None and cls_preds is not None:
        try:
            val_acc = accuracy_score(cls_targets, cls_preds)
        except Exception:
            val_acc = None
    if cls_targets is not None and cls_probs is not None:
        try:
            val_rocauc = roc_auc_score(cls_targets, cls_probs)
        except Exception:
            val_rocauc = None
    return val_acc, val_rocauc


# WandB Logging Utility
def log_wandb(
    epoch,
    avg_train_loss,
    avg_val_loss,
    avg_dice=None,
    avg_manual_dice=None,
    avg_iou=None,
    val_acc=None,
    val_rocauc=None,
    task="multitask"
):
    log_dict = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
    }

    if task in ["multitask", "segmentation"]:
        if avg_dice is not None:
            log_dict["val_dice"] = avg_dice
        if avg_manual_dice is not None:
            log_dict["val_dice_manual"] = avg_manual_dice
        if avg_iou is not None:
            log_dict["val_iou"] = avg_iou

    if task in ["multitask", "classification"]:
        if val_acc is not None:
            log_dict["val_acc"] = val_acc
        if val_rocauc is not None:
            log_dict["val_auc"] = val_rocauc

    # Log only keys with non-None values
    # wandb.log(log_dict)
    wandb.log({k: v for k, v in log_dict.items() if v is not None})


def log_sample_images(images, masks=None, preds=None, epoch=0, n=2, task="multitask"):
    """
    Log n sample images/masks/preds to W&B, robust to tensor shapes and task modes:
    - images: [B, 1, H, W], [B, H, W], or [B, 3, H, W]
    - masks, preds: [B, 1, H, W], [B, H, W], [B, C, H, W], or None
    - task: "multitask", "segmentation", "classification"
    """
    if images is None:
        print("log_sample_images: Images input is None, skipping logging.")
        return

    # Convert to numpy
    images = images.detach().cpu().numpy()
    batch_size = images.shape[0]
    n = min(n, batch_size)

    # Determine number of columns based on task
    if task in ("multitask", "segmentation"):
        n_cols = 3
    else:
        n_cols = 1  # For classification-only, just input images

    for i in range(n):
        fig, axs = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))

        # Handle both single and multiple axes
        if n_cols == 1:
            axs = [axs]

        # --- Input Image ---
        img = images[i]
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0]
                axs[0].imshow(img, cmap="gray")
            elif img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
                axs[0].imshow(img)
            else:
                axs[0].imshow(img[0], cmap="gray")
        elif img.ndim == 2:
            axs[0].imshow(img, cmap="gray")
        else:
            axs[0].text(0.5, 0.5, "Invalid img", ha="center")
        axs[0].set_title("Input")
        axs[0].axis("off")

        # --- Mask and Prediction (if available and relevant for task) ---
        if n_cols > 1:
            # Mask
            msk = masks[i] if (masks is not None) else None
            if msk is not None:
                if msk.ndim == 3:
                    if msk.shape[0] == 1:
                        msk = msk[0]
                        axs[1].imshow(msk, cmap="gray")
                    elif msk.shape[0] == 3:
                        msk = np.transpose(msk, (1, 2, 0))
                        axs[1].imshow(msk)
                    else:
                        axs[1].imshow(msk[0], cmap="gray")
                elif msk.ndim == 2:
                    axs[1].imshow(msk, cmap="gray")
                else:
                    axs[1].text(0.5, 0.5, "Invalid mask", ha="center")
                axs[1].set_title("Mask")
            else:
                axs[1].text(0.5, 0.5, "No mask", ha="center")
                axs[1].set_title("Mask")
            axs[1].axis("off")

            # Prediction
            prd = preds[i] if (preds is not None) else None
            if prd is not None:
                if prd.ndim == 3:
                    if prd.shape[0] == 1:
                        prd = prd[0]
                        axs[2].imshow(prd, cmap="gray")
                    elif prd.shape[0] == 3:
                        prd = np.transpose(prd, (1, 2, 0))
                        axs[2].imshow(prd)
                    else:
                        axs[2].imshow(prd[0], cmap="gray")
                elif prd.ndim == 2:
                    axs[2].imshow(prd, cmap="gray")
                else:
                    axs[2].text(0.5, 0.5, "Invalid pred", ha="center")
                axs[2].set_title("Pred")
            else:
                axs[2].text(0.5, 0.5, "No pred", ha="center")
                axs[2].set_title("Pred")
            axs[2].axis("off")

        fig.suptitle(f"Epoch {epoch + 1} | Sample {i}")
        try:
            wandb.log({f"Sample_{i}_epoch_{epoch + 1}": wandb.Image(fig)})
        except Exception as e:
            print(f"W&B image log failed: {e}")
        plt.close(fig)
