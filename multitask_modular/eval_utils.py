import torch
import torch.nn as nn
from monai.metrics import DiceMetric, MeanIoU
from sklearn.metrics import accuracy_score, roc_auc_score
from types import SimpleNamespace

def get_segmentation_metrics():
    return {
        "loss": nn.BCEWithLogitsLoss(),
        "dice": DiceMetric(include_background=True, reduction="mean"),
        "iou": MeanIoU(include_background=True, reduction="mean"),
    }

def get_classification_metrics():
    return {
        "loss": nn.CrossEntropyLoss(),
        "accuracy": accuracy_score,
        "roc_auc": roc_auc_score,
    }

def get_handlers(tag):
    # Optionally add MONAI/W&B handlers
    return []

def get_config(cfg):
    defaults = {
        "batch_size": 16,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "epochs": 40,
        "optimizer": "Adam"
    }
    if cfg is not None:
        defaults.update(dict(cfg))
    return SimpleNamespace(**defaults)
