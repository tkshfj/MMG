# simple_cnn.py
from typing import Any, List, Dict, Callable
import torch.nn as nn
from models.model_base import BaseModel
from metrics_utils import cls_output_transform


class SimpleCNNModel(BaseModel):
    """
    Registry wrapper for a lightweight CNN classifier.
    - Uses adaptive pooling so it doesn't assume 256x256 inputs.
    - Returns a weighted CrossEntropy loss if class weights are configured.
    - Supplies consistent classification handler kwargs and output transforms.
    """

    def build_model(self, config: Any) -> Any:
        self.config = config
        in_channels = int(self._cfg_get(config, "in_channels", 1))
        self.num_classes = int(self._cfg_get(config, "num_classes", 2))
        # Allow either "dropout" or "dropout_rate" in config
        dropout = float(self._cfg_get(config, "dropout", self._cfg_get(config, "dropout_rate", 0.5)))
        return SimpleCNN(in_channels=in_channels, num_classes=self.num_classes, dropout=dropout)

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    # Ensure all classification metrics see the same robust transform
    def get_output_transform(self) -> Callable:
        return cls_output_transform

    def get_cls_output_transform(self) -> Callable:
        return cls_output_transform

    def get_loss_fn(self) -> Callable:
        # Provided by BaseModel: builds CE with optional class weights from config
        return self._make_class_weighted_ce()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Names align with handler/metrics wiring
        return {
            "add_classification_metrics": True,
            "add_segmentation_metrics": False,
            "num_classes": self.get_num_classes(),
            "cls_output_transform": cls_output_transform,
            "acc_name": "val_acc",
            "auc_name": "val_auc",
            "confmat_name": "val_cls_confmat",
            # Present but unused when segmentation is off; keeps generic handlers happy
            "seg_output_transform": None,
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


class SimpleCNN(nn.Module):
    """
    A tiny CNN head for binary/multi-class classification.
    Feature stack -> AdaptiveAvgPool -> MLP. No assumption on HxW.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Collapses any spatial size to 1x1 -> 64 features total
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        logits = self.classifier(x)
        if logits.ndim != 2:
            raise RuntimeError(f"[SimpleCNN] Expected [B, num_classes], got {tuple(logits.shape)}")
        return logits
