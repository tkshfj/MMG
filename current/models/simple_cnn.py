# simple_cnn.py
import logging
from typing import Any, List, Dict, Callable

import torch
import torch.nn as nn
from models.model_base import BaseModel
from metrics_utils import cls_output_transform

logger = logging.getLogger(__name__)


class SimpleCNNModel(BaseModel):
    """
    Baseline CNN classifier compatible with the refactored BaseModel API.
    - Stores config on instance for BaseModel helpers.
    - Uses BaseModel.get_loss_fn(task, cfg) (do not override).
    """

    # Registry / construction
    def build_model(self, config: Any) -> Any:
        self.config = config
        in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        self.num_classes: int = int(self._cfg_get(config, "num_classes", 2))
        # Allow either "dropout" or "dropout_rate" in config
        dropout: float = float(self._cfg_get(config, "dropout", self._cfg_get(config, "dropout_rate", 0.5)))

        counts = self._cfg_get(config, "class_counts", None)
        if counts and isinstance(counts, (list, tuple)) and len(counts) == self.num_classes:
            counts_t = torch.tensor(counts, dtype=torch.float32)
            weights = counts_t.sum() / (len(counts_t) * counts_t.clamp_min(1))
            weights = weights / weights.mean()
            # Store on cfg regardless of type
            if isinstance(config, dict):
                config["class_weights"] = weights.tolist()
            else:
                setattr(config, "class_weights", weights.tolist())
            logger.info("[SimpleCNN] class_counts=%s -> CE class_weights=%s", counts, [round(float(w), 4) for w in weights.tolist()])

        model = SimpleCNN(in_channels=in_channels, num_classes=self.num_classes, dropout=dropout)

        # Bias prior: p1 = positives / total, p0 = 1 - p1
        last = model.classifier[-1]
        if hasattr(last, "bias") and last.bias is not None:
            with torch.no_grad():
                if counts and len(counts) == self.num_classes and sum(counts) > 0:
                    import math
                    total = float(sum(counts))
                    # log priors; any constant shift is OK for softmax
                    priors = [max(c / total, 1e-6) for c in counts]
                    bias = torch.tensor([math.log(p) for p in priors], dtype=last.bias.dtype)
                    # optional: zero-mean the bias to avoid large constant offsets
                    bias = bias - bias.mean()
                    last.bias.copy_(bias)
                    logger.info("[SimpleCNN] init logits prior (classes=%d): %s", self.num_classes, [round(float(b), 4) for b in bias.tolist()])

        return model

    # Capabilities
    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    # Metrics / handlers wiring
    def get_output_transform(self) -> Callable:
        return cls_output_transform

    def get_cls_output_transform(self) -> Callable:
        return cls_output_transform

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Keep minimal and consistent with other wrappers
        return {
            "num_classes": self.get_num_classes(),
            "cls_output_transform": cls_output_transform,
            "seg_output_transform": None,  # uniform signature
        }

    # Optional convenience for external callers
    def extract_logits(self, y_pred: Any) -> "torch.Tensor":
        if isinstance(y_pred, dict):
            for k in ("class_logits", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        return y_pred  # tensor passthrough / fallback


class SimpleCNN(nn.Module):
    """
    Tiny CNN for binary/multi-class classification.
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
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.features(x)
        x = self.gap(x)
        logits = self.classifier(x)
        if logits.ndim != 2:
            raise RuntimeError(f"[SimpleCNN] Expected [B, num_classes], got {tuple(logits.shape)}")
        return logits
