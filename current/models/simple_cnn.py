# simple_cnn.py
import logging
from typing import Any, List, Dict, Callable, Optional

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
        # expose to BaseModel helpers
        self.config = config
        self.class_counts: Optional[List[int]] = self._cfg_get(config, "class_counts", None)

        in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        self.num_classes: int = int(self._cfg_get(config, "num_classes", 2))
        dropout: float = float(self._cfg_get(config, "dropout", self._cfg_get(config, "dropout_rate", 0.5)))

        # unified binary policy
        pos_idx = int(self._cfg_get(config, "positive_index", 1))
        use_single = bool(self._cfg_get(config, "binary_single_logit",
                                        (self.num_classes == 2 and self._cfg_get(config, "use_single_logit", True))))
        out_dim = 1 if use_single else self.num_classes

        model = SimpleCNN(in_channels=in_channels, out_dim=out_dim, dropout=dropout)

        # initialize classifier bias from dataset priors (no-op if counts missing)
        last = model.classifier[-1]
        self.init_bias_from_priors(
            head=last,
            class_counts=self.class_counts,
            binary_single_logit=use_single,
            positive_index=pos_idx,
        )
        if self.class_counts:
            logger.info("[SimpleCNN] class_counts=%s -> init_bias applied (single_logit=%s, pos_idx=%d)",
                        self.class_counts, use_single, pos_idx)
        return model

    # Capabilities
    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    # Metrics / handlers wiring
    def get_output_transform(self) -> Callable:
        return cls_output_transform

    def get_cls_output_transform(self) -> Callable:
        return cls_output_transform

    # (Optional) keep if external code expects it; otherwise rely on BaseModel.get_metrics()
    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {"num_classes": self.get_num_classes(),
                "cls_output_transform": cls_output_transform,
                "seg_output_transform": None}


class SimpleCNN(nn.Module):
    """
    Tiny CNN for binary/multi-class classification.
    Feature stack -> AdaptiveAvgPool -> MLP. No assumption on HxW.
    """
    def __init__(self, in_channels: int = 1, out_dim: int = 2, dropout: float = 0.5):
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
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.gap(x)
        logits = self.classifier(x)
        if logits.ndim not in (1, 2):
            raise RuntimeError(f"[SimpleCNN] Expected [B] or [B, C], got {tuple(logits.shape)}")
        # Always return raw logits; BaseModel handles BCE/CE + thresholds
        return {"cls_logits": logits}
