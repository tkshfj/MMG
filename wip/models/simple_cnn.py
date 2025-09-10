# simple_cnn.py
import logging
from typing import Any, Dict, Mapping, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimpleCNNModel(nn.Module):
    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        """
        Accepts either a config dict via `cfg` or plain kwargs.
        _build_with_class(...) may call us as cls(**kwargs) or cls(); this handles both.
        """
        super().__init__()

        def _get(key: str, default: Any = None):
            if kwargs and key in kwargs:
                return kwargs[key]
            if cfg is not None and key in cfg:
                return cfg.get(key)
            return default

        in_channels = int(_get("in_channels", 1))
        num_classes = int(_get("num_classes", 2))
        dropout = float(_get("dropout", _get("dropout_rate", 0.5)))
        binary_single_logit = bool(_get("binary_single_logit", num_classes == 2))
        out_dim = 1 if binary_single_logit else num_classes

        # Feature extractor
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

        # Head: in_features = 64 after GAP
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(64, out_dim)

        # Keep a note for bias-init routines, if called later
        self._cfg = dict(cfg) if cfg is not None else {}
        self._cfg.update({k: v for k, v in kwargs.items() if k not in self._cfg})

    def param_groups(self):
        # No real backbone/head split here, but keep the interface
        return {"backbone": list(self.features.parameters()),
                "head": list(self.cls_head.parameters())}

    def forward(self, batch: Dict[str, torch.Tensor] | torch.Tensor) -> Dict[str, torch.Tensor]:
        x = batch["image"] if isinstance(batch, dict) else batch
        h = self.features(x)            # [B, 64, H', W']
        h = self.gap(h)                 # [B, 64, 1, 1]
        h = self.flatten(h)             # [B, 64]
        h = self.dropout(h)             # [B, 64]
        logits = self.cls_head(h)       # [B, 1] or [B, C]
        return {"cls_logits": logits}
