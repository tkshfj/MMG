# multitask_unet.py
from typing import Any, Dict, Mapping, Optional, Tuple
import logging
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from models.model_base import BaseModel

logger = logging.getLogger(__name__)


class MultitaskUNetModel(nn.Module, BaseModel):
    """
    MONAI U-Net for segmentation with an added classification head taken from the encoder bottleneck.

    Contract:
      - forward(...) returns a dict with raw logits ONLY:
            {
              "logits":     [B, 1 or C],     # classification logits (alias)
              "cls_logits": [B, 1 or C],     # classification logits (primary)
              "seg_logits": [B, seg_C, H, W] # segmentation logits
            }
      - No sigmoid/softmax here (handled externally).
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        super().__init__()
        cfg = dict(cfg) if cfg is not None else {}

        # Pull knobs from kwargs first; fall back to cfg
        in_ch: int = int(kwargs.get("in_channels", cfg.get("in_channels", 1)))
        seg_out_ch: int = int(kwargs.get("out_channels", cfg.get("out_channels", 1)))
        num_classes_req: int = int(kwargs.get("num_classes", cfg.get("num_classes", 2)))
        spatial_dims: int = int(kwargs.get("spatial_dims", cfg.get("spatial_dims", 2)))
        features: Tuple[int, ...] = tuple(kwargs.get("features", cfg.get("features", (32, 64, 128, 256, 512))))

        # For binary classification, prefer a single-logit head (BCEWithLogits)
        binary_single: bool = bool(kwargs.get(
            "binary_single_logit",
            cfg.get("binary_single_logit", num_classes_req == 2)
        ))
        cls_out_dim: int = 1 if (num_classes_req == 2 and binary_single) else num_classes_req

        # Segmentation UNet (raw logits out)
        strides = tuple([2] * (len(features) - 1))
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=seg_out_ch,  # raw seg logits
            channels=features,
            strides=strides,
            num_res_units=2,
        )

        # Classification head: pooled bottleneck features -> Linear (raw cls logits)
        self.spatial_dims = spatial_dims
        self.gap = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(features[-1], cls_out_dim)
        # Expose registry-friendly alias (so head discovery is robust)
        self.classification_head = self.classifier

        # Bottleneck capture
        self._bottleneck: Optional[torch.Tensor] = None
        self._register_bottleneck_hook(features[-1])

        # Config snapshot for downstream (PosProbCfg, bias init, etc.)
        self.config: Dict[str, Any] = dict(cfg)
        for k, v in kwargs.items():
            self.config.setdefault(k, v)
        # Provide the standard fields PosProbCfg expects for classification
        self.config.setdefault("num_classes", int(num_classes_req))
        self.config.setdefault("positive_index", 1)
        self.config.setdefault("head_logits", int(cls_out_dim))  # 1 => BCE1; >=2 => CE

    def param_groups(self) -> Dict[str, list]:
        """Split params for optimizer group scaling (backbone vs head)."""
        backbones, heads = [], []
        head_params = set(p for p in self.classifier.parameters())
        for p in self.parameters():
            (heads if p in head_params else backbones).append(p)
        return {"backbone": backbones, "head": heads}

    # encoder bottleneck capture
    def _register_bottleneck_hook(self, bottleneck_channels: int) -> None:
        Conv = nn.Conv3d if self.spatial_dims == 3 else nn.Conv2d
        last = None
        for m in self.unet.modules():
            if isinstance(m, Conv) and getattr(m, "out_channels", None) == bottleneck_channels:
                last = m
        if last is None:
            raise RuntimeError("Could not locate a bottleneck Conv layer for hook registration.")
        last.register_forward_hook(self._save_bottleneck)

    def _save_bottleneck(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self._bottleneck = output

    def forward(self, batch: Dict[str, torch.Tensor] | torch.Tensor) -> Dict[str, torch.Tensor]:
        # Inputs pass through without forcing dtype (AMP-safe)
        x = batch["image"] if isinstance(batch, dict) else batch

        self._bottleneck = None
        seg_logits = self.unet(x)  # raw seg logits [B, seg_C, ...]
        if self._bottleneck is None:
            raise RuntimeError("Bottleneck feature was not set by the forward hook.")

        pooled = self.gap(self._bottleneck)          # [B, C, 1, 1]
        pooled = pooled.reshape(pooled.size(0), -1)  # [B, C]
        cls_logits = self.classifier(pooled)         # raw cls logits [B, 1 or C]

        # Sanity checks
        if cls_logits.ndim != 2:
            raise RuntimeError(f"[MultitaskUNetModel] Expected cls_logits [B, K], got {tuple(cls_logits.shape)}")
        if seg_logits.ndim not in (4, 5):
            raise RuntimeError(f"[MultitaskUNetModel] Expected seg_logits [B, C, ...], got {tuple(seg_logits.shape)}")

        # IMPORTANT: return logits only; provide standard keys for extractor
        return {
            "logits": cls_logits,       # alias for extractor
            "cls_logits": cls_logits,   # primary classification logits
            "seg_logits": seg_logits    # raw segmentation logits
        }
