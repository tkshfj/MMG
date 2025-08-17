# multitask_unet.py
from typing import Any, List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from monai.networks.nets import UNet

from models.model_base import BaseModel
from metrics_utils import (
    cls_output_transform,
    seg_output_transform,
)


class MultitaskUNetModel(BaseModel):
    """
    Wrapper around a MONAI U-Net with an added classification head.
    Plays nicely with BaseModel's task-aware loss and the refactored, task-gated metrics.
    """

    # Registry / construction
    def build_model(self, config: Any) -> Any:
        # Keep config on the instance so BaseModel can read knobs (task, weights, etc.)
        self.config = config

        # Sweep weights stored under legacy keys -> alias to what BaseModel reads
        cls_w = float(self._cfg_get(config, "cls_loss_weight", self._cfg_get(config, "cls_weight", 1.0)))
        seg_w = float(self._cfg_get(config, "seg_loss_weight", self._cfg_get(config, "seg_weight", 1.0)))
        # Save locally (optional) and alias in config for the base class
        self.cls_loss_weight = cls_w
        self.seg_loss_weight = seg_w
        self._cfg_set(config, "cls_weight", cls_w)
        self._cfg_set(config, "seg_weight", seg_w)
        # Also set alpha/beta aliases for completeness
        self._cfg_set(config, "alpha", cls_w)
        self._cfg_set(config, "beta", seg_w)

        # Model hyperparams
        self.in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        self.seg_out_channels: int = int(self._cfg_get(config, "out_channels", 1))  # segmentation C
        self.num_classes: int = int(self._cfg_get(config, "num_classes", 2))        # classification C
        self.features: Tuple[int, ...] = tuple(self._cfg_get(config, "features", (32, 64, 128, 256, 512)))
        self.spatial_dims: int = int(self._cfg_get(config, "spatial_dims", 2))

        return MultiTaskUNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.seg_out_channels,  # segmentation head channels
            num_classes=self.num_classes,        # classification classes
            features=self.features,
        )

    # Capabilities
    def get_supported_tasks(self) -> List[str]:
        # The wrapper supports all three modes; active mode is chosen in cfg["task"]
        return ["classification", "segmentation", "multitask"]

    # Handlers / metrics wiring (keep neutral; attach_metrics handles gating)
    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Avoid forcing metric flags here
        return {
            "num_classes": self.get_num_classes(),
            "cls_output_transform": cls_output_transform,
            "seg_output_transform": seg_output_transform,
        }

    # logits extraction for external callers
    def extract_logits(self, y_pred: Any) -> torch.Tensor:
        """
        Robustly pull classification logits from containers. Falls back to BaseModel for tensors/tuples.
        """
        if isinstance(y_pred, dict):
            v = y_pred.get("class_logits")
            if isinstance(v, torch.Tensor):
                return v
        if isinstance(y_pred, (list, tuple)):
            for e in y_pred:
                if isinstance(e, dict) and "class_logits" in e and isinstance(e["class_logits"], torch.Tensor):
                    return e["class_logits"]
                if isinstance(e, torch.Tensor):
                    return e
        return super().extract_logits(y_pred)


class MultiTaskUNet(nn.Module):
    """
    U-Net backbone for segmentation with a classification head drawn from deep encoder features.
    Forward returns:
      {
        "class_logits": [B, num_classes],     # for classification losses/metrics
        "seg_out":      [B, out_channels, H, W],  # for segmentation losses/metrics
      }
    """
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = 2,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,   # segmentation head
            channels=features,
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(features[-1], num_classes)
        self._bottleneck: Optional[torch.Tensor] = None

        # Register a forward hook on the deepest Conv2d to capture encoder features
        self._register_bottleneck_hook(features[-1])

    def _register_bottleneck_hook(self, bottleneck_channels: int) -> None:
        for name, module in self.unet.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == bottleneck_channels:
                module.register_forward_hook(self._save_bottleneck)
                break
        else:
            raise RuntimeError(
                "Could not find a Conv2d layer with the expected bottleneck channels to register the hook."
            )

    def _save_bottleneck(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self._bottleneck = output

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._bottleneck = None
        seg_out = self.unet(x)  # [B, out_channels, H, W]

        if self._bottleneck is None:
            raise RuntimeError("Bottleneck feature was not set by forward hook.")

        pooled = self.gap(self._bottleneck)              # [B, C, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)         # [B, C]
        class_logits = self.classifier(pooled)           # [B, num_classes]

        # Lightweight sanity checks
        if class_logits.ndim != 2:
            raise RuntimeError(f"[MultiTaskUNet] Expected class logits [B, num_classes], got {tuple(class_logits.shape)}")
        if seg_out.ndim != 4:
            raise RuntimeError(f"[MultiTaskUNet] Expected seg_out [B, C, H, W], got {tuple(seg_out.shape)}")
        if class_logits.shape[0] != seg_out.shape[0]:
            raise RuntimeError("Batch size mismatch between classification and segmentation outputs.")

        return {"class_logits": class_logits, "seg_out": seg_out}
