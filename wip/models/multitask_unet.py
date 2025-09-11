# multitask_unet.py (refactored to mirror SimpleCNNModel patterns)
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
    Follows the SimpleCNNModel contract:
      - __init__(cfg: Optional[Mapping], **kwargs) accepts flexible config/kwargs
      - forward(batch|tensor) returns a dict with:
            {"cls_logits": [B, 1 or C], "seg_logits": [B, seg_C, H, W]}
      - No activation in the head (use BCEWithLogits/CE externally)
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        super().__init__()
        cfg = dict(cfg) if cfg is not None else {}
        # Pull knobs from kwargs first (registry may pass these), fall back to cfg, then defaults
        in_ch: int = int(kwargs.get("in_channels", cfg.get("in_channels", 1)))
        seg_out_ch: int = int(kwargs.get("out_channels", cfg.get("out_channels", 1)))
        num_classes_req: int = int(kwargs.get("num_classes", cfg.get("num_classes", 2)))
        spatial_dims: int = int(kwargs.get("spatial_dims", cfg.get("spatial_dims", 2)))
        features: Tuple[int, ...] = tuple(kwargs.get("features", cfg.get("features", (32, 64, 128, 256, 512))))
        # Binary-single-logit for 2-class setups (matches SimpleCNNModel)
        binary_single: bool = bool(kwargs.get(
            "binary_single_logit",
            cfg.get("binary_single_logit", num_classes_req == 2)
        ))
        cls_out_dim: int = 1 if (num_classes_req == 2 and binary_single) else num_classes_req

        # Build UNet for segmentation
        # Strides length must be len(features)-1; default features length 5 => strides of 4
        strides = tuple([2] * (len(features) - 1))
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_ch,
            out_channels=seg_out_ch,
            channels=features,
            strides=strides,
            num_res_units=2,
        )

        # Classification head: pooled bottleneck features -> Linear
        self.spatial_dims = spatial_dims
        self.gap = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(features[-1], cls_out_dim)

        # Internal: store last bottleneck feature via forward hook
        self._bottleneck: Optional[torch.Tensor] = None
        self._register_bottleneck_hook(features[-1])

        # Keep config snapshot for optional downstream routines (e.g., bias init)
        # self._cfg = dict(cfg)
        # self._cfg.update({k: v for k, v in kwargs.items() if k not in self._cfg})
        self.config = dict(cfg) if cfg is not None else {}
        self.config.update({k: v for k, v in kwargs.items() if k not in self.config})
        # self.config = dict(cfg) if cfg is not None else {}
        # for k, v in (kwargs or {}).items():
        #     self.config.setdefault(k, v)

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

    # forward
    def forward(self, batch: Dict[str, torch.Tensor] | torch.Tensor) -> Dict[str, torch.Tensor]:
        x = batch["image"] if isinstance(batch, dict) else batch
        self._bottleneck = None
        seg_out = self.unet(x)  # [B, seg_C, H, W] (or 3D)
        if self._bottleneck is None:
            raise RuntimeError("Bottleneck feature was not set by the forward hook.")
        pooled = self.gap(self._bottleneck)          # [B, C, 1, 1]
        pooled = pooled.reshape(pooled.size(0), -1)  # [B, C]
        cls_logits = self.classifier(pooled)         # [B, 1] or [B, C]

        # Sanity checks to assist debugging
        if cls_logits.ndim != 2:
            raise RuntimeError(f"[MultitaskUNetModel] Expected cls_logits [B, K], got {tuple(cls_logits.shape)}")
        if seg_out.ndim not in (4, 5):
            raise RuntimeError(f"[MultitaskUNetModel] Expected seg_logits [B, C, ...], got {tuple(seg_out.shape)}")

        return {"cls_logits": cls_logits, "seg_logits": seg_out}


# # multitask_unet.py
# from typing import Any, List, Dict, Optional, Tuple
# import torch
# import torch.nn as nn
# from monai.networks.nets import UNet

# from models.model_base import BaseModel
# from metrics_utils import (
#     cls_output_transform,
#     seg_output_transform,
# )


# class MultitaskUNetModel(BaseModel):
#     """
#     Wrapper around a MONAI U-Net with an added classification head.
#     Plays nicely with BaseModel's task-aware loss and the refactored, task-gated metrics.
#     """

#     # Registry / construction
#     def build_model(self, config: Any) -> Any:
#         # Keep config on the instance so BaseModel can read knobs (task, weights, etc.)
#         self.config = config
#         # Sweep weights stored under legacy keys -> alias to what BaseModel reads
#         cls_w = float(self._cfg_get(config, "cls_loss_weight", self._cfg_get(config, "cls_weight", 1.0)))
#         seg_w = float(self._cfg_get(config, "seg_loss_weight", self._cfg_get(config, "seg_weight", 1.0)))
#         # Save locally (optional) and alias in config for the base class
#         self.cls_loss_weight = cls_w
#         self.seg_loss_weight = seg_w
#         self._cfg_set(config, "cls_weight", cls_w)
#         self._cfg_set(config, "seg_weight", seg_w)
#         # Also set alpha/beta aliases for completeness
#         self._cfg_set(config, "alpha", cls_w)
#         self._cfg_set(config, "beta", seg_w)
#         # Model hyperparams
#         self.in_channels: int = int(self._cfg_get(config, "in_channels", 1))
#         self.seg_out_channels: int = int(self._cfg_get(config, "out_channels", 1))  # segmentation C
#         # self.num_classes: int = int(self._cfg_get(config, "num_classes", 2))        # classification C
#         self.num_classes: int = int(self._cfg_get(config, "num_classes", 2))        # classification classes
#         # Prefer single-logit for binary classification (removes channel ambiguity)
#         self.binary_single_logit: bool = bool(self._cfg_get(config, "binary_single_logit", True))
#         cls_out_channels = 1 if (self.num_classes == 2 and self.binary_single_logit) else self.num_classes
#         self.features: Tuple[int, ...] = tuple(self._cfg_get(config, "features", (32, 64, 128, 256, 512)))
#         self.spatial_dims: int = int(self._cfg_get(config, "spatial_dims", 2))

#         return MultiTaskUNet(
#             spatial_dims=self.spatial_dims,
#             in_channels=self.in_channels,
#             out_channels=self.seg_out_channels,  # segmentation head channels
#             cls_out_channels=cls_out_channels,   # 1 for binary-single-logit; C for multi-class
#             features=self.features,
#         )

#     # Capabilities
#     def get_supported_tasks(self) -> List[str]:
#         # The wrapper supports all three modes; active mode is chosen in cfg["task"]
#         return ["classification", "segmentation", "multitask"]

#     # Handlers / metrics wiring (keep neutral; attach_metrics handles gating)
#     def get_handler_kwargs(self) -> Dict[str, Any]:
#         # Avoid forcing metric flags here
#         return {
#             "num_classes": self.get_num_classes(),
#             "cls_output_transform": cls_output_transform,
#             "seg_output_transform": seg_output_transform,
#         }

#     # logits extraction for external callers
#     def extract_logits(self, y_pred: Any) -> torch.Tensor:
#         """
#         Robustly pull classification logits from containers. Falls back to BaseModel for tensors/tuples.
#         """
#         if isinstance(y_pred, dict):
#             # v = y_pred.get("class_logits")
#             v = y_pred.get("cls_out", y_pred.get("class_logits"))
#             if isinstance(v, torch.Tensor):
#                 return v
#         if isinstance(y_pred, (list, tuple)):
#             for e in y_pred:
#                 # if isinstance(e, dict) and "class_logits" in e and isinstance(e["class_logits"], torch.Tensor):
#                 #     return e["class_logits"]
#                 if isinstance(e, dict):
#                     vv = e.get("cls_out", e.get("class_logits"))
#                     if isinstance(vv, torch.Tensor):
#                         return vv
#                 if isinstance(e, torch.Tensor):
#                     return e
#         return super().extract_logits(y_pred)


# class MultiTaskUNet(nn.Module):
#     """
#     U-Net backbone for segmentation with a classification head drawn from deep encoder features.
#     Forward returns:
#       {
#         "class_logits": [B, num_classes],     # for classification losses/metrics
#         "seg_out":      [B, out_channels, H, W],  # for segmentation losses/metrics
#       }
#     """
#     def __init__(
#         self,
#         spatial_dims: int = 2,
#         in_channels: int = 1,
#         out_channels: int = 1,
#         cls_out_channels: int = 1,
#         features: Tuple[int, ...] = (32, 64, 128, 256, 512),
#     ) -> None:
#         super().__init__()
#         self.unet = UNet(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=out_channels,   # segmentation head
#             channels=features,
#             strides=(2, 2, 2, 2),
#             num_res_units=2,
#         )
#         self.gap = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(features[-1], cls_out_channels)
#         self.cls_out_channels = int(cls_out_channels)
#         self._bottleneck: Optional[torch.Tensor] = None
#         # Register a forward hook on the deepest Conv2d to capture encoder features
#         self._register_bottleneck_hook(features[-1])

#     def _register_bottleneck_hook(self, bottleneck_channels: int) -> None:
#         Conv = nn.Conv3d if isinstance(self.gap, nn.AdaptiveAvgPool3d) else nn.Conv2d
#         last_match = None
#         for name, module in self.unet.named_modules():
#             if isinstance(module, Conv) and module.out_channels == bottleneck_channels:
#                 last_match = module
#         if last_match is not None:
#             last_match.register_forward_hook(self._save_bottleneck)
#             return
#         raise RuntimeError("Could not find a bottleneck Conv layer to register the hook.")

#     def _save_bottleneck(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
#         self._bottleneck = output

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         self._bottleneck = None
#         seg_out = self.unet(x)  # [B, out_channels, H, W]
#         if self._bottleneck is None:
#             raise RuntimeError("Bottleneck feature was not set by forward hook.")
#         pooled = self.gap(self._bottleneck)              # [B, C, 1, 1]
#         pooled = pooled.view(pooled.size(0), -1)         # [B, C]
#         # class_logits = self.classifier(pooled)           # [B, num_classes]
#         cls = self.classifier(pooled)                    # [B, cls_out_channels]
#         # For binary-single-logit, emit shape (B,) to make BCE auto-detect
#         if self.cls_out_channels == 1:
#             cls = cls.squeeze(-1)                        # [B]
#         # Sanity checks
#         if not (cls.ndim == 1 or (cls.ndim == 2 and cls.shape[1] >= 1)):
#             raise RuntimeError(f"[MultiTaskUNet] Unexpected cls_out shape {tuple(cls.shape)}")
#         if seg_out.ndim != 4:
#             raise RuntimeError(f"[MultiTaskUNet] Expected seg_out [B, C, H, W], got {tuple(seg_out.shape)}")
#         return {"cls_out": cls, "seg_logits": seg_out}
