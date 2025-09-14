# models/unet.py
import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from models.model_base import BaseModel

logger = logging.getLogger(__name__)


class UNetModel(nn.Module, BaseModel):
    """
    2D UNet segmentation wrapper aligned with SimpleCNN/DenseNet121/SwinUNETR style.

    - __init__(cfg, **kwargs): constructs modules and snapshots config
    - forward(...): returns {"logits": seg_logits, "seg_logits": seg_logits}
    - param_groups(): single group (seg usually doesn't split backbone/head)
    - Strides auto-derived from len(features) unless provided
    - Supports norm/act/dropout knobs from cfg
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()

        def _get(key: str, default: Any = None):
            if kwargs and key in kwargs:
                return kwargs[key]
            if cfg is not None and key in cfg:
                return cfg.get(key)
            return default

        def _as_tuple(x: Any) -> Tuple[int, ...]:
            if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
                return tuple(int(v) for v in x)
            return (int(x),)

        def _derive_strides_from_channels(channels: Sequence[int], strides: Optional[Sequence[int]]) -> Tuple[int, ...]:
            depth = len(channels)
            expected = max(0, depth - 1)
            if strides is None:
                return tuple(2 for _ in range(expected))
            s = tuple(int(v) for v in strides)
            if len(s) != expected:
                logger.warning(
                    "UNet: len(strides)=%d does not match len(features)-1=%d; using strides=(2,)*%d",
                    len(s), expected, expected,
                )
                return tuple(2 for _ in range(expected))
            return s

        # Core config (2D segmentation)
        spatial_dims = int(_get("spatial_dims", 2))
        if spatial_dims != 2:
            logger.info("UNetModel forcing spatial_dims=2 (was %s).", spatial_dims)
            spatial_dims = 2

        in_channels = int(_get("in_channels", 1))
        out_channels = int(_get("out_channels", _get("num_classes", 2)))

        features = _as_tuple(_get("features", (32, 64, 128, 256, 512)))
        strides = _get("strides", None)
        strides = _derive_strides_from_channels(features, strides)

        num_res_units = int(_get("num_res_units", 2))
        # MONAI accepts "instance"/"batch" or tuple forms; pass through
        norm = _get("norm", "instance")
        act = _get("act", "prelu")
        dropout = float(_get("dropout", 0.0))

        if bool(_get("debug", False)):
            logger.info(
                "UNet DEBUG: spatial_dims=%d in=%d out=%d features=%s strides=%s "
                "num_res_units=%d norm=%s act=%s dropout=%.3f",
                spatial_dims, in_channels, out_channels, features, strides,
                num_res_units, str(norm), str(act), dropout,
            )

        # Build MONAI UNet (raw logits; no activation)
        self.net = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
            dropout=dropout,
        )

        # Snapshot config for downstream helpers
        config_snapshot: Dict[str, Any] = dict(cfg) if cfg is not None else {}
        for k, v in (kwargs or {}).items():
            config_snapshot.setdefault(k, v)
        config_snapshot.setdefault("seg_num_classes", int(out_channels))
        config_snapshot.setdefault("out_channels", int(out_channels))
        config_snapshot.setdefault("num_classes", int(out_channels))  # safe alias for seg
        self.config = config_snapshot

    def forward(self, batch: Dict[str, torch.Tensor] | torch.Tensor) -> Dict[str, torch.Tensor]:
        x = batch["image"] if isinstance(batch, dict) else batch
        seg_logits = self.net(x)  # [B, C, H, W]
        return {"logits": seg_logits, "seg_logits": seg_logits}

    def param_groups(self) -> Dict[str, list]:
        # Single param group for segmentation by default
        return {"backbone": list(self.net.parameters())}

    def get_supported_tasks(self) -> list[str]:
        return ["segmentation"]

    def get_num_classes(self, config=None) -> int:
        cfg = self._cfg(config)
        return int(self._cfg_get(cfg, "out_channels", self._cfg_get(cfg, "seg_num_classes", 1)))

    # Optional convenience extractor
    def extract_logits(self, y_pred: Any):
        if isinstance(y_pred, dict):
            for k in ("seg_logits", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        return y_pred


# import logging
# from typing import Any, List, Dict, Sequence, Tuple

# from models.model_base import BaseModel
# from metrics_utils import seg_output_transform
# from monai.networks.nets import UNet

# logger = logging.getLogger(__name__)


# class UNetModel(BaseModel):
#     """
#     UNet segmentation wrapper compatible with the refactored BaseModel API.

#     - Stores config on instance for BaseModel helpers.
#     - Uses BaseModel.get_loss_fn(task, cfg) (do not override).
#     - Exposes seg_output_transform for evaluator metrics.
#     """

#     # helpers
#     @staticmethod
#     def _as_tuple(x: Any) -> Tuple[int, ...]:
#         if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
#             return tuple(int(v) for v in x)
#         return (int(x),)

#     @staticmethod
#     def _fix_strides_for_channels(channels: Sequence[int], strides: Sequence[int] | None) -> Tuple[int, ...]:
#         depth = len(channels)
#         expected = max(0, depth - 1)
#         if strides is None:
#             return tuple(2 for _ in range(expected))
#         s = tuple(int(v) for v in strides)
#         if len(s) != expected:
#             logger.warning(
#                 "UNet: len(strides)=%d does not match len(channels)-1=%d; "
#                 "using strides=(2,)*%d",
#                 len(s), expected, expected,
#             )
#             return tuple(2 for _ in range(expected))
#         return s

#     # BaseModel API
#     def build_model(self, config: Any) -> Any:
#         self.config = config

#         spatial_dims: int = int(self._cfg_get(config, "spatial_dims", 2))
#         in_channels: int = int(self._cfg_get(config, "in_channels", 1))
#         # For consistency across the codebase, treat segmentation classes as num_classes.
#         self.num_classes: int = int(self._cfg_get(config, "out_channels", 1))

#         channels = self._as_tuple(self._cfg_get(config, "features", (32, 64, 128, 256, 512)))
#         strides = self._cfg_get(config, "strides", None)
#         strides = self._fix_strides_for_channels(channels, strides)

#         num_res_units: int = int(self._cfg_get(config, "num_res_units", 2))

#         if self._cfg_get(config, "debug", False):
#             logger.info(
#                 "UNet DEBUG: spatial_dims=%d in_channels=%d num_classes=%d channels=%s strides=%s num_res_units=%d",
#                 spatial_dims, in_channels, self.num_classes, channels, strides, num_res_units,
#             )

#         model = UNet(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=self.num_classes,
#             channels=channels,
#             strides=strides,
#             num_res_units=num_res_units,
#         )
#         return model

#     def get_supported_tasks(self) -> List[str]:
#         return ["segmentation"]

#     def get_handler_kwargs(self) -> Dict[str, Any]:
#         # Keep minimal and uniform with other wrappers
#         return {
#             "num_classes": self.get_num_classes(),     # uses self.num_classes set in build_model
#             "cls_output_transform": None,               # not used for pure segmentation
#             "seg_output_transform": seg_output_transform,
#         }

#     # Optional convenience for callers expecting a logits extractor
#     def extract_logits(self, y_pred: Any):
#         if isinstance(y_pred, dict):
#             for k in ("cls_out", "seg_logits", "logits", "y_pred"):
#                 v = y_pred.get(k, None)
#                 if v is not None:
#                     return v
#         return y_pred
