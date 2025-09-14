# models/swin_unetr.py
import inspect
import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from models.model_base import BaseModel

logger = logging.getLogger(__name__)


class SwinUNETRModel(nn.Module, BaseModel):
    """SwinUNETR segmentation wrapper aligned with SimpleCNN/DenseNet121 style."""

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

        def _ensure_len(x: Sequence[int], n: int, name: str) -> Tuple[int, ...]:
            t = tuple(int(v) for v in x)
            if len(t) != n:
                raise ValueError(f"{name} must have length {n}, got {t}")
            return t

        def _resolve_norm_name() -> str:
            return str(_get("norm_name", _get("norm_layer", "instance")))

        # Core config
        img_size = _as_tuple(_get("img_size", _get("input_shape", (256, 256))))
        if len(img_size) not in (2, 3):
            raise ValueError(f"img_size must have length 2 or 3, got {img_size}")

        in_channels = int(_get("in_channels", 1))
        out_channels = int(_get("out_channels", _get("num_classes", 2)))

        feature_size = int(_get("feature_size", 48))
        depths = _as_tuple(_get("depths", (2, 2, 6, 2)))
        num_heads = _as_tuple(_get("num_heads", (3, 6, 12, 24)))
        window_size = int(_get("window_size", 7))
        mlp_ratio = float(_get("mlp_ratio", 4.0))
        use_checkpoint = bool(_get("use_checkpoint", False))
        norm_name = _resolve_norm_name()
        drop_rate = float(_get("drop_rate", 0.0))
        drop_path_rate = float(_get("drop_path_rate", 0.0))

        try:
            depths = _ensure_len(depths, 4, "depths")
            num_heads = _ensure_len(num_heads, 4, "num_heads")
        except ValueError as e:
            logger.warning(
                "SwinUNETR config: %s; falling back to depths=(2,2,6,2), heads=(3,6,12,24).",
                e,
            )
            depths, num_heads = (2, 2, 6, 2), (3, 6, 12, 24)

        if bool(_get("debug", False)):
            logger.info(
                "SwinUNETR DEBUG: img_size=%s in=%d out=%d feat=%d depths=%s heads=%s win=%d "
                "mlp=%.2f ckpt=%s norm=%s drop=%.2f drop_path=%.2f",
                img_size, in_channels, out_channels, feature_size, depths, num_heads,
                window_size, mlp_ratio, use_checkpoint, norm_name, drop_rate, drop_path_rate,
            )

        spatial_dims = 3 if len(img_size) == 3 else 2

        # Ensure feature_size % 12 == 0
        fs = feature_size
        if fs % 12 != 0:
            fs_aligned = max(12, int(round(fs / 12)) * 12)
            if fs_aligned != fs:
                logger.warning("SwinUNETR: feature_size=%d not divisible by 12; aligning to %d.", fs, fs_aligned)
            fs = fs_aligned

        # Build once with signature-filtered kwargs (kwargs-only)
        candidate = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=fs,
            use_checkpoint=use_checkpoint,
            norm_name=norm_name,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            spatial_dims=spatial_dims,  # only passed if supported
            img_size=img_size,          # only passed if supported
        )
        sig = inspect.signature(SwinUNETR)
        accepted = set(sig.parameters.keys())
        init_kwargs = {k: v for k, v in candidate.items() if k in accepted}
        logger.debug("SwinUNETR signature params=%s", list(accepted))

        # Pure kwargs—no positional fallback
        try:
            self.net = SwinUNETR(**init_kwargs)
        except TypeError as e:
            # Last-resort: drop img_size/spatial_dims and retry
            fallback = {k: v for k, v in init_kwargs.items() if k not in ("img_size", "spatial_dims")}
            self.net = SwinUNETR(**fallback)
            logger.warning(
                "SwinUNETR init fell back without img_size/spatial_dims due to: %s. "
                "Consider pinning MONAI to a compatible version.",
                e,
            )

        # Snapshot config once
        config_snapshot: Dict[str, Any] = dict(cfg) if cfg is not None else {}
        for k, v in (kwargs or {}).items():
            config_snapshot.setdefault(k, v)
        config_snapshot.setdefault("seg_num_classes", int(out_channels))
        config_snapshot.setdefault("out_channels", int(out_channels))
        config_snapshot.setdefault("num_classes", int(out_channels))
        self.config = config_snapshot

    def forward(self, batch: Dict[str, torch.Tensor] | torch.Tensor) -> Dict[str, torch.Tensor]:
        x = batch["image"] if isinstance(batch, dict) else batch
        seg_logits = self.net(x)
        return {"logits": seg_logits, "seg_logits": seg_logits}

    def param_groups(self) -> Dict[str, list]:
        backbone, head = [], []
        for name, p in self.net.named_parameters():
            (backbone if name.startswith("swinViT") else head).append(p)
        if not head:
            backbone = list(self.net.parameters())
        return {"backbone": backbone, "head": head}

    def get_supported_tasks(self) -> list[str]:
        return ["segmentation"]

    def get_num_classes(self, config=None) -> int:
        cfg = self._cfg(config)
        return int(self._cfg_get(cfg, "out_channels", self._cfg_get(cfg, "seg_num_classes", 1)))

    def extract_logits(self, y_pred: Any):
        if isinstance(y_pred, dict):
            for k in ("seg_logits", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        return y_pred
