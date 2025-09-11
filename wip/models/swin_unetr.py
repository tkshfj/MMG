# swin_unetr.py
import logging
from typing import Any, Sequence, Tuple
from models.model_base import BaseModel
from monai.networks.nets import SwinUNETR

logger = logging.getLogger(__name__)


class SwinUNETRModel(BaseModel):
    """
    SwinUNETR segmentation wrapper compatible with the refactored BaseModel API.

    - Builds a MONAI SwinUNETR for 2D/3D (img_size length decides).
    - Uses BaseModel.get_loss_fn(task, cfg) (do not override).
    - Exposes seg_output_transform for evaluator metrics.
    - Aligns config keys (norm_name/norm_layer, out_channels/num_classes).
    """

    # helpers
    @staticmethod
    def _as_tuple(x: Any) -> Tuple[int, ...]:
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            return tuple(int(v) for v in x)
        return (int(x),)

    @staticmethod
    def _ensure_len(x: Sequence[int], n: int, name: str) -> Tuple[int, ...]:
        t = tuple(int(v) for v in x)
        if len(t) != n:
            raise ValueError(f"{name} must have length {n}, got {t}")
        return t

    @staticmethod
    def _resolve_norm_name(cfg: Any) -> str:
        # Accept legacy "norm_layer" but pass as "norm_name" to MONAI
        return str(BaseModel._cfg_get(cfg, "norm_name", BaseModel._cfg_get(cfg, "norm_layer", "instance")))

    # BaseModel API
    def build_model(self, config: Any) -> Any:
        self.config = config

        # 2D by default; pass (D,H,W) for 3D
        img_size = self._as_tuple(self._cfg_get(config, "img_size", self._cfg_get(config, "input_shape", (256, 256))))
        if len(img_size) not in (2, 3):
            raise ValueError(f"img_size must have length 2 or 3, got {img_size}")

        in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        # Prefer explicit seg out_channels; fall back to num_classes
        self.num_classes: int = int(self._cfg_get(config, "out_channels", self._cfg_get(config, "num_classes", 1)))

        feature_size: int = int(self._cfg_get(config, "feature_size", 48))
        depths = self._as_tuple(self._cfg_get(config, "depths", (2, 2, 6, 2)))
        num_heads = self._as_tuple(self._cfg_get(config, "num_heads", (3, 6, 12, 24)))
        window_size: int = int(self._cfg_get(config, "window_size", 7))
        mlp_ratio: float = float(self._cfg_get(config, "mlp_ratio", 4.0))
        use_checkpoint: bool = bool(self._cfg_get(config, "use_checkpoint", False))
        norm_name: str = self._resolve_norm_name(config)

        # Swin has 4 stages
        try:
            depths = self._ensure_len(depths, 4, "depths")
            num_heads = self._ensure_len(num_heads, 4, "num_heads")
        except ValueError as e:
            logger.warning("SwinUNETR config: %s; falling back to (2,2,6,2)/(3,6,12,24).", e)
            depths, num_heads = (2, 2, 6, 2), (3, 6, 12, 24)

        if self._cfg_get(config, "debug", False):
            logger.info(
                "SwinUNETR DEBUG: img_size=%s in=%d out=%d feat=%d depths=%s heads=%s win=%d mlp=%.2f ckpt=%s norm=%s",
                img_size, in_channels, self.num_classes, feature_size, depths, num_heads, window_size,
                mlp_ratio, use_checkpoint, norm_name
            )

        # Spatial dims inferred from len(img_size)
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=self.num_classes,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            norm_name=norm_name,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )
        return model

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("segmentation",)

    # Optional extractor for callers that hand back dicts
    def extract_logits(self, y_pred: Any):
        if isinstance(y_pred, dict):
            for k in ("seg_logits", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        return y_pred

    # Ensure BaseModel.get_num_classes sees the seg channels correctly
    def get_num_classes(self, config=None) -> int:
        cfg = self._cfg(config)
        return int(self._cfg_get(cfg, "out_channels", self._cfg_get(cfg, "num_classes", getattr(self, "num_classes", 1))))
