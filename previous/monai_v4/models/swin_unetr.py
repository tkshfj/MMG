# swin_unetr.py
import logging
from typing import Any, List, Dict, Callable, Sequence, Tuple

from models.model_base import BaseModel
from metrics_utils import seg_output_transform
from monai.networks.nets import SwinUNETR

logger = logging.getLogger(__name__)


class SwinUNETRModel(BaseModel):
    """
    SwinUNETR segmentation wrapper compatible with the refactored BaseModel API.

    - Stores config on instance for BaseModel helpers.
    - Uses BaseModel.get_loss_fn(task, cfg) (do not override).
    - Exposes seg_output_transform for evaluator metrics.
    """

    # ---- helpers ----
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
        # Accept either "norm_name" or legacy "norm_layer" in config; pass as norm_name to MONAI
        return str(
            BaseModel._cfg_get(cfg, "norm_name", BaseModel._cfg_get(cfg, "norm_layer", "instance"))
        )

    # ---- BaseModel API ----
    def build_model(self, config: Any) -> Any:
        self.config = config

        # 2D by default in this project; pass a 3-tuple if we use 3D
        img_size = self._cfg_get(config, "img_size", self._cfg_get(config, "input_shape", (256, 256)))
        img_size = self._as_tuple(img_size)
        if len(img_size) not in (2, 3):
            raise ValueError(f"img_size must have length 2 or 3, got {img_size}")

        in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        self.num_classes: int = int(self._cfg_get(config, "out_channels", 1))

        feature_size: int = int(self._cfg_get(config, "feature_size", 48))
        depths = self._as_tuple(self._cfg_get(config, "depths", (2, 2, 6, 2)))
        num_heads = self._as_tuple(self._cfg_get(config, "num_heads", (3, 6, 12, 24)))
        window_size: int = int(self._cfg_get(config, "window_size", 7))
        mlp_ratio: float = float(self._cfg_get(config, "mlp_ratio", 4.0))
        use_checkpoint: bool = bool(self._cfg_get(config, "use_checkpoint", False))
        norm_name: str = self._resolve_norm_name(config)

        # Swin stages are 4; sanity-check tuple lengths
        try:
            depths = self._ensure_len(depths, 4, "depths")
            num_heads = self._ensure_len(num_heads, 4, "num_heads")
        except ValueError as e:
            logger.warning("SwinUNETR config: %s; falling back to (2,2,6,2)/(3,6,12,24).", e)
            depths = (2, 2, 6, 2)
            num_heads = (3, 6, 12, 24)

        if self._cfg_get(config, "debug", False):
            logger.info(
                "SwinUNETR DEBUG: img_size=%s in_channels=%d num_classes=%d "
                "feature_size=%d depths=%s heads=%s window_size=%d mlp_ratio=%.2f use_ckpt=%s norm=%s",
                img_size, in_channels, self.num_classes,
                feature_size, depths, num_heads, window_size, mlp_ratio, use_checkpoint, norm_name
            )

        # Note: Many MONAI versions infer spatial dims from len(img_size); no need to pass spatial_dims explicitly.
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=self.num_classes,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            norm_name=norm_name,          # accept legacy config 'norm_layer' via _resolve_norm_name
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )
        return model

    def get_supported_tasks(self) -> List[str]:
        return ["segmentation"]

    def get_seg_output_transform(self) -> Callable:
        # Standard logits -> (y_pred, y_true) for segmentation metrics
        return seg_output_transform

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Minimal and uniform with other wrappers
        return {
            "num_classes": self.get_num_classes(),
            "cls_output_transform": None,               # not used for pure segmentation
            "seg_output_transform": seg_output_transform,
        }

    # Optional convenience for callers expecting a logits extractor
    def extract_logits(self, y_pred: Any):
        if isinstance(y_pred, dict):
            for k in ("seg_logits", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        return y_pred

# from typing import Any, List, Dict, Callable
# from models.model_base import BaseModel


# class SwinUNETRModel(BaseModel):
#     from monai.networks.nets import SwinUNETR

#     def build_model(self, config: Any) -> Any:
#         self.config = config
#         self.in_channels = int(self._cfg_get(config, "in_channels", 1))
#         self.seg_out_channels = int(self._cfg_get(config, "out_channels", 1))
#         return self.SwinUNETR(
#             in_channels=self.in_channels,
#             out_channels=self.seg_out_channels,
#             img_size=self._cfg_get(config, "img_size", (256, 256)),
#             feature_size=self._cfg_get(config, "feature_size", 48),
#             use_checkpoint=self._cfg_get(config, "use_checkpoint", False),
#             norm_layer=self._cfg_get(config, "norm_layer", "instance"),
#             depths=self._cfg_get(config, "depths", (2, 2, 6, 2)),
#             num_heads=self._cfg_get(config, "num_heads", (3, 6, 12, 24)),
#             window_size=self._cfg_get(config, "window_size", 7),
#             mlp_ratio=self._cfg_get(config, "mlp_ratio", 4.0),
#         )

#     def get_supported_tasks(self) -> List[str]:
#         return ["segmentation"]

#     def get_output_transform(self):
#         # Use segmentation output transform
#         return self.get_seg_output_transform()

#     def get_loss_fn(self) -> Callable:
#         from monai.losses import DiceLoss
#         return DiceLoss(to_onehot_y=True, softmax=True)

#     def get_handler_kwargs(self) -> Dict[str, Any]:
#         # Segmentation: segmentation metrics should be enabled
#         return {
#             "add_segmentation_metrics": True,
#             "num_classes": int(self._get("seg_out_channels", self._cfg_get(self.config, "out_channels", 1))),
#             "seg_output_transform": self.get_seg_output_transform(),
#             "dice_name": "val_dice",
#             "iou_name": "val_iou",
#         }
