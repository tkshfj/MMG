# unet.py
import logging
from typing import Any, List, Dict, Callable, Sequence, Tuple

from models.model_base import BaseModel
from metrics_utils import seg_output_transform
from monai.networks.nets import UNet

logger = logging.getLogger(__name__)


class UNetModel(BaseModel):
    """
    UNet segmentation wrapper compatible with the refactored BaseModel API.

    - Stores config on instance for BaseModel helpers.
    - Uses BaseModel.get_loss_fn(task, cfg) (do not override).
    - Exposes seg_output_transform for evaluator metrics.
    """

    # helpers
    @staticmethod
    def _as_tuple(x: Any) -> Tuple[int, ...]:
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            return tuple(int(v) for v in x)
        return (int(x),)

    @staticmethod
    def _fix_strides_for_channels(channels: Sequence[int], strides: Sequence[int] | None) -> Tuple[int, ...]:
        depth = len(channels)
        expected = max(0, depth - 1)
        if strides is None:
            return tuple(2 for _ in range(expected))
        s = tuple(int(v) for v in strides)
        if len(s) != expected:
            logger.warning(
                "UNet: len(strides)=%d does not match len(channels)-1=%d; "
                "using strides=(2,)*%d",
                len(s), expected, expected,
            )
            return tuple(2 for _ in range(expected))
        return s

    # BaseModel API
    def build_model(self, config: Any) -> Any:
        self.config = config

        spatial_dims: int = int(self._cfg_get(config, "spatial_dims", 2))
        in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        # For consistency across the codebase, treat segmentation classes as num_classes.
        self.num_classes: int = int(self._cfg_get(config, "out_channels", 1))

        channels = self._as_tuple(self._cfg_get(config, "features", (32, 64, 128, 256, 512)))
        strides = self._cfg_get(config, "strides", None)
        strides = self._fix_strides_for_channels(channels, strides)

        num_res_units: int = int(self._cfg_get(config, "num_res_units", 2))

        if self._cfg_get(config, "debug", False):
            logger.info(
                "UNet DEBUG: spatial_dims=%d in_channels=%d num_classes=%d channels=%s strides=%s num_res_units=%d",
                spatial_dims, in_channels, self.num_classes, channels, strides, num_res_units,
            )

        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.num_classes,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )
        return model

    def get_supported_tasks(self) -> List[str]:
        return ["segmentation"]

    def get_seg_output_transform(self) -> Callable:
        # Standard logits -> (y_pred, y_true) for segmentation metrics
        return seg_output_transform

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Keep minimal and uniform with other wrappers
        return {
            "num_classes": self.get_num_classes(),     # uses self.num_classes set in build_model
            "cls_output_transform": None,               # not used for pure segmentation
            "seg_output_transform": seg_output_transform,
        }

    # Optional convenience for callers expecting a logits extractor
    def extract_logits(self, y_pred: Any):
        if isinstance(y_pred, dict):
            for k in ("cls_out", "seg_logits", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        return y_pred
