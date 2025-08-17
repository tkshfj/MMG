# densenet121.py
import logging
from typing import Any, List, Dict, Callable

import torch
from models.model_base import BaseModel
from metrics_utils import cls_output_transform
from monai.networks.nets import DenseNet121

logger = logging.getLogger(__name__)


class DenseNet121Model(BaseModel):
    """
    DenseNet121 classifier wrapper compatible with the refactored BaseModel API.

    - Stores config on instance for BaseModel helpers.
    - Uses BaseModel.get_loss_fn(task, cfg) (do not override).
    - Exposes cls_output_transform for metrics.
    """

    # Registry / construction
    def build_model(self, config: Any) -> Any:
        self.config = config
        in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        self.num_classes: int = int(self._cfg_get(config, "num_classes", 2))
        pretrained: bool = bool(self._cfg_get(config, "pretrained", False))

        # Optional: derive CE class_weights from class_counts and store on cfg
        counts = self._cfg_get(config, "class_counts", None)
        if counts and isinstance(counts, (list, tuple)) and len(counts) == self.num_classes:
            counts_t = torch.tensor(counts, dtype=torch.float32)
            # inverse frequency, normalized to mean=1
            weights = counts_t.sum() / (len(counts_t) * counts_t.clamp_min(1))
            weights = weights / weights.mean()
            if isinstance(config, dict):
                config["class_weights"] = weights.tolist()
            else:
                setattr(config, "class_weights", weights.tolist())
            logger.info(
                "[DenseNet121] class_counts=%s -> CE class_weights=%s",
                counts, [round(float(w), 4) for w in weights.tolist()]
            )

        model = DenseNet121(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=self.num_classes,
            pretrained=pretrained,
        )
        return model

    def get_supported_tasks(self) -> List[str]:
        # Classification only
        return ["classification"]

    def get_cls_output_transform(self) -> Callable:
        # Provide the standard logits->labels transform for metrics
        return cls_output_transform

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Keep a minimal, uniform signature with other wrappers
        return {
            "num_classes": self.get_num_classes(),
            "cls_output_transform": cls_output_transform,
            "seg_output_transform": None,  # not used here
        }

    # Optional convenience for external callers (tensor passthrough or common keys)
    def extract_logits(self, y_pred: Any):
        if isinstance(y_pred, dict):
            for k in ("class_logits", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        return y_pred

# from typing import Any, List, Dict, Callable
# from models.model_base import BaseModel
# class DenseNet121Model(BaseModel):
#     from monai.networks.nets import DenseNet121
#     def build_model(self, config: Any) -> Any:
#         self.config = config
#         in_channels = int(self._cfg_get(config, "in_channels", 1))
#         self.num_classes = int(self._cfg_get(config, "num_classes", 2))
#         return self.DenseNet121(
#             spatial_dims=2,
#             in_channels=in_channels,
#             out_channels=self.num_classes,
#             pretrained=bool(self._cfg_get(config, "pretrained", False)),
#         )

#     def get_supported_tasks(self) -> List[str]:
#         return ["classification"]

#     def get_handler_kwargs(self) -> Dict[str, Any]:
#         # Classification: no segmentation metrics
#         return {
#             "add_segmentation_metrics": False,
#             "num_classes": self.get_num_classes(),
#             "seg_output_transform": None,
#             "dice_name": "val_dice",
#             "iou_name": "val_iou",
#         }
