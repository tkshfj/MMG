# densenet121.py
from typing import Any, List, Dict, Callable
from models.model_base import BaseModel


class DenseNet121Model(BaseModel):
    from monai.networks.nets import DenseNet121

    def build_model(self, config: Any) -> Any:
        self.config = config
        in_channels = int(self._cfg_get(config, "in_channels", 1))
        self.num_classes = int(self._cfg_get(config, "num_classes", 2))
        return self.DenseNet121(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=self.num_classes,
            pretrained=bool(self._cfg_get(config, "pretrained", False)),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    def get_loss_fn(self) -> Callable:
        return self._make_class_weighted_ce()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Classification: no segmentation metrics
        return {
            "add_segmentation_metrics": False,
            "num_classes": self.get_num_classes(),
            "seg_output_transform": None,
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }
