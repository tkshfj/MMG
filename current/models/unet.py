# unet.py
from typing import Any, List, Dict, Callable
from models.model_base import BaseModel


class UNetModel(BaseModel):
    from monai.networks.nets import UNet

    def build_model(self, config: Any) -> Any:
        self.config = config
        self.in_channels = int(self._cfg_get(config, "in_channels", 1))
        self.seg_out_channels = int(self._cfg_get(config, "out_channels", 1))
        return self.UNet(
            spatial_dims=2,
            in_channels=self.in_channels,
            out_channels=self.seg_out_channels,
            channels=self._cfg_get(config, "features", (32, 64, 128, 256, 512)),
            strides=self._cfg_get(config, "strides", (2, 2, 2, 2)),
            num_res_units=self._cfg_get(config, "num_res_units", 2),
        )

    def get_output_transform(self):
        # Use segmentation output transform
        return self.get_seg_output_transform()

    def get_supported_tasks(self) -> List[str]:
        return ["segmentation"]

    def get_loss_fn(self) -> Callable:
        from monai.losses import DiceLoss
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": True,
            "num_classes": int(self._get("seg_out_channels", self._cfg_get(self.config, "out_channels", 1))),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }
