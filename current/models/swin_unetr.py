# swin_unetr.py
from typing import Any, List, Dict, Callable
from models.model_base import BaseModel


class SwinUNETRModel(BaseModel):
    from monai.networks.nets import SwinUNETR

    def build_model(self, config: Any) -> Any:
        self.config = config
        self.in_channels = int(self._cfg_get(config, "in_channels", 1))
        self.seg_out_channels = int(self._cfg_get(config, "out_channels", 1))
        return self.SwinUNETR(
            in_channels=self.in_channels,
            out_channels=self.seg_out_channels,
            img_size=self._cfg_get(config, "img_size", (256, 256)),
            feature_size=self._cfg_get(config, "feature_size", 48),
            use_checkpoint=self._cfg_get(config, "use_checkpoint", False),
            norm_layer=self._cfg_get(config, "norm_layer", "instance"),
            depths=self._cfg_get(config, "depths", (2, 2, 6, 2)),
            num_heads=self._cfg_get(config, "num_heads", (3, 6, 12, 24)),
            window_size=self._cfg_get(config, "window_size", 7),
            mlp_ratio=self._cfg_get(config, "mlp_ratio", 4.0),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["segmentation"]

    def get_output_transform(self):
        # Use segmentation output transform
        return self.get_seg_output_transform()

    def get_loss_fn(self) -> Callable:
        from monai.losses import DiceLoss
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Segmentation: segmentation metrics should be enabled
        return {
            "add_segmentation_metrics": True,
            "num_classes": int(self._get("seg_out_channels", self._cfg_get(self.config, "out_channels", 1))),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }
