# model_utils.py
from typing import Any, Callable, Dict, List
from torch.optim import Adam, SGD, RMSprop, Optimizer
from model_protocol import ModelRegistryProtocol
from metrics_utils import (
    cls_output_transform,
    cls_confmat_output_transform,
    auc_output_transform,
    seg_output_transform,
    seg_confmat_output_transform,
    make_metrics
)


class BaseModel(ModelRegistryProtocol):
    def get_num_classes(self, config=None):
        # Define a generic way to fetch num_classes for each model
        if hasattr(self, "num_classes"):
            return self.num_classes
        if config is not None and hasattr(config, "num_classes"):
            return config.num_classes
        return 2  # default fallback

    def get_output_transform(self):
        return cls_output_transform

    def get_cls_output_transform(self):
        return cls_output_transform

    def get_cls_confmat_output_transform(self):
        return cls_confmat_output_transform

    def get_auc_output_transform(self):
        return auc_output_transform

    def get_seg_output_transform(self):
        return seg_output_transform

    def get_seg_confmat_output_transform(self):
        return seg_confmat_output_transform

    def get_metrics(self, config=None) -> Dict[str, Any]:
        tasks = self.get_supported_tasks()
        num_classes = self.get_num_classes(config)
        return make_metrics(
            tasks=tasks,
            num_classes=num_classes,
            loss_fn=self.get_loss_fn(),
            cls_output_transform=self.get_cls_output_transform(),
            auc_output_transform=self.get_auc_output_transform(),
            seg_confmat_output_transform=self.get_seg_confmat_output_transform()
        )


# SimpleCNN
class SimpleCNNModel(BaseModel):
    from models.simple_cnn import SimpleCNN

    def build_model(self, config: Any) -> Any:
        in_channels = getattr(config, "in_channels", 1)
        num_classes = getattr(config, "num_classes", 2)
        return self.SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    # def get_metrics(self) -> Dict[str, Any]:
    #     from ignite.metrics import Accuracy, ConfusionMatrix, Loss
    #     from ignite.metrics.roc_auc import ROC_AUC
    #     from metrics_utils import cls_output_transform, auc_output_transform
    #     num_classes = getattr(self, "num_classes", 2)
    #     return {
    #         "val_acc": Accuracy(output_transform=cls_output_transform),
    #         "val_auc": ROC_AUC(output_transform=auc_output_transform),
    #         "val_loss": Loss(loss_fn=self.get_loss_fn(), output_transform=cls_output_transform),
    #         "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform),
    #     }

    def get_loss_fn(self) -> Callable:
        import torch.nn as nn
        return nn.CrossEntropyLoss()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": False,
            "num_classes": getattr(self, "num_classes", 1),
            "seg_output_transform": None,
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# DenseNet121
class DenseNet121Model(BaseModel):
    from monai.networks.nets import DenseNet121

    def build_model(self, config: Any) -> Any:
        return self.DenseNet121(
            spatial_dims=2,
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            pretrained=getattr(config, "pretrained", False),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    # def get_metrics(self) -> Dict[str, Any]:
    #     from ignite.metrics import Accuracy, ConfusionMatrix, Loss
    #     from ignite.metrics.roc_auc import ROC_AUC
    #     from metrics_utils import cls_output_transform, auc_output_transform
    #     num_classes = getattr(self, "num_classes", 2)
    #     return {
    #         "val_acc": Accuracy(output_transform=cls_output_transform),
    #         "val_auc": ROC_AUC(output_transform=auc_output_transform),
    #         "val_loss": Loss(loss_fn=self.get_loss_fn(), output_transform=cls_output_transform),
    #         "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform),
    #     }

    def get_loss_fn(self) -> Callable:
        import torch.nn as nn
        return nn.CrossEntropyLoss()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Classification: no segmentation metrics
        return {
            "add_segmentation_metrics": False,
            "num_classes": getattr(self, "num_classes", 1),
            "seg_output_transform": None,
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# UNet
class UNetModel(BaseModel):
    from monai.networks.nets import UNet

    def build_model(self, config: Any) -> Any:
        return self.UNet(
            spatial_dims=2,
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            channels=getattr(config, "features", (32, 64, 128, 256, 512)),
            strides=getattr(config, "strides", (2, 2, 2, 2)),
            num_res_units=getattr(config, "num_res_units", 2),
        )

    def get_output_transform(self):
        # Use segmentation output transform
        return self.get_seg_output_transform()

    def get_supported_tasks(self) -> List[str]:
        return ["segmentation"]

    # def get_metrics(self) -> Dict[str, Any]:
    #     from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
    #     from metrics_utils import seg_confmat_output_transform
    #     num_classes = getattr(self, "num_classes", getattr(self, "out_channels", 1))
    #     cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=seg_confmat_output_transform)
    #     return {
    #         "dice": DiceCoefficient(cm=cm_metric),
    #         "iou": JaccardIndex(cm=cm_metric),
    #         # "confmat": cm_metric,  # (optional)
    #     }

    def get_loss_fn(self) -> Callable:
        from monai.losses import DiceLoss
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": True,
            "num_classes": getattr(self, "num_classes", 1),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# Multitask UNet
class MultitaskUNetModel(BaseModel):
    from models.multitask_unet import MultiTaskUNet

    def build_model(self, config: Any) -> Any:
        return self.MultiTaskUNet(
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            num_classes=getattr(config, "num_classes", 2),
            features=getattr(config, "features", (32, 64, 128, 256, 512)),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification", "segmentation", "multitask"]

    # def get_metrics(self) -> Dict[str, Any]:
    #     from ignite.metrics import Accuracy, ConfusionMatrix, DiceCoefficient, JaccardIndex, Loss
    #     from ignite.metrics.roc_auc import ROC_AUC
    #     from metrics_utils import cls_output_transform, auc_output_transform, seg_confmat_output_transform
    #     num_classes = getattr(self, "num_classes", 2)
    #     # Segmentation confusion matrix for Dice/IoU
    #     seg_cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=seg_confmat_output_transform)
    #     return {
    #         "val_acc": Accuracy(output_transform=cls_output_transform),
    #         "val_auc": ROC_AUC(output_transform=auc_output_transform),
    #         "val_loss": Loss(loss_fn=self.get_loss_fn(), output_transform=cls_output_transform),
    #         "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform),
    #         "val_dice": DiceCoefficient(cm=seg_cm_metric),
    #         "val_iou": JaccardIndex(cm=seg_cm_metric),
    #     }

    def get_loss_fn(self) -> Callable:
        from metrics_utils import get_classification_metrics, get_segmentation_metrics
        import torch

        def multitask_loss(y_pred, y_true):
            """Handles both dict (training) and tensor (metric) cases."""
            # If y_true is a dict (from training step)
            if isinstance(y_true, dict):
                class_logits, seg_out = None, None
                if isinstance(y_pred, tuple):
                    if len(y_pred) == 2:
                        class_logits, seg_out = y_pred
                    elif len(y_pred) == 1:
                        class_logits, seg_out = y_pred[0], None
                elif isinstance(y_pred, dict):
                    class_logits = y_pred.get("label", None)
                    seg_out = y_pred.get("mask", None)
                else:
                    class_logits, seg_out = y_pred, None

                loss = 0.0
                if "label" in y_true and class_logits is not None:
                    loss += get_classification_metrics()["loss"](class_logits, y_true["label"])
                if "mask" in y_true and seg_out is not None:
                    loss += get_segmentation_metrics()["loss"](seg_out, y_true["mask"])
                if loss == 0.0:
                    raise ValueError(f"No valid targets found in y_true: keys={list(y_true.keys())}")
                return loss
            # If y_true is a tensor (from Ignite metric)
            elif torch.is_tensor(y_true):
                # Assume classification-only, match y_pred shape
                return get_classification_metrics()["loss"](y_pred, y_true)
            else:
                raise TypeError(f"Unsupported y_true type for multitask_loss: {type(y_true)}")
        return multitask_loss

    # def get_loss_fn(self) -> Callable:
    #     from metrics_utils import multitask_loss
    #     return multitask_loss

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": True,
            "num_classes": getattr(self, "num_classes", 2),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# ViT
class ViTModel(BaseModel):
    from monai.networks.nets import ViT

    def build_model(self, config: Any) -> Any:
        img_size = tuple(getattr(config, "input_shape", (256, 256)))
        patch_size = getattr(config, "patch_size", 16)
        in_channels = getattr(config, "in_channels", 1)
        num_classes = getattr(config, "num_classes", 2)
        print(">>> ViT DEBUG:", {"img_size": img_size, "patch_size": patch_size, "in_channels": in_channels})

        return self.ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            spatial_dims=getattr(config, "spatial_dims", 2),
            classification=True,
            num_classes=num_classes,
            hidden_size=getattr(config, "hidden_size", 768),
            mlp_dim=getattr(config, "mlp_dim", 3072),
            num_layers=getattr(config, "num_layers", 12),
            num_heads=getattr(config, "num_heads", 12),
            dropout_rate=getattr(config, "dropout_rate", 0.1),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    @staticmethod
    def extract_logits(y_pred):
        """Extracts logits from ViT output (handles tuple or plain tensor)."""
        if isinstance(y_pred, tuple):
            return y_pred[0]
        return y_pred

    # def get_metrics(self) -> Dict[str, Any]:
    #     from ignite.metrics import Accuracy, ConfusionMatrix, Loss
    #     from ignite.metrics.roc_auc import ROC_AUC
    #     from metrics_utils import cls_output_transform, auc_output_transform
    #     num_classes = getattr(self, "num_classes", 2)
    #     return {
    #         "val_acc": Accuracy(output_transform=cls_output_transform),
    #         "val_auc": ROC_AUC(output_transform=auc_output_transform),
    #         "val_loss": Loss(loss_fn=self.get_loss_fn(), output_transform=cls_output_transform),
    #         "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform),
    #     }

    def get_loss_fn(self) -> Callable:
        import torch.nn as nn

        def loss_fn(y_pred, y_true):
            y_pred = self.extract_logits(y_pred)
            # For classification, y_true might be a dict: {"label": ...}
            if isinstance(y_true, dict):
                # Try all possible keys in order of likelihood
                for key in ["label", "classification", "class"]:
                    if key in y_true:
                        y_true = y_true[key]
                        break
                # If not found, raise an error
                else:
                    raise ValueError(f"Expected key 'label' or 'classification' in y_true dict, got keys: {list(y_true.keys())}")
            return nn.CrossEntropyLoss()(y_pred, y_true)
        return loss_fn

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": False,
            "num_classes": getattr(self, "num_classes", 2),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }

    def get_seg_output_transform(self):
        # No segmentation output for ViT, but keep signature consistent
        def _noop_transform(output):
            return output
        return _noop_transform


# SwinUNETR
class SwinUNETRModel(BaseModel):
    from monai.networks.nets import SwinUNETR

    def build_model(self, config: Any) -> Any:
        return self.SwinUNETR(
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            img_size=getattr(config, "img_size", (256, 256)),
            feature_size=getattr(config, "feature_size", 48),
            use_checkpoint=getattr(config, "use_checkpoint", False),
            norm_layer=getattr(config, "norm_layer", "instance"),
            depths=getattr(config, "depths", (2, 2, 6, 2)),
            num_heads=getattr(config, "num_heads", (3, 6, 12, 24)),
            window_size=getattr(config, "window_size", 7),
            mlp_ratio=getattr(config, "mlp_ratio", 4.0),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["segmentation"]

    def get_output_transform(self):
        # Use segmentation output transform
        return self.get_seg_output_transform()

    # def get_metrics(self) -> Dict[str, Any]:
    #     from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
    #     from metrics_utils import seg_confmat_output_transform
    #     num_classes = getattr(self, "out_channels", getattr(self, "num_classes", 1))
    #     cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=seg_confmat_output_transform)
    #     return {
    #         "val_dice": DiceCoefficient(cm=cm_metric),
    #         "val_iou": JaccardIndex(cm=cm_metric),
    #         # "confmat": cm_metric,  # (optional)
    #     }

    def get_loss_fn(self) -> Callable:
        from monai.losses import DiceLoss
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Segmentation: segmentation metrics should be enabled
        return {
            "add_segmentation_metrics": True,
            "num_classes": getattr(self, "out_channels", 1),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# Model registry
MODEL_REGISTRY: Dict[str, ModelRegistryProtocol] = {
    "simple_cnn": SimpleCNNModel(),
    "densenet121": DenseNet121Model(),
    "unet": UNetModel(),
    "multitask_unet": MultitaskUNetModel(),
    "vit": ViTModel(),
    "swin_unetr": SwinUNETRModel(),
}


# Optimizer construction
def get_optimizer(
    name: str,
    parameters: Any,
    lr: float,
    weight_decay: float = 1e-4
) -> Optimizer:
    name = name.lower()
    if name == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif name == "rmsprop":
        return RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


__all__ = [
    "SimpleCNNModel",
    "DenseNet121Model",
    "UNetModel",
    "MultitaskUNetModel",
    "ViTModel",
    "SwinUNETRModel",
]
