# model_utils.py
from typing import Any, Callable, Dict, List  # Tuple, Type
from torch.optim import Adam, SGD, RMSprop, Optimizer
from monai.networks.nets import DenseNet121, UNet, ViT, SwinUNETR
from models.simple_cnn import SimpleCNN
from models.multitask_unet import MultiTaskUNet
from model_protocol import ModelRegistryProtocol
from metrics_utils import (
    cls_output_transform,
    auc_output_transform,
    seg_output_transform_for_confmat,
)


class BaseModel(ModelRegistryProtocol):
    def get_output_transform(self):
        # Default: use classification output transform
        return self.get_cls_output_transform()

    def get_cls_output_transform(self):
        return cls_output_transform

    def get_auc_output_transform(self):
        return auc_output_transform

    def get_seg_output_transform(self):
        return seg_output_transform_for_confmat


# SimpleCNN
class SimpleCNNModel(BaseModel):
    def build_model(self, config: Any) -> Any:
        in_channels = getattr(config, "in_channels", 1)
        num_classes = getattr(config, "num_classes", 2)
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    def get_metrics(self) -> Dict[str, Any]:
        from ignite.metrics import Accuracy, ConfusionMatrix, Loss
        from ignite.metrics.roc_auc import ROC_AUC
        num_classes = getattr(self, "num_class_labels", 2)
        return {
            "val_acc": Accuracy(output_transform=self.get_cls_output_transform()),
            "val_auc": ROC_AUC(output_transform=self.get_auc_output_transform()),
            "val_loss": Loss(loss_fn=self.get_loss_fn(), output_transform=self.get_cls_output_transform()),
            "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=self.get_cls_output_transform()),
        }

    def get_loss_fn(self) -> Callable:
        import torch.nn as nn
        return nn.CrossEntropyLoss()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": False,
            "num_classes": getattr(self, "num_class_labels", 1),
            "seg_output_transform": None,
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# DenseNet121
class DenseNet121Model(BaseModel):
    def build_model(self, config: Any) -> Any:
        return DenseNet121(
            spatial_dims=2,
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            pretrained=getattr(config, "pretrained", False),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    def get_metrics(self) -> Dict[str, Any]:
        from ignite.metrics import Accuracy, ConfusionMatrix, Loss
        from ignite.metrics.roc_auc import ROC_AUC
        num_classes = getattr(self, "num_class_labels", 2)
        return {
            "val_acc": Accuracy(output_transform=self.get_cls_output_transform()),
            "val_auc": ROC_AUC(output_transform=self.get_auc_output_transform()),
            "val_loss": Loss(loss_fn=self.get_loss_fn(), output_transform=self.get_cls_output_transform()),
            "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=self.get_cls_output_transform()),
        }

    def get_output_transform(self):
        # Use segmentation output transform
        return self.get_seg_output_transform()

    def get_loss_fn(self) -> Callable:
        import torch.nn as nn
        return nn.CrossEntropyLoss()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Classification: no segmentation metrics
        return {
            "add_segmentation_metrics": False,
            "num_classes": getattr(self, "num_class_labels", 1),
            "seg_output_transform": None,
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# UNet
class UNetModel(BaseModel):
    def build_model(self, config: Any) -> Any:
        return UNet(
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

    def get_metrics(self) -> Dict[str, Any]:
        from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
        num_classes = getattr(self, "out_channels", 1)
        cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=self.get_seg_output_transform())
        return {
            "dice": DiceCoefficient(cm=cm_metric),
            "iou": JaccardIndex(cm=cm_metric)
        }

    def get_loss_fn(self) -> Callable:
        from monai.losses import DiceLoss
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": True,
            "num_classes": getattr(self, "num_class_labels", 1),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# Multitask UNet
class MultitaskUNetModel(BaseModel):
    def build_model(self, config: Any) -> Any:
        return MultiTaskUNet(
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            num_class_labels=getattr(config, "num_class_labels", 2),
            features=getattr(config, "features", (32, 64, 128, 256, 512)),
        )

    def get_supported_tasks(self) -> List[str]:
        return ["classification", "segmentation", "multitask"]

    def get_metrics(self) -> Dict[str, Any]:
        from ignite.metrics import Accuracy, ConfusionMatrix, DiceCoefficient, JaccardIndex, Loss
        from ignite.metrics.roc_auc import ROC_AUC
        # from eval_utils import get_classification_metrics
        num_classes = getattr(self, "num_class_labels", 2)
        cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=self.get_seg_output_transform())
        return {
            "val_acc": Accuracy(output_transform=self.get_cls_output_transform()),
            "val_auc": ROC_AUC(output_transform=self.get_auc_output_transform()),
            "val_loss": Loss(loss_fn=self.get_loss_fn(), output_transform=self.get_cls_output_transform()),
            "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=self.get_cls_output_transform()),
            "dice": DiceCoefficient(cm=cm_metric),
            "iou": JaccardIndex(cm=cm_metric),
        }

    def get_loss_fn(self) -> Callable:
        from metrics_utils import multitask_loss
        return multitask_loss

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": True,
            "num_classes": getattr(self, "num_class_labels", 2),
            "seg_output_transform": self.get_seg_output_transform(),
            "dice_name": "val_dice",
            "iou_name": "val_iou",
        }


# ViT
class ViTModel(BaseModel):
    def build_model(self, config: Any) -> Any:
        img_size = tuple(getattr(config, "input_shape", (256, 256)))
        patch_size = getattr(config, "patch_size", 16)
        in_channels = getattr(config, "in_channels", 1)
        num_classes = getattr(config, "num_class_labels", 2)
        print(">>> ViT DEBUG:", {"img_size": img_size, "patch_size": patch_size, "in_channels": in_channels})

        return ViT(
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

    def get_cls_output_transform(self):
        def _output_transform(output):
            # Accept tuple or list: treat both as (y_pred, y)
            if isinstance(output, (tuple, list)):
                if len(output) == 2:
                    y_pred, y = output
                elif len(output) > 2:
                    y_pred, y = output[0], output[1]  # Take only the first two
                else:
                    raise ValueError(f"Output tuple/list too short: {output}")
            elif isinstance(output, dict):
                y_pred = output.get("y_pred") or output.get("pred") or output.get("logits")
                y = output.get("y") or output.get("label") or output.get("classification")
                if y_pred is None or y is None:
                    raise ValueError(f"Cannot extract y_pred and y from dict output: keys={output.keys()}")
            else:
                raise ValueError(f"Unexpected output type in metric output_transform: {type(output)}")
            # If y is still a dict, extract the classification label tensor
            if isinstance(y, dict):
                # Use the most likely key(s) for your label tensor
                y = y.get("label") or y.get("classification") or y.get("class")
                if y is None:
                    raise ValueError(f"Could not extract tensor from y dict: {output}")
            y_pred = self.extract_logits(y_pred)
            return y_pred, y
        return _output_transform

    def get_auc_output_transform(self):
        # For AUC, identical to classification
        return self.get_cls_output_transform()

    def get_metrics(self) -> Dict[str, Any]:
        from ignite.metrics import Accuracy, ConfusionMatrix, Loss
        from ignite.metrics.roc_auc import ROC_AUC
        num_classes = getattr(self, "num_class_labels", 2)
        return {
            "val_acc": Accuracy(output_transform=self.get_cls_output_transform()),
            "val_auc": ROC_AUC(output_transform=self.get_auc_output_transform()),
            "val_loss": Loss(loss_fn=self.get_loss_fn(), output_transform=self.get_cls_output_transform()),
            "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=self.get_cls_output_transform()),
        }

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
            # print(">>> LOSS DEBUG y_pred shape:", getattr(y_pred, "shape", None))
            # print(">>> LOSS DEBUG y_true shape:", getattr(y_true, "shape", None))
            return nn.CrossEntropyLoss()(y_pred, y_true)
        return loss_fn

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_segmentation_metrics": False,
            "num_classes": getattr(self, "num_class_labels", 2),
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
    def build_model(self, config: Any) -> Any:
        return SwinUNETR(
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

    def get_metrics(self) -> Dict[str, Any]:
        from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
        num_classes = getattr(self, "out_channels", 1)
        cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=self.get_seg_output_transform())
        return {
            "dice": DiceCoefficient(cm=cm_metric),
            "iou": JaccardIndex(cm=cm_metric)
        }

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
