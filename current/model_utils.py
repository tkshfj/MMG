# model_utils.py
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type
from torch.optim import Adam, SGD, RMSprop
from models.multitask_unet import MultiTaskUNet
from models.simple_cnn import SimpleCNN
from monai.networks.nets import DenseNet121, UNet, ViT, SwinUNETR


# Model context classes
@dataclass
class SimpleCNNContext:
    in_channels: int = 1
    num_classes: int = 2

    @classmethod
    def from_config(cls, config: Any) -> "SimpleCNNContext":
        return cls(
            in_channels=getattr(config, "in_channels", 1),
            num_classes=getattr(config, "num_classes", 2),
        )


@dataclass
class DenseNet121Context:
    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    pretrained: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "DenseNet121Context":
        return cls(
            spatial_dims=2,
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            pretrained=getattr(config, "pretrained", False),
        )


@dataclass
class UNetContext:
    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    features: Tuple[int, ...] = (32, 64, 128, 256, 512)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2

    @classmethod
    def from_config(cls, config: Any) -> "UNetContext":
        return cls(
            spatial_dims=2,
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            features=getattr(config, "features", (32, 64, 128, 256, 512)),
            strides=getattr(config, "strides", (2, 2, 2, 2)),
            num_res_units=getattr(config, "num_res_units", 2),
        )


@dataclass
class MultiTaskUNetContext:
    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    num_class_labels: int = 2
    features: Tuple[int, ...] = (32, 64, 128, 256, 512)

    @classmethod
    def from_config(cls, config: Any) -> "MultiTaskUNetContext":
        return cls(
            spatial_dims=2,
            in_channels=getattr(config, "in_channels", 1),
            out_channels=getattr(config, "out_channels", 1),
            num_class_labels=getattr(config, "num_class_labels", 2),
            features=getattr(config, "features", (32, 64, 128, 256, 512)),
        )


@dataclass
class ViTContext:
    in_channels: int = 1
    img_size: int = 224
    patch_size: int = 16
    pos_embed: str = "conv"
    classification: bool = True
    num_classes: int = 2
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1

    @classmethod
    def from_config(cls, config: Any) -> "ViTContext":
        return cls(
            in_channels=getattr(config, "in_channels", 1),
            img_size=getattr(config, "img_size", 224),
            patch_size=getattr(config, "patch_size", 16),
            pos_embed="conv",
            classification=True,
            num_classes=getattr(config, "num_class_labels", 2),
            hidden_size=getattr(config, "hidden_size", 768),
            mlp_dim=getattr(config, "mlp_dim", 3072),
            num_layers=getattr(config, "num_layers", 12),
            num_heads=getattr(config, "num_heads", 12),
            dropout_rate=getattr(config, "dropout_rate", 0.1),
        )


@dataclass
class SwinUNETRContext:
    in_channels: int = 1
    out_channels: int = 1
    img_size: Tuple[int, int] = (256, 256)
    feature_size: int = 48
    use_checkpoint: bool = False
    norm_layer: str = "instance"
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0

    @classmethod
    def from_config(cls, config: Any) -> "SwinUNETRContext":
        return cls(
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


# Model registry
MODEL_REGISTRY: Dict[str, Tuple[Type[Any], Type[Any]]] = {
    "simple_cnn": (SimpleCNNContext, SimpleCNN),
    "densenet121": (DenseNet121Context, DenseNet121),
    "unet": (UNetContext, UNet),
    "multitask_unet": (MultiTaskUNetContext, MultiTaskUNet),
    "vit": (ViTContext, ViT),
    "swin_unetr": (SwinUNETRContext, SwinUNETR),
}


# Model construction
def build_model(config: Any):
    architecture = getattr(config, "architecture", "multitask_unet").lower()
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Model '{architecture}' is not registered! Available: {list(MODEL_REGISTRY.keys())}")
    context_cls, model_cls = MODEL_REGISTRY[architecture]
    context = context_cls.from_config(config)
    return model_cls(**vars(context))


# Optimizer construction
def get_optimizer(name: str, parameters, lr, weight_decay=1e-4):
    name = name.lower()
    if name == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif name == "rmsprop":
        return RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
