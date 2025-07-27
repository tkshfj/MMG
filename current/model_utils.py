# model_utils.py
# This module provides utilities for building models and optimizers in a PyTorch-based framework.
from torch.optim import Adam, SGD, RMSprop
from models.multitask_unet import MultiTaskUNet
from models.simple_cnn import SimpleCNN
from monai.networks.nets import DenseNet121, UNet, ViT, SwinUNETR

# Define registry for model constructors
MODEL_REGISTRY = {
    # For simple CNN models
    "simple_cnn": lambda config: SimpleCNN(
        in_channels=getattr(config, "in_channels", 1),
        num_classes=getattr(config, "num_classes", 2)
    ),
    # For classification models
    "densenet121": lambda config: DenseNet121(
        spatial_dims=2,
        in_channels=getattr(config, "in_channels", 1),
        out_channels=getattr(config, "out_channels", 1),
        pretrained=getattr(config, "pretrained", False)
    ),
    # For segmentation models
    "unet": lambda config: UNet(
        spatial_dims=2,
        in_channels=getattr(config, "in_channels", 1),
        out_channels=getattr(config, "out_channels", 1),
        channels=getattr(config, "features", (32, 64, 128, 256, 512)),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ),
    # For multitask models
    "multitask_unet": lambda config: MultiTaskUNet(
        spatial_dims=2,
        in_channels=getattr(config, "in_channels", 1),
        out_channels=getattr(config, "out_channels", 1),
        num_class_labels=getattr(config, "num_class_labels", 2),
        features=getattr(config, "features", (32, 64, 128, 256, 512))
    ),
    # For Vision Transformer models
    "vit": lambda config: ViT(
        in_channels=getattr(config, "in_channels", 1),
        img_size=getattr(config, "img_size", 224),
        patch_size=getattr(config, "patch_size", 16),
        pos_embed="conv",
        classification=True,
        num_classes=getattr(config, "num_class_labels", 2),
        hidden_size=768,  # Standard ViT-Base
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.1,
    ),
    # For Swin UNETR models
    "swin_unetr": lambda config: SwinUNETR(
        in_channels=getattr(config, "in_channels", 1),
        out_channels=getattr(config, "out_channels", 1),
        img_size=getattr(config, "img_size", (256, 256)),
        feature_size=48,
        use_checkpoint=getattr(config, "use_checkpoint", False),
        norm_layer="instance",
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
    ),
}


def build_model(config):
    architecture = getattr(config, "architecture", "multitask_unet").lower()
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Model '{architecture}' is not registered! Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[architecture](config)


def get_optimizer(name, parameters, lr, weight_decay=1e-4):
    name = name.lower()
    if name == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "rmsprop":
        return RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
