# import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
from multitask_unet import MultiTaskUNet
from monai.networks.nets import DenseNet121, UNet  # DenseNet169

# Define registry for model constructors
MODEL_REGISTRY = {
    "multitask_unet": lambda config: MultiTaskUNet(
        spatial_dims=2,
        in_channels=getattr(config, "in_channels", 1),
        out_channels=getattr(config, "out_channels", 1),
        num_class_labels=getattr(config, "num_class_labels", 2),
        features=getattr(config, "features", (32, 64, 128, 256, 512))
    ),
    # For classification models
    "densenet121": lambda config: DenseNet121(
        spatial_dims=2,
        in_channels=getattr(config, "in_channels", 1),
        out_channels=getattr(config, "out_channels", 1),
        pretrained=getattr(config, "pretrained", False)
    ),
    "unet": lambda config: UNet(
        spatial_dims=2,
        in_channels=getattr(config, "in_channels", 1),
        out_channels=getattr(config, "out_channels", 1),
        channels=getattr(config, "features", (32, 64, 128, 256, 512)),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ),
}


def build_model(config):
    model_name = getattr(config, "model_name", "multitask_unet").lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered! Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config)


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
