import torch
from torch.optim import Adam
from multitask_unet import MultiTaskUNet

def build_model(config):
    return MultiTaskUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_class_labels=2,
        features=(32, 64, 128, 256, 512)
    )

def get_optimizer(name, parameters, lr, weight_decay=1e-4):
    if name.lower() == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
