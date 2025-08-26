# model_registry.py
from typing import Dict
from protocols import ModelRegistryProtocol

from models.simple_cnn import SimpleCNNModel
from models.densenet121 import DenseNet121Model
from models.unet import UNetModel
from models.multitask_unet import MultitaskUNetModel
from models.vit import ViTModel
from models.swin_unetr import SwinUNETRModel


MODEL_REGISTRY: Dict[str, ModelRegistryProtocol] = {
    "simple_cnn": SimpleCNNModel,
    "densenet121": DenseNet121Model,
    "unet": UNetModel,
    "multitask_unet": MultitaskUNetModel,
    "vit": ViTModel,
    "swin_unetr": SwinUNETRModel,
}


__all__ = ["MODEL_REGISTRY"]
