from model_protocol import ModelContextProtocol
from typing import Dict
from model_utils import (
    SimpleCNNModel,
    DenseNet121Model,
    UNetModel,
    MultitaskUNetModel,
    ViTModel,
    SwinUNETRModel,
)

MODEL_REGISTRY: Dict[str, ModelContextProtocol] = {
    "simple_cnn": SimpleCNNModel(),
    "densenet121": DenseNet121Model(),
    "unet": UNetModel(),
    "multitask_unet": MultitaskUNetModel(),
    "vit": ViTModel(),
    "swin_unetr": SwinUNETRModel(),
}
