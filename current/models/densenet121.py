# models/densenet121.py
import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from models.model_base import BaseModel

logger = logging.getLogger(__name__)


class DenseNet121Model(nn.Module, BaseModel):
    """
    Refactored to match SimpleCNNModel:
      - __init__(cfg, **kwargs) sets up modules
      - forward(...) returns {"logits": ..., "cls_logits": ...}
      - param_groups() exposes backbone/head for optimizer splits
      - config snapshot carries num_classes, head_logits, class_weights, etc.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()

        def _get(key: str, default: Any = None):
            if kwargs and key in kwargs:
                return kwargs[key]
            if cfg is not None and key in cfg:
                return cfg.get(key)
            return default

        # Core hyperparameters
        in_channels = int(_get("in_channels", 1))
        num_classes = int(_get("num_classes", 2))
        binary_single_logit = bool(_get("binary_single_logit", num_classes == 2))
        out_dim = 1 if binary_single_logit and num_classes == 2 else num_classes
        pretrained = bool(_get("pretrained", False))

        # Optional CE class weights derived from class_counts
        config_snapshot: Dict[str, Any] = dict(cfg) if cfg is not None else {}
        for k, v in (kwargs or {}).items():
            config_snapshot.setdefault(k, v)

        counts = _get("class_counts", None)
        if counts and isinstance(counts, (list, tuple)) and len(counts) == num_classes:
            counts_t = torch.as_tensor(counts, dtype=torch.float32)
            weights = counts_t.sum() / (len(counts_t) * counts_t.clamp_min(1))
            weights = weights / weights.mean()
            config_snapshot["class_weights"] = [float(w) for w in weights.tolist()]
            logger.info(
                "[DenseNet121] class_counts=%s -> CE class_weights=%s",
                counts, [round(float(w), 4) for w in weights.tolist()],
            )

        # Build MONAI DenseNet121 (no activation on head; raw logits only)
        self.net = DenseNet121(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_dim,
            pretrained=pretrained,
        )

        # Config snapshot for downstream helpers (PosProbCfg, etc.)
        config_snapshot.setdefault("num_classes", int(num_classes))
        config_snapshot.setdefault("positive_index", int(_get("positive_index", 1)))
        config_snapshot.setdefault("head_logits", int(out_dim))  # 1 => BCEWithLogits; >=2 => CrossEntropy
        self.config = config_snapshot

    def forward(self, batch: Dict[str, torch.Tensor] | torch.Tensor) -> Dict[str, torch.Tensor]:
        x = batch["image"] if isinstance(batch, dict) else batch
        logits = self.net(x)  # shape [B, out_dim]
        return {"logits": logits, "cls_logits": logits}

    def param_groups(self) -> Dict[str, list]:
        """
        Split params so optimizers/schedulers can use different LR/WD on backbone vs head.
        MONAI DenseNet121 exposes classifier parameters under 'class_layers.*'.
        Falls back to 'classifier'/'fc' if needed.
        """
        head_prefixes: Tuple[str, ...] = ("class_layers", "classifier", "fc")
        head_params, backbone_params = [], []
        for name, p in self.net.named_parameters():
            if any(name.startswith(pref) for pref in head_prefixes):
                head_params.append(p)
            else:
                backbone_params.append(p)
        if not head_params:  # robust fallback
            backbone_params = list(self.net.parameters())
        return {"backbone": backbone_params, "head": head_params}

    # (Optional) keep for symmetry with other wrappers
    def get_supported_tasks(self) -> list[str]:
        return ["classification"]


# # densenet121.py
# import logging
# from typing import Any, List, Dict

# import torch
# import torch.nn as nn
# from models.model_base import BaseModel
# from metrics_utils import cls_output_transform
# from monai.networks.nets import DenseNet121

# logger = logging.getLogger(__name__)


# class DenseNet121Model(nn.Module, BaseModel):
#     """
#     DenseNet121 classifier wrapper compatible with the refactored BaseModel API.

#     - Stores config on instance for BaseModel helpers.
#     - Uses BaseModel.get_loss_fn(task, cfg) (do not override).
#     - Exposes cls_output_transform for metrics.
#     """

#     # Registry / construction
#     def build_model(self, config: Any) -> Any:
#         self.config = config
#         in_channels: int = int(self._cfg_get(config, "in_channels", 1))
#         self.num_classes: int = int(self._cfg_get(config, "num_classes", 2))
#         pretrained: bool = bool(self._cfg_get(config, "pretrained", False))

#         # Optional: derive CE class_weights from class_counts and store on cfg
#         counts = self._cfg_get(config, "class_counts", None)
#         if counts and isinstance(counts, (list, tuple)) and len(counts) == self.num_classes:
#             counts_t = torch.tensor(counts, dtype=torch.float32)
#             # inverse frequency, normalized to mean=1
#             weights = counts_t.sum() / (len(counts_t) * counts_t.clamp_min(1))
#             weights = weights / weights.mean()
#             if isinstance(config, dict):
#                 config["class_weights"] = weights.tolist()
#             else:
#                 setattr(config, "class_weights", weights.tolist())
#             logger.info(
#                 "[DenseNet121] class_counts=%s -> CE class_weights=%s",
#                 counts, [round(float(w), 4) for w in weights.tolist()]
#             )

#         model = DenseNet121(
#             spatial_dims=2,
#             in_channels=in_channels,
#             out_channels=self.num_classes,
#             pretrained=pretrained,
#         )
#         return model

#     def get_supported_tasks(self) -> List[str]:
#         # Classification only
#         return ["classification"]

#     def get_handler_kwargs(self) -> Dict[str, Any]:
#         # Keep a minimal, uniform signature with other wrappers
#         return {
#             "num_classes": self.get_num_classes(),
#             "cls_output_transform": cls_output_transform,
#             "seg_output_transform": None,  # not used here
#         }

#     # Optional convenience for external callers (tensor passthrough or common keys)
#     def extract_logits(self, y_pred: Any):
#         if isinstance(y_pred, dict):
#             for k in ("cls_out", "class_logits", "logits", "y_pred"):
#                 v = y_pred.get(k, None)
#                 if v is not None:
#                     return v
#         return y_pred
