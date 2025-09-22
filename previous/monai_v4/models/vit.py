# vit.py
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import torch
import torch.nn as nn
from monai.networks.nets import ViT
from models.model_base import BaseModel
from metrics_utils import cls_output_transform

logger = logging.getLogger(__name__)


class ViTModel(BaseModel):
    """
    ViT classifier wrapper compatible with the refactored BaseModel API.

    - Stores config on instance for BaseModel helpers.
    - Uses BaseModel.get_loss_fn(task, cfg) (do not override).
    - Supports class_counts -> class_weights (for CE) and bias init from class priors.
    - Exposes cls_output_transform for evaluator metrics.
    """

    # helpers: shapes & patching
    @staticmethod
    def _normalize_img_size(
        input_shape: Sequence[int] | Tuple[int, ...],
        in_channels: int,
        spatial_dims: int,
    ) -> Tuple[int, ...]:
        shape = tuple(input_shape)
        if spatial_dims == 2:
            # (C, H, W) or (H, W)
            if len(shape) == 3 and shape[0] == in_channels:
                return shape[1:3]
            if len(shape) == 2:
                return shape
        elif spatial_dims == 3:
            # (C, D, H, W) or (D, H, W)
            if len(shape) == 4 and shape[0] == in_channels:
                return shape[1:4]
            if len(shape) == 3:
                return shape
        raise ValueError(
            f"input_shape {shape} is not compatible with spatial_dims={spatial_dims} and in_channels={in_channels}"
        )

    @staticmethod
    def _validate_patching(img_size: Sequence[int], patch_size: int | Sequence[int]) -> Tuple[int, ...]:
        ps = (patch_size,) * len(img_size) if isinstance(patch_size, int) else tuple(patch_size)
        if len(ps) != len(img_size):
            raise ValueError(f"patch_size {ps} must match dims of img_size {img_size}")
        if any(s % p != 0 for s, p in zip(img_size, ps)):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {ps}")
        return ps

    # helpers: priors & bias init
    @staticmethod
    def _priors_from_counts(counts: Sequence[int]) -> torch.Tensor:
        t = torch.as_tensor(counts, dtype=torch.float32)
        p = (t / t.sum().clamp_min(1.0)).clamp_min(1e-8)
        return p

    def _find_classifier_linear(self, root: nn.Module) -> Optional[nn.Linear]:
        """
        Scan modules and pick the last Linear whose out_features matches 1 or self.num_classes.
        Fallback to the last Linear if no perfect match is found.
        """
        modules = [m for m in root.modules() if isinstance(m, nn.Linear)]
        if not modules:
            return None
        for m in reversed(modules):
            if m.out_features in (1, getattr(self, "num_classes", None)):
                return m
        return modules[-1]

    def _init_classifier_bias_from_priors(self, model: nn.Module) -> None:
        priors = getattr(self, "class_priors", None)
        if priors is None:
            return
        lin = self._find_classifier_linear(model)
        if lin is None or lin.bias is None:
            logger.warning("No classifier Linear with bias; skipping prior bias init.")
            return

        with torch.no_grad():
            if lin.out_features == 1:
                # BCEWithLogits single-logit case
                p_pos = float(priors[-1].item())
                p_pos = min(max(p_pos, 1e-8), 1.0 - 1e-8)
                value = math.log(p_pos) - math.log(1.0 - p_pos)
                lin.bias.fill_(value)
                logger.info("Initialized binary classifier bias from priors (p_pos=%.6f).", p_pos)
            else:
                # Cross-entropy multi-logit case: bias_k = log p_k
                lin.bias.copy_(torch.log(priors.to(lin.bias.device)))
                logger.info("Initialized multi-class classifier bias from priors: %s", priors.tolist())

    # helpers: make dims sane
    def _sanitize_vit_dims(self, config: Any) -> Any:
        hs = int(self._cfg_get(config, "hidden_size", 384))
        nh = int(self._cfg_get(config, "num_heads", 8))

        if hs <= 0 or nh <= 0:
            raise ValueError(f"hidden_size and num_heads must be > 0, got {hs}, {nh}")

        if hs % nh != 0:
            new_hs = max(nh, (hs // nh) * nh)
            logger.warning(
                "hidden_size %d not divisible by num_heads %d; snapping hidden_size -> %d",
                hs, nh, new_hs
            )
            hs = new_hs

        self._cfg_set(config, "hidden_size", hs)
        self._cfg_set(config, "num_heads", nh)
        return config

    # BaseModel API implementations
    def build_model(self, config: Any) -> Any:
        config = self._sanitize_vit_dims(config)
        self.config = config

        spatial_dims: int = int(self._cfg_get(config, "spatial_dims", 2))
        in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        self.num_classes: int = int(self._cfg_get(config, "num_classes", 2))
        patch_size = self._cfg_get(config, "patch_size", 16)
        input_shape = self._cfg_get(config, "input_shape", (256, 256))

        img_size = self._normalize_img_size(input_shape, in_channels, spatial_dims)
        patch_size = self._validate_patching(img_size, patch_size)

        # derive CE class_weights from class_counts and store on cfg
        counts = self._cfg_get(config, "class_counts", None)
        self.class_counts = list(map(int, counts)) if counts is not None else None
        if self.class_counts and len(self.class_counts) == self.num_classes:
            counts_t = torch.tensor(self.class_counts, dtype=torch.float32)
            # inverse frequency, normalized to mean=1
            weights = counts_t.sum() / (len(counts_t) * counts_t.clamp_min(1))
            weights = weights / weights.mean()
            # Persist to config for BaseModel.get_loss_fn(...)
            if isinstance(config, dict):
                config["class_weights"] = weights.tolist()
            else:
                setattr(config, "class_weights", weights.tolist())
            logger.info(
                "[ViT] class_counts=%s -> CE class_weights=%s",
                self.class_counts, [round(float(w), 4) for w in weights.tolist()]
            )

        # Optional: set class priors (for bias init)
        self.class_priors = self._priors_from_counts(self.class_counts) if self.class_counts else None

        if self._cfg_get(config, "debug", False):
            logger.info(
                "ViT DEBUG: img_size=%s patch_size=%s in_channels=%d num_classes=%d spatial_dims=%d",
                img_size, patch_size, in_channels, self.num_classes, spatial_dims
            )
            logger.info("config.class_counts = %s", self.class_counts)

        model = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            spatial_dims=spatial_dims,
            classification=True,
            num_classes=self.num_classes,
            hidden_size=int(self._cfg_get(config, "hidden_size", 384)),
            mlp_dim=int(self._cfg_get(config, "mlp_dim", 3072)),
            num_layers=int(self._cfg_get(config, "num_layers", 12)),
            num_heads=int(self._cfg_get(config, "num_heads", 8)),
            dropout_rate=float(self._cfg_get(config, "dropout_rate", 0.1)),
        )

        # Initialize classifier bias from priors (multi-class or binary)
        self._init_classifier_bias_from_priors(model)

        return model

    def get_supported_tasks(self) -> List[str]:
        # Classification only
        return ["classification"]

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Keep this minimal and uniform with other wrappers
        return {
            "num_classes": self.get_num_classes(),
            "cls_output_transform": cls_output_transform,
            "seg_output_transform": None,
        }

    # Optional convenience for external callers
    def extract_logits(self, y_pred: Any):
        if isinstance(y_pred, dict):
            for k in ("class_logits", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        return y_pred
        # return y_pred[0] if isinstance(y_pred, tuple) else y_pred
