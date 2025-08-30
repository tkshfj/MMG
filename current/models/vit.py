# vit.py
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import ViT

from models.model_base import BaseModel
from metrics_utils import cls_output_transform

logger = logging.getLogger(__name__)


class ViTModel(BaseModel):
    """
    MONAI ViT wrapper for classification.

    Design:
    - build_model(cfg) returns the underlying nn.Module (MONAI ViT) so caller can `.to()` it.
    - Loss/metrics are taken from BaseModel (handle tensor or dict predictions).
    - Binary paths:
        * If cfg.binary_single_logit == True and cfg.binary_bce_from_two_logits == False,
          we build a single-logit head (num_classes=1).
        * Otherwise, we keep num_classes=C (typically 2) and let the loss collapse [B,2]->[B]
          when cfg.binary_bce_from_two_logits == True.
    - Bias init for class imbalance comes from BaseModel.init_bias_from_priors().
    """

    # helpers: shapes & patching
    @staticmethod
    def _normalize_img_size(
        input_shape: Sequence[int] | Tuple[int, ...],
        in_channels: int,
        spatial_dims: int,
    ) -> Tuple[int, ...]:
        shape = tuple(int(s) for s in input_shape)
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
        ps = (patch_size,) * len(img_size) if isinstance(patch_size, int) else tuple(int(p) for p in patch_size)
        if len(ps) != len(img_size):
            raise ValueError(f"patch_size {ps} must have the same dims as img_size {img_size}")
        if any(s % p != 0 for s, p in zip(img_size, ps)):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {ps}")
        return ps

    @staticmethod
    def _priors_from_counts(counts: Sequence[int]) -> torch.Tensor:
        t = torch.as_tensor(counts, dtype=torch.float32)
        total = t.sum().clamp_min(1.0)
        p = (t / total).clamp(1e-8, 1.0 - 1e-8)
        return p / p.sum()  # exact normalize after clamp

    def _find_classifier_linear(self, root: nn.Module) -> Optional[nn.Linear]:
        # Pick the last Linear that plausibly belongs to the classifier head
        candidates = [m for m in root.modules() if isinstance(m, nn.Linear)]
        if not candidates:
            return None
        # Prefer those matching 1 or configured num_classes
        for m in reversed(candidates):
            if getattr(self, "num_classes", None) in (None, m.out_features) or m.out_features == 1:
                return m
        return candidates[-1]

    def _sanitize_vit_dims(self, config: Any) -> Any:
        hs = int(self._cfg_get(config, "hidden_size", 384))
        nh = int(self._cfg_get(config, "num_heads", 8))

        # If pretrained is requested, hidden_size must already be divisible by num_heads
        wants_pretrained = bool(
            self._cfg_get(config, "pretrained", False) or self._cfg_get(config, "pretrained_path", None) or self._cfg_get(
                config, "weights", None) or self._cfg_get(config, "checkpoint", None) or self._cfg_get(config, "state_dict", None)
        )

        if hs <= 0 or nh <= 0:
            raise ValueError(f"hidden_size and num_heads must be > 0, got {hs}, {nh}")
        if hs % nh != 0:
            if wants_pretrained:
                raise ValueError(
                    f"hidden_size={hs} is not divisible by num_heads={nh}; fix dims to match pretrained weights."
                )
            snapped = max(nh, (hs // nh) * nh)
            if snapped != hs:
                logger.info("ViT hidden_size adjusted %d -> %d to be divisible by num_heads=%d.", hs, snapped, nh)
            hs = snapped
            self._cfg_set(config, "hidden_size", hs)
        return config

    # public API
    def build_model(self, config: Any) -> nn.Module:
        config = self._sanitize_vit_dims(config)
        self.config = config  # keep for BaseModel helpers

        spatial_dims: int = int(self._cfg_get(config, "spatial_dims", 2))
        in_channels: int = int(self._cfg_get(config, "in_channels", 1))
        requested_classes: int = int(self._cfg_get(config, "num_classes", 2))
        patch_size = self._cfg_get(config, "patch_size", 16)
        input_shape = self._cfg_get(config, "input_shape", (256, 256))

        # Binary controls
        binary_single: bool = bool(self._cfg_get(config, "binary_single_logit", requested_classes == 2))
        two_to_one: bool = bool(self._cfg_get(config, "binary_bce_from_two_logits", True))
        self.two_to_one = two_to_one  # consumed by loss/output transforms if needed

        # Decide the actual classifier head size
        if binary_single and not two_to_one and requested_classes == 2:
            out_classes = 1
        else:
            out_classes = requested_classes
        self.num_classes = out_classes  # used by get_handler_kwargs()

        img_size = self._normalize_img_size(input_shape, in_channels, spatial_dims)
        patch_size = self._validate_patching(img_size, patch_size)

        if bool(self._cfg_get(config, "debug", False)):
            logger.info(
                "ViT config: img_size=%s patch_size=%s in_ch=%d out_classes=%d spatial_dims=%d",
                img_size, patch_size, in_channels, self.num_classes, spatial_dims
            )

        # Instantiate MONAI ViT
        net = ViT(
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

        # Optional bias init from class priors
        class_counts = self._cfg_get(config, "class_counts", None)
        if class_counts:
            head = self._find_classifier_linear(net)
            if head is None:
                logger.warning("Could not find classifier Linear for bias init.")
            else:
                try:
                    self.init_bias_from_priors(
                        head=head,
                        class_counts=class_counts,
                        binary_single_logit=(self.num_classes == 1),
                        positive_index=int(self._cfg_get(config, "positive_index", 1)),
                    )
                    logger.info("Initialized classifier bias from priors.")
                except Exception as e:
                    logger.warning("Bias init from priors failed: %s", e)

        # Keep a handle if needed by callers
        self.net = net
        return net  # IMPORTANT: return nn.Module so caller can `.to()` it

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    def get_handler_kwargs(self) -> Dict[str, Any]:
        # Evaluator/metrics wiring. cls_output_transform handles tensor or dict.
        return {
            "num_classes": self.get_num_classes(),
            "cls_output_transform": cls_output_transform,
            "seg_output_transform": None,
        }

    # Convenience: extract logits from model outputs in case a caller needs it here
    def extract_logits(self, y_pred: Any):
        if isinstance(y_pred, torch.Tensor):
            return y_pred
        if isinstance(y_pred, dict):
            for k in ("cls_logits", "cls_out", "logits", "y_pred"):
                v = y_pred.get(k, None)
                if v is not None:
                    return v
        if isinstance(y_pred, (list, tuple)) and len(y_pred):
            return self.extract_logits(y_pred[0])
        return y_pred
