# vit.py
import math
import torch
import torch.nn as nn
from typing import Any, List, Dict, Callable, Sequence, Optional
from models.model_base import BaseModel


class ViTModel(BaseModel):
    from monai.networks.nets import ViT

    # helpers for size checks
    def _normalize_img_size(self, input_shape, in_channels, spatial_dims):
        shape = tuple(input_shape)
        if spatial_dims == 2:
            if len(shape) == 3 and shape[0] == in_channels:
                return shape[1:3]
            if len(shape) == 2:
                return shape
        elif spatial_dims == 3:
            if len(shape) == 4 and shape[0] == in_channels:
                return shape[1:4]
            if len(shape) == 3:
                return shape
        raise ValueError(f"input_shape {shape} is not compatible with spatial_dims={spatial_dims} and in_channels={in_channels}")

    def _validate_patching(self, img_size, patch_size):
        ps = (patch_size,) * len(img_size) if isinstance(patch_size, int) else tuple(patch_size)
        if len(ps) != len(img_size):
            raise ValueError(f"patch_size {ps} must match img_size dims {img_size}")
        if any(s % p != 0 for s, p in zip(img_size, ps)):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {ps}")

    # new: priors + bias init
    @staticmethod
    def _priors_from_counts(counts: Sequence[int]) -> torch.Tensor:
        t = torch.as_tensor(counts, dtype=torch.float32)
        p = (t / t.sum().clamp_min(1.0)).clamp_min(1e-8)
        return p

    def _find_classifier_linear(self, root: nn.Module) -> Optional[nn.Linear]:
        """
        Be robust to different heads: scan all modules and pick the last Linear
        whose out_features matches 1 or self.num_classes. Fallback to the last Linear.
        """
        modules = [m for m in root.modules() if isinstance(m, nn.Linear)]
        if not modules:
            return None
        for m in reversed(modules):
            if m.out_features in (1, getattr(self, "num_classes", None)):
                return m
        return modules[-1]

    def _init_classifier_bias_from_priors(self, model: nn.Module) -> None:
        if getattr(self, "class_priors", None) is None:
            return
        lin = self._find_classifier_linear(model)
        if lin is None or lin.bias is None:
            print("[WARN] No classifier Linear with bias; skipping prior bias init.")
            return

        with torch.no_grad():
            if lin.out_features == 1:
                # BCEWithLogits single-logit case (if used)
                p_pos = float(self.class_priors[-1].item())
                p_pos = min(max(p_pos, 1e-8), 1.0 - 1e-8)
                value = math.log(p_pos) - math.log(1.0 - p_pos)
                lin.bias.fill_(value)
            else:
                # Cross-entropy (K logits): bias_k = log p_k
                lin.bias.copy_(torch.log(self.class_priors.to(lin.bias.device)))
        print(f"[INFO] Initialized classifier bias from priors: {self.class_priors.tolist()}")

    def _sanitize_vit_dims(self, config):
        # from math import gcd
        hs = int(getattr(config, "hidden_size", 384))
        nh = int(getattr(config, "num_heads", 8))

        if hs <= 0 or nh <= 0:
            raise ValueError(f"hidden_size and num_heads must be > 0, got {hs}, {nh}")

        if hs % nh != 0:
            # Option A: auto-fix by snapping hidden_size down to nearest multiple
            new_hs = max(nh, (hs // nh) * nh)
            print(f"[WARN] hidden_size {hs} not divisible by num_heads {nh}; "
                  f"snapping hidden_size -> {new_hs}")
            hs = new_hs
            # Option B (stricter): instead raise with a helpful hint
            # raise ValueError(f"Choose values so hidden_size % num_heads == 0, e.g. "
            #                  f"(384,6|8|12), (512,8), (768,12). Got ({hs},{nh}).")

        # Optional: nudge toward typical per-head dims (~32/48/64)
        head_dim = hs // nh
        if head_dim not in (32, 48, 64):
            # print(f"[INFO] head_dim={head_dim}. Typical are 32/48/64; not required, just a hint.")
            print(f"[INFO] head_dim={head_dim}. Typical are 32/48/64; not required.")

        # write back so the constructor sees the sanitized values
        self._cfg_set(config, "hidden_size", hs)
        self._cfg_set(config, "num_heads", nh)
        return config

    def build_model(self, config: Any) -> Any:
        config = self._sanitize_vit_dims(config)
        self.config = config
        spatial_dims = int(self._cfg_get(config, "spatial_dims", 2))
        in_channels = int(self._cfg_get(config, "in_channels", 1))
        self.num_classes = int(self._cfg_get(config, "num_classes", 2))
        patch_size = self._cfg_get(config, "patch_size", 16)
        input_shape = self._cfg_get(config, "input_shape", (256, 256))
        img_size = self._normalize_img_size(input_shape, in_channels, spatial_dims)
        self._validate_patching(img_size, patch_size)

        # pull class_counts into the model and compute priors
        counts = self._cfg_get(config, "class_counts", None)
        self.class_counts = list(map(int, counts)) if counts is not None else None
        self.class_priors = self._priors_from_counts(self.class_counts) if self.class_counts else None
        try:
            setattr(self.config, "class_counts", self.class_counts)
        except Exception:
            pass

        if self._cfg_get(config, "debug", False):
            print(">>> ViT DEBUG:", {
                "img_size": img_size, "patch_size": patch_size,
                "in_channels": in_channels, "num_classes": self.num_classes,
                "spatial_dims": spatial_dims,
            })
            print("[debug] config.class_counts =", self.class_counts)

        # Build the ViT
        vit = self.ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            spatial_dims=spatial_dims,
            classification=True,
            num_classes=self.num_classes,
            hidden_size=int(self._cfg_get(config, "hidden_size", 768)),
            mlp_dim=int(self._cfg_get(config, "mlp_dim", 3072)),
            num_layers=int(self._cfg_get(config, "num_layers", 12)),
            num_heads=int(self._cfg_get(config, "num_heads", 12)),
            dropout_rate=float(self._cfg_get(config, "dropout_rate", 0.1)),
        )

        # NEW: initialize classifier bias from priors
        # self._set_priors_from_counts(self._cfg_get(config, "class_counts", None))
        self._init_classifier_bias_from_priors(vit)

        self.model = vit
        return vit

    def get_supported_tasks(self) -> List[str]:
        return ["classification"]

    def extract_logits(self, y_pred):
        return y_pred[0] if isinstance(y_pred, tuple) else y_pred

    def get_loss_fn(self) -> Callable:
        return self._make_class_weighted_ce()

    def get_handler_kwargs(self) -> Dict[str, Any]:
        return {
            "add_classification_metrics": True,
            "num_classes": self.num_classes,
            "acc_name": "val_acc",
            "auc_name": "val_auc",
            "confmat_name": "val_cls_confmat",
            "add_segmentation_metrics": False,
        }
