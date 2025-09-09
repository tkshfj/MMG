# vit.py
import logging
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from monai.networks.nets import ViT

logger = logging.getLogger(__name__)


def _normalize_img_size(
    input_shape: Sequence[int], in_channels: int, spatial_dims: int
) -> Tuple[int, ...]:
    shape = tuple(int(s) for s in input_shape)
    if spatial_dims == 2:
        # Accept (C,H,W) or (H,W)
        if len(shape) == 3 and shape[0] == in_channels:
            return (shape[1], shape[2])
        if len(shape) == 2:
            return shape
    elif spatial_dims == 3:
        # Accept (C,D,H,W) or (D,H,W)
        if len(shape) == 4 and shape[0] == in_channels:
            return (shape[1], shape[2], shape[3])
        if len(shape) == 3:
            return shape
    raise ValueError(
        f"input_shape {shape} incompatible with spatial_dims={spatial_dims} and in_channels={in_channels}"
    )


class ViTModel(nn.Module):
    """
    MONAI ViT for classification, mirroring SimpleCNNModel:
      - __init__(cfg: Optional[Mapping], **kwargs)
      - forward(batch|tensor) -> {"cls_logits": [B,C]}
      - No activation inside (use BCEWithLogits/CE)
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        super().__init__()
        cfg = dict(cfg) if cfg is not None else {}

        # Pull params (kwargs > cfg > defaults)
        spatial_dims: int = int(kwargs.get("spatial_dims", cfg.get("spatial_dims", 2)))
        in_ch: int = int(kwargs.get("in_channels", cfg.get("in_channels", 1)))
        req_classes: int = int(kwargs.get("num_classes", cfg.get("num_classes", 2)))
        patch_size = kwargs.get("patch_size", cfg.get("patch_size", 16))
        img_size = kwargs.get("img_size", kwargs.get("image_size", cfg.get("img_size", cfg.get("image_size", None))))
        input_shape = kwargs.get("input_shape", cfg.get("input_shape", (256, 256)))

        # Binary single-logit policy (like SimpleCNN)
        binary_single: bool = bool(kwargs.get("binary_single_logit", cfg.get("binary_single_logit", req_classes == 2)))
        out_dim: int = 1 if (req_classes == 2 and binary_single) else req_classes

        # Image/patch geometry
        if img_size is None:
            img_size = _normalize_img_size(input_shape, in_ch, spatial_dims)
        else:
            img_size = tuple(int(s) for s in img_size)
        if len(img_size) != spatial_dims:
            raise ValueError(f"img_size {img_size} rank != spatial_dims={spatial_dims}")

        if isinstance(patch_size, int):
            ps = (patch_size,) * len(img_size)
        else:
            ps = tuple(int(p) for p in patch_size)
        if len(ps) != len(img_size) or any(s % p != 0 for s, p in zip(img_size, ps)):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {ps}")

        # Core dims (snap hidden_size to be divisible by num_heads)
        hidden_size = int(kwargs.get("hidden_size", cfg.get("hidden_size", 384)))
        # mlp_dim = int(kwargs.get("mlp_dim", cfg.get("mlp_dim", 3072)))
        mlp_dim = int(kwargs.get("mlp_dim", cfg.get("mlp_dim", 4 * hidden_size)))
        num_layers = int(kwargs.get("num_layers", cfg.get("num_layers", 12)))
        num_heads = int(kwargs.get("num_heads", cfg.get("num_heads", 8)))
        dropout = float(kwargs.get("dropout_rate", cfg.get("dropout_rate", 0.1)))
        if hidden_size % num_heads != 0:

            def _snap_to_multiple(x: int, m: int) -> int:
                down = (x // m) * m
                up = down + m
                return up if (x - down) > (up - x) else down

            snapped = _snap_to_multiple(hidden_size, num_heads)
            if snapped != hidden_size:
                logger.info("ViT hidden_size adjusted %d -> %d to be divisible by num_heads=%d.",
                            hidden_size, snapped, num_heads)
            if abs(hidden_size - snapped) > max(8, 0.1 * hidden_size):
                raise ValueError("hidden_size too far from a valid multiple of num_heads; adjust search space.")
            hidden_size = max(num_heads, snapped)

        # Instantiate MONAI ViT (classification=True -> returns logits)
        self.net = ViT(
            in_channels=in_ch,
            img_size=img_size,
            patch_size=ps,
            spatial_dims=spatial_dims,
            classification=True,
            num_classes=out_dim,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
        )

        # Optional head-level dropout (before the final Linear)
        # Use cfg["head_dropout"] if provided; else reuse model dropout
        head_p = float(cfg.get("head_dropout", dropout))
        if head_p > 0.0:
            head_attr = next((c for c in ("classification_head", "head", "cls_head", "classifier") if hasattr(self.net, c)), None)
            # if head_attr == "classification_head" and not hasattr(self.net, "head"):
            #     self.net.head = getattr(self.net, head_attr)  # alias for downstream matchers
            head_mod = getattr(self.net, head_attr) if head_attr is not None else None
            if isinstance(head_mod, nn.Linear):
                in_features = head_mod.in_features
                out_features = head_mod.out_features
                new_head = nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Dropout(p=head_p),
                    nn.Linear(in_features, out_features, bias=True),
                )
                # copy existing weights to preserve initialization
                with torch.no_grad():
                    new_head[-1].weight.copy_(head_mod.weight)
                    if head_mod.bias is not None:
                        new_head[-1].bias.copy_(head_mod.bias)
                setattr(self.net, head_attr, new_head)
            elif isinstance(head_mod, nn.Sequential) and isinstance(list(head_mod.children())[-1], nn.Linear):
                # wrap existing stack by inserting LN/Dropout before final Linear
                last: nn.Linear = list(head_mod.children())[-1]  # type: ignore
                in_features, out_features = last.in_features, last.out_features
                pre = nn.Sequential(*list(head_mod.children())[:-1])
                new_head = nn.Sequential(
                    pre,
                    nn.LayerNorm(in_features),
                    nn.Dropout(p=head_p),
                    nn.Linear(in_features, out_features, bias=True),
                )
                with torch.no_grad():
                    new_head[-1].weight.copy_(last.weight)
                    if last.bias is not None:
                        new_head[-1].bias.copy_(last.bias)
                setattr(self.net, head_attr, new_head)
            else:
                logger.info("Head dropout requested (p=%.3f) but no linear head found; skipping.", head_p)

        # Bias init for single-logit binary head (stabilizes early training)
        # Expect TRAIN class counts: class_counts = (neg, pos)
        cc = cfg.get("class_counts", None)
        if cc is not None and out_dim == 1:
            try:
                neg, pos = float(cc[0]), float(cc[1])
                denom = max(1.0, neg + pos)
                pos_prior = max(1e-6, min(1.0 - 1e-6, pos / denom))
                self._init_binary_bias_from_prior(pos_prior)
            except Exception as e:
                logger.warning("Bias init skipped (class_counts=%s): %s", cc, e)

        # Keep a snapshot for downstream utilities
        self._cfg = dict(cfg)
        self._cfg.update({k: v for k, v in kwargs.items() if k not in self._cfg})

    @staticmethod
    def _first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, dict):
            for k in ("logits", "y_pred", "cls_logits", "output", "out", "pred"):
                v = x.get(k, None)
                if isinstance(v, torch.Tensor):
                    return v
            return None
        if isinstance(x, (list, tuple)):
            for e in x:
                t = ViTModel._first_tensor(e)
                if isinstance(t, torch.Tensor):
                    return t
        return None

    def _head_module(self) -> Optional[nn.Module]:
        for attr in ("classification_head", "head", "cls_head", "classifier"):
            if hasattr(self.net, attr):
                return getattr(self.net, attr)
        return None

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        """
        All trainable params EXCEPT the classification head.
        Adjust the prefixes if head is named differently.
        """
        head = self._head_module()
        head_param_ids = set()
        if head is not None:
            head_param_ids = {id(p) for p in head.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in head_param_ids:
                yield p

    def _init_binary_bias_from_prior(self, p: float):
        eps = 1e-4
        p = float(min(max(p, eps), 1 - eps))
        import math
        b = torch.tensor([math.log(p / (1 - p))], dtype=torch.float32)
        head = self._head_module()
        if isinstance(head, nn.Linear) and head.out_features == 1 and head.bias is not None:
            with torch.no_grad():
                # head.bias.copy_(b)
                head.bias.copy_(b.to(device=head.bias.device, dtype=head.bias.dtype))
                logger.info("Initialized binary head bias to %.4f.", b.item())
            return
        if isinstance(head, nn.Sequential):
            last = list(head.children())[-1]
            if isinstance(last, nn.Linear) and last.out_features == 1 and last.bias is not None:
                with torch.no_grad():
                    # last.bias.copy_(b)
                    last.bias.copy_(b.to(device=last.bias.device, dtype=last.bias.dtype))
                    logger.info("Initialized binary head bias to %.4f.", b.item())
                return
        # fallback (rare)
        for m in reversed(list(self.net.modules())):
            if isinstance(m, nn.Linear) and m.out_features == 1 and m.bias is not None:
                with torch.no_grad():
                    # m.bias.copy_(b)
                    m.bias.copy_(b.to(device=m.bias.device, dtype=m.bias.dtype))
                    logger.info("Initialized binary head bias to %.4f.", b.item())
                    break

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """Only the classification head params."""
        head = self._head_module()
        if head is None:
            return iter(())
        for p in head.parameters():
            if p.requires_grad:
                yield p

    def param_groups(self):
        return {
            "backbone": list(self.backbone_parameters()),
            "head": list(self.head_parameters()),
        }

    def forward(self, batch: Dict[str, torch.Tensor] | torch.Tensor) -> Dict[str, torch.Tensor]:
        # x = batch["image"] if isinstance(batch, dict) else batch
        if isinstance(batch, dict):
            x = batch.get("image") or batch.get("images")
            if x is None:
                # last resort: first tensor value that looks like an image batch
                x = next((v for v in batch.values() if isinstance(v, torch.Tensor) and v.ndim >= 4), None)
            if x is None:
                raise KeyError("[ViTModel] Could not find image tensor in batch dict.")
        else:
            x = batch
        raw = self.net(x)  # MONAI usually returns a Tensor
        logits = self._first_tensor(raw)
        if logits is None:
            raise RuntimeError(f"[ViTModel] Unexpected forward output type: {type(raw)}")

        # Contract: [B,C]. If backend gives [B], unsqueeze to [B,1] so registry check passes.
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if logits.ndim == 2 and logits.shape[1] < 1:
            raise RuntimeError(f"[ViTModel] num_classes < 1: got {tuple(logits.shape)}")
        elif logits.ndim != 2:
            raise RuntimeError(f"[ViTModel] Expected [B,C] or [B], got {tuple(logits.shape)}")

        return {"cls_logits": logits}
