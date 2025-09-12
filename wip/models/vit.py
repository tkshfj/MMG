# vit.py
import logging
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT
from models.model_base import BaseModel

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


class ViTModel(nn.Module, BaseModel):
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
        # Size the final layer explicitly from head_logits:
        head_logits: int = int(kwargs.get("head_logits", cfg.get("head_logits", 1)))  # 1:BCEWithLogits, 2:CE
        # Persist essentials used by forward()
        self.in_channels: int = int(in_ch)
        self.image_key: str = str(kwargs.get("image_key", cfg.get("image_key", "image")))
        self.spatial_dims: int = int(spatial_dims)
        patch_size = kwargs.get("patch_size", cfg.get("patch_size", 16))
        img_size = kwargs.get("img_size", kwargs.get("image_size", cfg.get("img_size", cfg.get("image_size", None))))
        input_shape = kwargs.get("input_shape", cfg.get("input_shape", (256, 256)))

        # Decide out_dim from head_logits: 1 for single-logit binary; else num_classes
        out_dim: int = 1 if (req_classes == 2 and head_logits == 1) else req_classes
        self.head_logits: int = int(head_logits)
        self.num_classes: int = int(req_classes)
        self.out_dim: int = int(out_dim)

        # Image/patch geometry
        if img_size is None:
            img_size = _normalize_img_size(input_shape, in_ch, spatial_dims)
        else:
            if isinstance(img_size, int):
                img_size = (int(img_size),) * spatial_dims
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
        if hidden_size < 8:
            raise ValueError(f"hidden_size appears invalid ({hidden_size}); expected >= 8.")
        mlp_dim = int(kwargs.get("mlp_dim", cfg.get("mlp_dim", 4 * hidden_size)))
        if mlp_dim < max(8, hidden_size):  # typical ViT uses 2–4× hidden
            logger.warning("mlp_dim (%d) unusually small vs hidden_size (%d).", mlp_dim, hidden_size)
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
        self.img_size = tuple(img_size)
        self.patch_size = tuple(ps)
        self.net = ViT(
            in_channels=in_ch,
            img_size=self.img_size,
            patch_size=self.patch_size,
            spatial_dims=spatial_dims,
            classification=True,
            num_classes=out_dim,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
        )

        blk0 = self.net.blocks[0]
        assert isinstance(blk0.mlp.linear1, nn.Linear) and isinstance(blk0.mlp.linear2, nn.Linear)
        if blk0.mlp.linear1.out_features < 8:
            raise RuntimeError(f"mlp_dim too small: {blk0.mlp.linear1.out_features}")
        if blk0.mlp.linear2.in_features != blk0.mlp.linear1.out_features:
            raise RuntimeError("mlp_dim mismatch between linear1.out and linear2.in")

        # Optional head-level dropout (before the final Linear)
        head_p = float(cfg.get("head_dropout", 0.0))
        use_head_norm = bool(cfg.get("head_norm", False))  # default False

        if head_p > 0.0 or use_head_norm:
            head_attr = next((c for c in ("classification_head", "cls_head", "classifier") if hasattr(self.net, c)), None)
            head_mod = getattr(self.net, head_attr) if head_attr is not None else None
            if isinstance(head_mod, nn.Linear):
                in_features, out_features = head_mod.in_features, head_mod.out_features
                layers = []
                if use_head_norm:
                    layers.append(nn.LayerNorm(in_features))
                if head_p > 0.0:
                    layers.append(nn.Dropout(p=head_p))
                layers.append(nn.Linear(in_features, out_features, bias=True))
                new_head = nn.Sequential(*layers)
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
                # logger.info("Head dropout requested (p=%.3f) but no linear head found; skipping.", head_p)
                logger.info("Head dropout requested (p=%.3f), but no linear classification head found; skipping.", head_p)

        # Bias init for single-logit binary head (stabilizes early training)
        # Expect TRAIN class counts: class_counts = (neg, pos)
        cc = cfg.get("class_counts", None)
        if cc is not None and out_dim == 1:
            try:
                neg, pos = float(cc[0]), float(cc[1])
                denom = max(1.0, neg + pos)
                pos_prior = max(1e-6, min(1.0 - 1e-6, pos / denom))
                soften = float(cfg.get("bias_init_soften", 0.50))  # 0..1, default 0.5
                self._init_binary_bias_from_prior(pos_prior, scale=soften)
            except Exception as e:
                logger.warning("Bias init skipped (class_counts=%s): %s", cc, e)

        # Keep a snapshot for downstream utilities
        self.config = dict(cfg) if cfg is not None else {}
        self.config.update({k: v for k, v in kwargs.items() if k not in self.config})
        # Log resolved core dims once to catch misconfigs early
        logger.info(
            "ViT resolved: spatial=%d, in_ch=%d, img=%s, patch=%s, hidden=%d, mlp_dim=%d, heads=%d, out_dim=%d",
            self.spatial_dims, self.in_channels, self.img_size, self.patch_size, hidden_size, mlp_dim, num_heads, self.out_dim
        )

    def _head_module(self) -> Optional[nn.Module]:
        # Prefer explicit names; avoid generic "head" unless it's clearly a linear classifier
        for attr in ("classification_head", "cls_head", "classifier"):
            if hasattr(self.net, attr):
                return getattr(self.net, attr)
        if hasattr(self.net, "head"):
            h = getattr(self.net, "head")
            # accept only if it’s a Linear or ends with a Linear
            if isinstance(h, nn.Linear):
                return h
            if isinstance(h, (nn.Sequential, nn.ModuleList)) and len(list(h.children())) > 0 and isinstance(list(h.children())[-1], nn.Linear):
                return h
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

    def _init_binary_bias_from_prior(self, p: float, scale: float = 1.0):
        eps = 1e-4
        p = float(min(max(p, eps), 1 - eps))
        import math
        b = torch.tensor([math.log(p / (1 - p))], dtype=torch.float32) * float(max(0.0, min(1.0, scale)))
        head = self._head_module()
        if isinstance(head, nn.Linear) and head.out_features == 1 and head.bias is not None:
            with torch.no_grad():
                head.bias.copy_(b.to(device=head.bias.device, dtype=head.bias.dtype))
                logger.info("Initialized binary head bias to %.4f.", b.item())
            return
        if isinstance(head, nn.Sequential):
            last = list(head.children())[-1]
            if isinstance(last, nn.Linear) and last.out_features == 1 and last.bias is not None:
                with torch.no_grad():
                    last.bias.copy_(b.to(device=last.bias.device, dtype=last.bias.dtype))
                    logger.info("Initialized binary head bias to %.4f.", b.item())
                return

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

    def _coerce_logits(self, raw: Any) -> torch.Tensor:
        """
        Accept MONAI ViT outputs in multiple forms:
        - Tensor
        - Dict with common keys
        - Tuple/list where first Tensor (or dict containing a Tensor) is logits
        """
        def _from_dict(d: dict) -> Optional[torch.Tensor]:
            for k in ("logits", "cls_logits", "class_logits", "y_pred", "out", "pred", "classification"):
                v = d.get(k, None)
                if torch.is_tensor(v):
                    return v
            for v in d.values():
                if torch.is_tensor(v):
                    return v
            return None
        # Tensor case
        if torch.is_tensor(raw):
            return raw
        # Dict case
        if isinstance(raw, dict):
            t = _from_dict(raw)
            if t is not None:
                return t
            raise RuntimeError(f"[ViTModel] Could not find logits tensor in dict keys={list(raw.keys())}")
        # Tuple/List case (e.g., (logits,) or (aux, logits...))
        if isinstance(raw, (list, tuple)):
            # 1) prefer a Tensor with trailing dim == out_dim
            candidates = []
            for item in raw:
                if torch.is_tensor(item):
                    candidates.append(item)
                elif isinstance(item, dict):
                    t = _from_dict(item)
                    if t is not None:
                        candidates.append(t)
            for t in candidates:
                if t.ndim >= 2 and t.shape[-1] == self.out_dim:
                    return t
            # 2) otherwise, first tensor-like is fine
            if candidates:
                return candidates[0]
            raise RuntimeError(f"[ViTModel] Could not find logits tensor in tuple/list types={[type(it) for it in raw]}")
        # Fallback
        raise RuntimeError(f"[ViTModel] Unexpected forward output type: {type(raw)}")

    def forward(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        # 1) Extract image tensor explicitly (no guessing)
        if isinstance(batch, dict):
            if self.image_key not in batch:
                raise KeyError(f"[ViTModel] Missing '{self.image_key}' in batch dict. "
                               f"Available keys: {list(batch.keys())}")
            x = batch[self.image_key]
        else:
            x = batch
        if not torch.is_tensor(x):
            raise TypeError(f"[ViTModel] Expected a Tensor for '{self.image_key}', got {type(x)}")

        # 2) Enforce N(C)(D)HW rank and in_channels; keep dtype float32
        expected_ndim = 2 + int(getattr(self, "spatial_dims", 2))  # N + C + spatial_dims
        if x.ndim != expected_ndim:
            raise RuntimeError(
                f"[ViTModel] Expected tensor with {expected_ndim} dims "
                f"(N,C,{ 'D,H,W' if self.spatial_dims==3 else 'H,W' }), got shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise RuntimeError(f"[ViTModel] Expected C={self.in_channels}, got C={x.shape[1]} (shape={tuple(x.shape)})")
        # Enforce spatial size or resize if configured
        spatial = tuple(int(s) for s in x.shape[2:])
        if spatial != self.img_size:
            if bool(getattr(self, "config", {}).get("auto_resize_to_img", False)):
                mode = "trilinear" if self.spatial_dims == 3 else "bilinear"
                x = F.interpolate(x, size=self.img_size, mode=mode, align_corners=False)
            else:
                raise RuntimeError(
                    f"[ViTModel] Expected spatial size={self.img_size}, got {spatial}. "
                    f"Set config['auto_resize_to_img']=True to enable on-the-fly resize."
                )
        x = x.contiguous()
        # only force float32 if explicitly configured
        if bool(getattr(self, "config", {}).get("force_float32_input", False)) and x.dtype != torch.float32:
            x = x.float()

        # 3) Call MONAI ViT (it should output [B, head_logits] for classification=True)
        raw = self.net(x)
        logits = self._coerce_logits(raw)

        # 4) Normalize classifier output to [B, C]
        if not torch.is_tensor(logits):
            raise RuntimeError(f"[ViTModel] _coerce_logits did not return a Tensor (got {type(logits)})")
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        elif logits.ndim != 2:
            raise RuntimeError(f"[ViTModel] Expected logits with shape [B,C] or [B], got {tuple(logits.shape)}")

        # 5) Lightweight diagnostics (optional but helpful)
        with torch.no_grad():
            if torch.isfinite(logits).all():
                s = logits if logits.size(1) == 1 else (logits.max(dim=1).values - logits.min(dim=1).values)
                if s.std().item() < 1e-6:
                    # Use your logger if available; print kept for simplicity
                    logger.debug("[ViTModel] logits near-constant this batch (std < 1e-6)")

        # Provide both keys for compatibility with existing pipelines
        return {"cls_logits": logits, "logits": logits}
