# vit.py
import logging
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT
from models.model_base import BaseModel

logger = logging.getLogger(__name__)


# small local helpers
def _normalize_img_size(input_shape: Sequence[int], in_channels: int, spatial_dims: int) -> Tuple[int, ...]:
    shape = tuple(int(s) for s in input_shape)
    if spatial_dims == 2:
        if len(shape) == 3 and shape[0] == in_channels:
            return (shape[1], shape[2])
        if len(shape) == 2:
            return shape
    elif spatial_dims == 3:
        if len(shape) == 4 and shape[0] == in_channels:
            return (shape[1], shape[2], shape[3])
        if len(shape) == 3:
            return shape
    raise ValueError(
        f"input_shape {shape} incompatible with spatial_dims={spatial_dims} and in_channels={in_channels}"
    )


def _coerce_logits_any(raw: Any, out_dim: Optional[int] = None) -> torch.Tensor:
    """
    Accept outputs in multiple forms and return a logits tensor:
      - Tensor
      - Dict with common keys
      - Tuple/list of items (pick best tensor)
    """
    def _from_dict(d: dict) -> Optional[torch.Tensor]:
        for k in ("logits", "cls_logits", "class_logits", "y_pred", "out", "pred", "classification"):
            v = d.get(k)
            if torch.is_tensor(v):
                return v
        for v in d.values():
            if torch.is_tensor(v):
                return v
        return None

    if torch.is_tensor(raw):
        return raw
    if isinstance(raw, dict):
        t = _from_dict(raw)
        if t is not None:
            return t
        raise RuntimeError(f"[ViTModel] Could not find logits tensor in dict keys={list(raw.keys())}")
    if isinstance(raw, (list, tuple)):
        candidates = []
        for item in raw:
            if torch.is_tensor(item):
                candidates.append(item)
            elif isinstance(item, dict):
                t = _from_dict(item)  # noqa
                if t is not None:
                    candidates.append(t)
        if not candidates:
            raise RuntimeError(f"[ViTModel] Could not find logits tensor in sequence; types={[type(it) for it in raw]}")
        if out_dim is not None:
            for t in candidates:
                if t.ndim >= 1 and t.shape[-1] == int(out_dim):
                    return t
        return candidates[0]
    raise RuntimeError(f"[ViTModel] Unexpected forward output type: {type(raw)}")


# model
class ViTModel(nn.Module, BaseModel):
    """
    MONAI ViT for classification.
    Guarantees:
      - Returns RAW logits (no Sigmoid/Softmax). Use BCEWithLogits/CE.
      - Head is discoverable via common names ('classification_head','classifier',...).
      - Provides backbone/head parameter helpers for optimizer split.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        super().__init__()
        cfg = dict(cfg) if cfg is not None else {}
        # Stash resolved config for forward-time options (e.g., auto_resize_to_img)
        self.config: dict[str, Any] = {**cfg, **kwargs}

        # core cfg/kwargs
        spatial_dims: int = int(kwargs.get("spatial_dims", cfg.get("spatial_dims", 2)))
        in_ch: int = int(kwargs.get("in_channels", cfg.get("in_channels", 1)))
        req_classes: int = int(kwargs.get("num_classes", cfg.get("num_classes", 2)))
        head_logits: int = int(kwargs.get("head_logits", cfg.get("head_logits", 1)))  # 1→BCEWithLogits, 2+→CE

        self.in_channels = in_ch
        self.spatial_dims = spatial_dims
        self.image_key = str(kwargs.get("image_key", cfg.get("image_key", "image")))

        patch_size = kwargs.get("patch_size", cfg.get("patch_size", 16))
        img_size = kwargs.get("img_size", kwargs.get("image_size", cfg.get("img_size", cfg.get("image_size", None))))
        input_shape = kwargs.get("input_shape", cfg.get("input_shape", (256, 256)))

        # out_dim from head_logits
        out_dim: int = 1 if (req_classes == 2 and head_logits == 1) else int(req_classes)
        self.head_logits = head_logits
        self.num_classes = req_classes
        self.out_dim = out_dim

        # image/patch geometry
        if img_size is None:
            img_size = _normalize_img_size(input_shape, in_ch, spatial_dims)
        else:
            img_size = (int(img_size),) * spatial_dims if isinstance(img_size, int) else tuple(int(s) for s in img_size)
        if len(img_size) != spatial_dims:
            raise ValueError(f"img_size {img_size} rank != spatial_dims={spatial_dims}")

        ps = (patch_size,) * len(img_size) if isinstance(patch_size, int) else tuple(int(p) for p in patch_size)
        if len(ps) != len(img_size) or any(s % p != 0 for s, p in zip(img_size, ps)):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {ps}")

        # transformer dims
        hidden_size = int(kwargs.get("hidden_size", cfg.get("hidden_size", 384)))
        mlp_dim = int(kwargs.get("mlp_dim", cfg.get("mlp_dim", 4 * hidden_size)))
        num_layers = int(kwargs.get("num_layers", cfg.get("num_layers", 12)))
        num_heads = int(kwargs.get("num_heads", cfg.get("num_heads", 8)))
        dropout = float(kwargs.get("dropout_rate", cfg.get("dropout_rate", 0.1)))

        if hidden_size % num_heads != 0:
            def _snap(x: int, m: int) -> int:
                down = (x // m) * m
                up = down + m
                return up if (x - down) > (up - x) else down
            snapped = _snap(hidden_size, num_heads)
            logger.info("ViT hidden_size adjusted %d → %d to be divisible by num_heads=%d.",
                        hidden_size, snapped, num_heads)
            hidden_size = max(num_heads, snapped)

        # MONAI ViT backbone (classification=True → Linear head; no activation)
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

        # make head recognizable (alias) and guaranteed Linear → raw logits
        head = getattr(self.net, "classification_head", None)
        if isinstance(head, nn.Linear):
            # Expose a stable name at both nested and top levels
            self.classifier = head
            setattr(self.net, "classifier", self.classifier)
        else:
            # Fallback: install a new Linear head
            self.classifier = nn.Linear(hidden_size, out_dim, bias=True)
            setattr(self.net, "classification_head", self.classifier)
            setattr(self.net, "classifier", self.classifier)

        # optional bias init from prior for single-logit binary
        cc = cfg.get("class_counts")
        if bool(cfg.get("init_head_bias_from_prior", False)) and cc is not None and out_dim == 1:
            try:
                neg, pos = float(cc[0]), float(cc[1])
                tot = max(1.0, neg + pos)
                p_pos = min(max(pos / tot, 1e-6), 1.0 - 1e-6)
                self._init_binary_bias_from_prior(p_pos)
            except Exception as e:
                logger.warning("Bias init skipped (class_counts=%s): %s", cc, e)

        logger.info(
            "ViT resolved: spatial=%d, in_ch=%d, img=%s, patch=%s, hidden=%d, mlp_dim=%d, heads=%d, out_dim=%d",
            self.spatial_dims, self.in_channels, self.img_size, self.patch_size, hidden_size, mlp_dim, num_heads, self.out_dim
        )

    # head discovery / mutation (class-level methods)
    def _head_module(self) -> Optional[nn.Module]:
        for attr in ("classifier", "classification_head", "mlp_head", "fc", "cls", "head"):
            if hasattr(self.net, attr):
                mod = getattr(self.net, attr)
                if isinstance(mod, nn.Linear):
                    return mod
                if isinstance(mod, (nn.Sequential, nn.ModuleList)) and len(list(mod.children())) > 0:
                    last = list(mod.children())[-1]
                    if isinstance(last, nn.Linear):
                        return mod
        return None

    def _set_head_module(self, new: nn.Module) -> None:
        # Prefer canonical aliasing on both names
        setattr(self.net, "classification_head", new)
        setattr(self.net, "classifier", new)
        self.classifier = new

    # param grouping helpers
    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        head = self._head_module()
        head_ids = {id(p) for p in head.parameters()} if head is not None else set()
        for p in self.parameters():
            if p.requires_grad and id(p) not in head_ids:
                yield p

    def head_parameters(self) -> Iterator[nn.Parameter]:
        head = self._head_module()
        if head is None:
            return iter(())
        return (p for p in head.parameters() if p.requires_grad)

    def param_groups(self):
        return {"backbone": list(self.backbone_parameters()), "head": list(self.head_parameters())}

    # bias seed
    def _init_binary_bias_from_prior(self, p: float):
        import math
        p = float(min(max(p, 1e-6), 1 - 1e-6))
        b = math.log(p / (1 - p))
        head = self._head_module()
        if head is None:
            return
        with torch.no_grad():
            if isinstance(head, nn.Linear) and head.out_features == 1 and head.bias is not None:
                head.bias.fill_(b)
                logger.info("Initialized binary head bias to %.4f.", b)
            elif isinstance(head, nn.Sequential):
                last = list(head.children())[-1]
                if isinstance(last, nn.Linear) and last.out_features == 1 and last.bias is not None:
                    last.bias.fill_(b)
                    logger.info("Initialized binary head bias to %.4f.", b)

    # forward (returns RAW logits)
    def forward(self, batch: Mapping[str, torch.Tensor] | torch.Tensor) -> Mapping[str, torch.Tensor]:
        x = batch[self.image_key] if isinstance(batch, Mapping) else batch
        if not torch.is_tensor(x):
            raise TypeError(f"[ViTModel] Expected Tensor for '{self.image_key}', got {type(x)}")

        expected_ndim = 2 + self.spatial_dims  # N + C + spatial
        if x.ndim != expected_ndim:
            raise RuntimeError(
                f"[ViTModel] Expected tensor with {expected_ndim} dims "
                f"(N,C,{'D,H,W' if self.spatial_dims==3 else 'H,W'}), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise RuntimeError(f"[ViTModel] Expected C={self.in_channels}, got C={x.shape[1]}")

        spatial = tuple(int(s) for s in x.shape[2:])
        if spatial != self.img_size:
            if bool(self.config.get("auto_resize_to_img", False)):
                mode = "trilinear" if self.spatial_dims == 3 else "bilinear"
                x = F.interpolate(x, size=self.img_size, mode=mode, align_corners=False)
            else:
                raise RuntimeError(
                    f"[ViTModel] Expected spatial size={self.img_size}, got {spatial}. "
                    f"Set config['auto_resize_to_img']=True to enable on-the-fly resize."
                )
        x = x.contiguous()

        # MONAI ViT with classification=True returns logits (no activation)
        raw = self.net(x)
        logits = raw if torch.is_tensor(raw) else _coerce_logits_any(raw, out_dim=self.out_dim)

        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if logits.ndim != 2:
            raise RuntimeError(f"[ViTModel] Expected logits [B,C] or [B], got {tuple(logits.shape)}")

        return {"cls_logits": logits, "logits": logits}
