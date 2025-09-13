# models/model_registry.py
from __future__ import annotations
from typing import Callable, Dict, Optional, Iterable, Tuple
import inspect
import math
import torch
import torch.nn as nn

# Import concrete model classes
from models.simple_cnn import SimpleCNNModel
from models.densenet121 import DenseNet121Model
from models.unet import UNetModel
from models.multitask_unet import MultitaskUNetModel
from models.vit import ViTModel
from models.swin_unetr import SwinUNETRModel


# Helpers: dims & compatibili
def _cls_out_dim(cfg: dict) -> int:
    """
    Preferred: honor head_logits when present.
    BC: if binary & binary_single_logit=True -> 1, else num_classes.
    """
    if "head_logits" in cfg:
        v = int(cfg["head_logits"])
        # For non-binary tasks allow arbitrary v; for binary allow {1,2}
        num_classes = int(cfg.get("num_classes", 2))
        if num_classes == 2 and v in (1, 2):
            return v
        return max(1, v)

    num_classes = int(cfg.get("num_classes", 2))
    single_logit = bool(cfg.get("binary_single_logit", True)) and num_classes == 2
    return 1 if single_logit else num_classes


def _seg_out_dim(cfg: dict) -> int:
    return int(cfg.get("seg_num_classes", cfg.get("num_classes", 2)) or 2)


def _loss_logits_compat_guard(cfg: dict, head_logits: int) -> None:
    """
    Enforce loss/logit compatibility:
      - head_logits == 1  -> BCE-with-logits
      - head_logits >= 2  -> CrossEntropy
    """
    loss_name = str(cfg.get("cls_loss", "bce")).lower()
    if head_logits == 1 and loss_name not in ("bce", "bce_logits", "bcewithlogits"):
        raise ValueError("head_logits=1 expects BCE-with-logits style loss (cfg.cls_loss).")
    if head_logits >= 2 and loss_name not in ("ce", "crossentropy", "cross_entropy"):
        raise ValueError("head_logits>=2 expects CrossEntropy loss (cfg.cls_loss).")


# Build machinery (safe kwargs)
def _accepts_any(cls: type, names: Iterable[str]) -> Dict[str, str]:
    """
    Return a mapping of accepted kwarg-name -> provided-name for the first
    found match in 'names'. E.g., prefer 'image_size' but accept 'img_size'.
    """
    sig = inspect.signature(cls)
    params = set(sig.parameters.keys())
    out = {}
    # image_size/img_size
    for want, alts in (("image_size", ("image_size", "img_size")),):
        for a in alts:
            if a in params:
                out[want] = a  # we'll map provided 'image_size' to the accepted name
                break
    return out


def _build_with_class(cls: type, cfg: dict) -> nn.Module:
    # Prefer class-level builders if present
    for method_name in ("build", "from_config"):
        meth = getattr(cls, method_name, None)
        if callable(meth):
            return meth(cfg)

    # Fallback: pass cfg + best-guess kwargs
    in_ch = int(cfg.get("in_channels", 1))
    img = tuple(cfg.get("input_shape", (256, 256)))
    task = str(cfg.get("task", "classification")).lower()

    kwargs: dict = {"in_channels": in_ch}
    if task in ("classification", "multitask"):
        kwargs["num_classes"] = _cls_out_dim(cfg)
    if task in ("segmentation", "multitask"):
        kwargs["out_channels"] = _seg_out_dim(cfg)

    # Map image size kwarg name only if accepted
    kwmap = _accepts_any(cls, ("image_size", "img_size"))
    if "image_size" in kwmap:
        kwargs[kwmap["image_size"]] = img

    # Try (cfg, **kwargs) ➜ (**kwargs) ➜ bare
    try:
        return cls(cfg, **kwargs)  # if __init__(cfg, **kw) exists
    except TypeError:
        try:
            return cls(**kwargs)
        except TypeError:
            return cls()  # last resort


# Head utilities (safe surgery)
_HEAD_ATTR_CANDIDATES = (
    "classification_head", "classifier", "mlp_head", "fc", "cls", "head"
)


def _strip_activation(m: nn.Module) -> nn.Module:
    """Remove trailing Sigmoid/Softmax layers if present in a sequential-like head."""
    if isinstance(m, (nn.Sigmoid, nn.Softmax, nn.LogSoftmax)):
        return nn.Identity()
    if isinstance(m, nn.Sequential) and len(m) > 0:
        last = m[-1]
        if isinstance(last, (nn.Sigmoid, nn.Softmax, nn.LogSoftmax)):
            return nn.Sequential(*list(m.children())[:-1])
    return m


def _has_any_linear(module: nn.Module) -> bool:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return True
    return False


def _locate_head_module(model: nn.Module) -> Optional[Tuple[nn.Module, str, nn.Module]]:
    """
    Find (parent, attr_name, head_module) on model or model.net.
    Prefer attributes that actually contain a Linear somewhere.
    """
    scopes = [model]
    net = getattr(model, "net", None)
    if isinstance(net, nn.Module):
        scopes.append(net)

    for scope in scopes:
        for attr in _HEAD_ATTR_CANDIDATES:
            if hasattr(scope, attr):
                head = getattr(scope, attr)
                if isinstance(head, nn.Module) and _has_any_linear(head):
                    return scope, attr, head
    return None


def _replace_head_last_linear(parent: nn.Module, attr: str, out_dim: int) -> bool:
    """
    Replace the final Linear within the head (only), preserving backbone.
    Returns True if replaced.
    """
    head = getattr(parent, attr)
    head = _strip_activation(head)

    # Case: head is a single Linear
    if isinstance(head, nn.Linear):
        new_head = nn.Linear(head.in_features, out_dim, bias=(head.bias is not None))
        setattr(parent, attr, new_head)
        return True

    # Case: find the last Linear inside the head container
    last_ref: Tuple[nn.Module, str, nn.Linear] | None = None  # (container, name, linear)
    stack = [(head, None, None)]
    while stack:
        mod, parent_mod, name = stack.pop()
        if isinstance(mod, nn.Linear):
            last_ref = (parent_mod, name, mod)  # type: ignore[arg-type]
        for nm, ch in mod.named_children():
            stack.append((ch, mod, nm))

    if last_ref is None:
        # write back any activation-stripped head
        setattr(parent, attr, head)
        return False

    p, name, lin = last_ref
    new_lin = nn.Linear(lin.in_features, out_dim, bias=(lin.bias is not None))
    if isinstance(p, nn.Sequential) and isinstance(name, str) and name.isdigit():
        p[int(name)] = new_lin
    elif isinstance(name, str) and hasattr(p, name):
        setattr(p, name, new_lin)
    else:
        # As a conservative fallback, replace the entire head with a Linear
        setattr(parent, attr, nn.Linear(lin.in_features, out_dim, bias=(lin.bias is not None)))
        return True

    # Ensure the (potentially activation-stripped) head is written back
    setattr(parent, attr, head)
    return True


def _configure_classification_head(model: nn.Module, out_dim: int) -> None:
    """Ensure head emits logits with desired out_dim; do not touch backbone."""
    loc = _locate_head_module(model)
    if not loc:
        return
    parent, attr, _ = loc
    _replace_head_last_linear(parent, attr, out_dim)


# Bias init (head-only, gated)
def _maybe_init_cls_bias(model: nn.Module, cfg: dict) -> None:
    """
    Initialize last Linear bias of the classification head from class prior.
    Supports 1-logit and 2-logit cases. No effect unless cfg.init_head_bias_from_prior is True
    and cfg.class_counts is provided.
    """
    if not bool(cfg.get("init_head_bias_from_prior", False)):
        return

    counts = cfg.get("class_counts")
    try:
        neg, pos = float(counts[0]), float(counts[1])
    except Exception:
        return

    tot = neg + pos
    if tot <= 0:
        return

    p_pos = max(1e-6, min(1.0 - 1e-6, pos / tot))
    b = math.log(p_pos / (1.0 - p_pos))
    pos_idx = int(cfg.get("positive_index", 1))

    loc = _locate_head_module(model)
    if not loc:
        return
    _, _, head = loc

    # Find the last Linear in the head only
    last_lin: Optional[nn.Linear] = None
    for m in head.modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            last_lin = m
    if last_lin is None:
        return

    with torch.no_grad():
        if last_lin.out_features == 1:
            last_lin.bias.fill_(b)
        elif last_lin.out_features == 2:
            vec = torch.empty(2, dtype=last_lin.bias.dtype, device=last_lin.bias.device)
            if pos_idx == 1:
                vec[:] = torch.tensor([-b, b], dtype=vec.dtype, device=vec.device)
            else:
                vec[:] = torch.tensor([b, -b], dtype=vec.dtype, device=vec.device)
            last_lin.bias.copy_(vec)


# Registry
BuildFn = Callable[[dict], nn.Module]


def _simple_cnn_builder(cfg: dict) -> nn.Module:
    return _build_with_class(SimpleCNNModel, cfg)


def _densenet121_builder(cfg: dict) -> nn.Module:
    return _build_with_class(DenseNet121Model, cfg)


def _vit_builder(cfg: dict) -> nn.Module:
    return _build_with_class(ViTModel, cfg)


def _unet_builder(cfg: dict) -> nn.Module:
    return _build_with_class(UNetModel, cfg)


def _swin_unetr_builder(cfg: dict) -> nn.Module:
    return _build_with_class(SwinUNETRModel, cfg)


def _multitask_unet_builder(cfg: dict) -> nn.Module:
    return _build_with_class(MultitaskUNetModel, cfg)


MODEL_REGISTRY: Dict[str, BuildFn] = {
    "simple_cnn": _simple_cnn_builder,
    "densenet121": _densenet121_builder,
    "vit": _vit_builder,
    "unet": _unet_builder,
    "swin_unetr": _swin_unetr_builder,
    "multitask_unet": _multitask_unet_builder,
}


# Public API
def build_model(cfg: dict) -> nn.Module:
    """
    Resolve and build a model from cfg['architecture'].
    Enforce head sizing from head_logits, strip activations,
    bias-init only on the head, and validate loss/logit compatibility.
    """
    name = str(cfg.get("architecture", "")).lower().strip()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    # Build base model
    model = MODEL_REGISTRY[name](cfg)

    # expose canonical bits on model.config
    model.config = dict(
        head_logits=int(cfg.get("head_logits", 1)),
        num_classes=int(cfg.get("num_classes", 2)),
        positive_index=int(cfg.get("positive_index", 1)),
        head_keys=tuple(cfg.get("head_keys", _HEAD_ATTR_CANDIDATES)),
    )

    # Classification head sizing & activation stripping (no backbone surgery)
    task = str(cfg.get("task", "classification")).lower()
    if task in ("classification", "multitask"):
        out_dim = _cls_out_dim(cfg)
        _configure_classification_head(model, out_dim)
        _maybe_init_cls_bias(model, cfg)             # bias init (gated) on the head only
        _loss_logits_compat_guard(cfg, out_dim)      # CE vs BCE check

    # Expose param-group helpers for optimizer
    head_keys: Iterable[str] = tuple(cfg.get("head_keys", _HEAD_ATTR_CANDIDATES))

    if not hasattr(model, "head_parameters"):
        def head_parameters():
            for n, p in model.named_parameters():
                if any(k in n.lower() for k in head_keys):
                    if p.requires_grad:
                        yield p
        model.head_parameters = head_parameters  # type: ignore[attr-defined]

    if not hasattr(model, "backbone_parameters"):
        def backbone_parameters():
            head_ids = {id(p) for p in model.head_parameters()}
            for p in model.parameters():
                if p.requires_grad and id(p) not in head_ids:
                    yield p
        model.backbone_parameters = backbone_parameters  # type: ignore[attr-defined]

    if not hasattr(model, "param_groups"):
        def param_groups(
            lr: float | None = None,
            weight_decay: float | None = None,
            *,
            head_lr_scale: float | None = None,
            head_wd_scale: float | None = None,
            no_decay_norm: bool = True,
            no_decay_bias: bool = True,
        ):
            # resolve from cfg if args not provided
            base_lr = float(lr if lr is not None else cfg.get("lr", 1e-3))
            base_wd = float(weight_decay if weight_decay is not None else cfg.get("weight_decay", 0.0))
            h_lr_s = float(head_lr_scale if head_lr_scale is not None else cfg.get("head_lr_scale", 1.0))
            h_wd_s = float(head_wd_scale if head_wd_scale is not None else cfg.get("head_wd_scale", 1.0))
            # collect head param ids once
            head_ids = {id(p) for p in model.head_parameters()}
            # norm layer types (kept local; no new helpers)
            _NORM_TYPES = (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                nn.LocalResponseNorm
            )
            # buckets
            g0_w, g1_b, g2_n = [], [], []
            g3_hw, g4_hb = [], []
            for m in model.modules():
                for name, p in m.named_parameters(recurse=False):
                    if not p.requires_grad:
                        continue
                    is_bias = name.endswith("bias")
                    is_norm = isinstance(m, _NORM_TYPES)
                    is_head = id(p) in head_ids

                    if is_head:
                        (g4_hb if is_bias else g3_hw).append(p)
                    else:
                        if is_norm and no_decay_norm:
                            g2_n.append(p)
                        elif is_bias and no_decay_bias:
                            g1_b.append(p)
                        else:
                            g0_w.append(p)
            groups = []
            if g0_w:
                groups.append({"params": g0_w, "lr": base_lr, "weight_decay": base_wd, "name": "g0_base_w"})
            if g1_b:
                groups.append({"params": g1_b, "lr": base_lr, "weight_decay": 0.0 if no_decay_bias else base_wd, "name": "g1_base_b"})
            if g2_n:
                groups.append({"params": g2_n, "lr": base_lr, "weight_decay": 0.0 if no_decay_norm else base_wd, "name": "g2_base_norm"})
            if g3_hw:
                groups.append({"params": g3_hw, "lr": base_lr * h_lr_s, "weight_decay": base_wd * h_wd_s, "name": "g3_head_w"})
            if g4_hb:
                groups.append({"params": g4_hb, "lr": base_lr * h_lr_s, "weight_decay": (0.0 if no_decay_bias else base_wd) * h_wd_s, "name": "g4_head_b"})
            return groups

        model.param_groups = param_groups  # type: ignore[attr-defined]

    # Make the active model available to loss builders (used by make_default_loss)
    try:
        cfg["_active_model"] = model
    except Exception:
        pass

    return model


__all__ = ["MODEL_REGISTRY", "build_model"]
