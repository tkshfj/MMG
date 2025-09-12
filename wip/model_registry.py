# model_registry.py
from __future__ import annotations
from typing import Callable, Dict, Optional, Iterable, Tuple
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


# Helpers: dims & bias
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


def _maybe_init_cls_bias(module: nn.Module, cfg: dict) -> None:
    """
    Initialize last linear bias from class prior for classification heads.
    Supports 1-logit and 2-logit cases.
    """
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

    # Bias only on the *classification head* linear, not arbitrary backbone layers.
    loc = _locate_head_module(module)
    if not loc:
        return
    _, _, head = loc

    # Limit bias init to the classification head only
    loc = _locate_head_module(module)
    if not loc:
        return
    _, _, head = loc
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


# Build machinery
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
    # Best-effort: some classes accept image/img_size via **kwargs even if not explicit.
    kwargs["image_size"] = img
    try:
        return cls(cfg, **kwargs)  # if __init__(cfg, **kw) exists
    except TypeError:
        try:
            return cls(**kwargs)
        except TypeError:
            return cls()  # last resort


# Head surgery utilities
_HEAD_ATTR_CANDIDATES = (
    "classification_head", "classifier", "mlp_head", "fc", "cls", "head"
)


def _strip_activation(m: nn.Module) -> nn.Module:
    """Remove trailing Sigmoid/Softmax layers if present in a sequential-like head."""
    if isinstance(m, (nn.Sigmoid, nn.Softmax)):
        return nn.Identity()
    if isinstance(m, nn.Sequential) and len(m) > 0:
        # if last layer is activation, drop it
        last = m[-1]
        if isinstance(last, (nn.Sigmoid, nn.Softmax)):
            return nn.Sequential(*list(m.children())[:-1])
    return m


def _locate_head_module(model: nn.Module) -> Optional[Tuple[nn.Module, str, nn.Module]]:
    """Find (parent, attr_name, head_module) on model or model.net."""
    scopes = [model]
    net = getattr(model, "net", None)
    if isinstance(net, nn.Module):
        scopes.append(net)
    for scope in scopes:
        for attr in _HEAD_ATTR_CANDIDATES:
            if hasattr(scope, attr):
                head = getattr(scope, attr)
                if isinstance(head, nn.Module):
                    return scope, attr, head
    return None


def _replace_head_last_linear(parent: nn.Module, attr: str, out_dim: int) -> bool:
    """Replace final Linear within the head (only), preserving backbone."""
    head = getattr(parent, attr)
    head = _strip_activation(head)
    if isinstance(head, nn.Linear):
        setattr(parent, attr, nn.Linear(head.in_features, out_dim, bias=(head.bias is not None)))
        return True
    last_ref = None  # (container, name, linear)
    stack = [(head, None, None)]
    while stack:
        mod, p, name = stack.pop()
        if isinstance(mod, nn.Linear):
            last_ref = (p, name, mod)
        for nm, ch in mod.named_children():
            stack.append((ch, mod, nm))
    if last_ref is None:
        setattr(parent, attr, head)  # write back stripped head if needed
        return False
    p, name, lin = last_ref
    new_lin = nn.Linear(lin.in_features, out_dim, bias=(lin.bias is not None))
    if isinstance(p, nn.Sequential) and isinstance(name, str) and name.isdigit():
        p[int(name)] = new_lin
    elif hasattr(p, name):
        setattr(p, name, new_lin)
    else:
        setattr(parent, attr, nn.Linear(lin.in_features, out_dim, bias=(lin.bias is not None)))
    setattr(parent, attr, head if getattr(parent, attr) is head else getattr(parent, attr))
    return True


def _configure_classification_head(model: nn.Module, out_dim: int) -> None:
    """Ensure head emits logits with desired out_dim; do not touch backbone."""
    loc = _locate_head_module(model)
    if not loc:
        return
    parent, attr, _ = loc
    _replace_head_last_linear(parent, attr, out_dim)


# Registry
BuildFn = Callable[[dict], nn.Module]


def _simple_cnn_builder(cfg: dict) -> nn.Module:
    m = _build_with_class(SimpleCNNModel, cfg)
    return m


def _densenet121_builder(cfg: dict) -> nn.Module:
    m = _build_with_class(DenseNet121Model, cfg)
    return m


def _vit_builder(cfg: dict) -> nn.Module:
    m = _build_with_class(ViTModel, cfg)
    return m


def _unet_builder(cfg: dict) -> nn.Module:
    return _build_with_class(UNetModel, cfg)


def _swin_unetr_builder(cfg: dict) -> nn.Module:
    return _build_with_class(SwinUNETRModel, cfg)


def _multitask_unet_builder(cfg: dict) -> nn.Module:
    m = _build_with_class(MultitaskUNetModel, cfg)
    return m


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
    Enforce head sizing from head_logits, strip activations, and validate loss/logit compatibility.
    """
    name = str(cfg.get("architecture", "")).lower().strip()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    # 1) Build base model
    model = MODEL_REGISTRY[name](cfg)

    # 2) Classification head sizing & activation stripping
    task = str(cfg.get("task", "classification")).lower()
    if task in ("classification", "multitask"):
        out_dim = _cls_out_dim(cfg)
        _configure_classification_head(model, out_dim)

    # 3) Initialize head bias from prior (if classification present)
    if task in ("classification", "multitask"):
        _maybe_init_cls_bias(model, cfg)

    # 4) Loss/logits compatibility checks
    head_logits = _cls_out_dim(cfg)  # already resolved to 1 or >=2 as configured
    loss_name = str(cfg.get("cls_loss", "bce")).lower()
    if head_logits == 1 and loss_name not in ("bce", "bce_logits", "bcewithlogits"):
        raise ValueError("head_logits=1 expects BCE-with-logits style loss.")
    if head_logits >= 2 and loss_name not in ("ce", "crossentropy", "cross_entropy"):
        raise ValueError("head_logits=2 expects CrossEntropy.")

    # 5) Expose param-group helpers for optimizer
    head_keys: Iterable[str] = tuple(cfg.get("head_keys", _HEAD_ATTR_CANDIDATES))

    if not hasattr(model, "head_parameters"):
        def head_parameters():
            for n, p in model.named_parameters():
                if any(k in n.lower() for k in head_keys):
                    if p.requires_grad:
                        yield p
        model.head_parameters = head_parameters

    if not hasattr(model, "backbone_parameters"):
        def backbone_parameters():
            head_ids = {id(p) for p in model.head_parameters()}
            for p in model.parameters():
                if p.requires_grad and id(p) not in head_ids:
                    yield p
        model.backbone_parameters = backbone_parameters

    if not hasattr(model, "param_groups"):
        def param_groups():
            return {
                "backbone": list(model.backbone_parameters()),
                "head": list(model.head_parameters()),
            }
        model.param_groups = param_groups

    # 6) Make the active model available to loss builders (used by make_default_loss)
    try:
        cfg["_active_model"] = model
    except Exception:
        pass

    # 7) (Optional) quick forward contract check can be re-enabled if desired
    #    left disabled here to avoid accidental CUDA init during config parsing.

    return model


__all__ = ["MODEL_REGISTRY", "build_model"]
