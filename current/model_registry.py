# model_registry.py
from __future__ import annotations
from typing import Callable, Dict, Any, Optional
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Import concrete model classes ----
from models.simple_cnn import SimpleCNNModel
from models.densenet121 import DenseNet121Model
from models.unet import UNetModel
from models.multitask_unet import MultitaskUNetModel
from models.vit import ViTModel
from models.swin_unetr import SwinUNETRModel


# -------------------------------
# Helpers (dims, bias init, checks)
# -------------------------------
def _cls_out_dim(cfg: dict) -> int:
    num_classes = int(cfg.get("num_classes", 2))
    single_logit = bool(cfg.get("binary_single_logit", True)) and num_classes == 2
    return 1 if single_logit else num_classes


def _seg_out_dim(cfg: dict) -> int:
    return int(cfg.get("seg_num_classes", cfg.get("num_classes", 2)) or 2)


def _maybe_init_cls_bias(module: nn.Module, cfg: dict) -> None:
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

    last_lin: Optional[nn.Linear] = None
    for m in module.modules():
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


# def _maybe_init_cls_bias(module: nn.Module, cfg: dict) -> None:
#     """
#     Initialize the last Linear classifier bias from priors if class_counts=(neg,pos) is provided.
#     - For C=1: bias = logit(p_pos)
#     - For C=2: bias = [-b, +b]
#     """
#     counts = cfg.get("class_counts")
#     try:
#         neg, pos = float(counts[0]), float(counts[1])
#     except Exception:
#         return
#     tot = neg + pos
#     if tot <= 0:
#         return
#     p_pos = max(1e-6, min(1.0 - 1e-6, pos / tot))
#     b = math.log(p_pos / (1.0 - p_pos))

#     last_lin: Optional[nn.Linear] = None
#     for m in module.modules():
#         if isinstance(m, nn.Linear) and m.bias is not None:
#             last_lin = m
#     if last_lin is None:
#         return

#     with torch.no_grad():
#         if last_lin.out_features == 1:
#             last_lin.bias.fill_(b)
#         elif last_lin.out_features == 2:
#             last_lin.bias[:] = torch.tensor([-b, b], dtype=last_lin.bias.dtype, device=last_lin.bias.device)


def _as_output_dict(out: Any, *, need_cls: bool, need_seg: bool) -> Dict[str, Any]:
    if isinstance(out, dict):
        return out
    if isinstance(out, torch.Tensor):
        # Map raw tensors to the contract key we actually need
        if need_seg and not need_cls:
            return {"seg_logits": out}
        return {"cls_logits": out}
    if isinstance(out, (list, tuple)):
        # Favor first tensor; fall back to a dict that already has the right keys
        for e in out:
            if isinstance(e, torch.Tensor):
                if need_seg and not need_cls:
                    return {"seg_logits": e}
                return {"cls_logits": e}
        for e in out:
            if isinstance(e, dict) and ("cls_logits" in e or "seg_logits" in e):
                return e
    return {}


def _enforce_forward_contract(out: Any, *, need_cls: bool, need_seg: bool) -> None:
    out = _as_output_dict(out, need_cls=need_cls, need_seg=need_seg)
    if need_cls:
        t = out.get("cls_logits", None)
        if not torch.is_tensor(t) or t.ndim != 2:
            raise ValueError(f"'cls_logits' must be [B,C], got {getattr(t, 'shape', None)}")
    if need_seg:
        s = out.get("seg_logits", None)
        if not torch.is_tensor(s) or s.ndim not in (4, 5):
            raise ValueError(f"'seg_logits' must be [B,C,H,W] or [B,C,D,H,W], got {getattr(s, 'shape', None)}")


def _class_weights_from_counts(counts, num_classes: int):
    if not counts or len(counts) != num_classes:
        return None
    w = torch.tensor(counts, dtype=torch.float32)
    # inverse frequency (normalized)
    w = w.clamp_min(1e-6)
    w = w.sum() / w
    return w


def make_default_loss(cfg: dict, class_counts: list[int] | None):
    task = str(cfg.get("task", "multitask")).lower()
    C_cls = int(cfg.get("num_classes", 2))
    K_seg = int(cfg.get("seg_num_classes", cfg.get("num_classes", 2)))  # noqa: F841
    single_logit = bool(cfg.get("binary_single_logit", True)) and C_cls == 2

    alpha = float(cfg.get("alpha", 1.0))
    beta = float(cfg.get("beta", 1.0))

    def _loss_fn(outputs, targets):
        total = 0.0
        out: Dict[str, torch.Tensor] = {}
        loss_device = None

        # --- classification ---
        if task in ("classification", "multitask") and "cls_logits" in outputs:
            logits = outputs["cls_logits"]      # [B, 1] or [B, C]
            loss_device = logits.device
            y = targets.get("label", targets.get("y"))
            if y is None:
                raise ValueError("Targets must include 'label' for classification.")
            y = torch.as_tensor(y, device=logits.device)

            if single_logit:
                yf = y.float().view(-1, 1) if y.ndim == 1 else y.float()
                # pos_weight = neg/pos if available
                pw = None
                if class_counts and len(class_counts) == 2 and class_counts[1] > 0:
                    neg, pos = float(class_counts[0]), float(class_counts[1])
                    pw = torch.tensor([neg / max(pos, 1e-6)], device=logits.device, dtype=logits.dtype)
                cls_loss = F.binary_cross_entropy_with_logits(logits, yf, pos_weight=pw)
            else:
                if y.ndim >= 2 and y.shape[-1] > 1:
                    y = y.argmax(dim=-1)
                w = None
                if class_counts and len(class_counts) == C_cls:
                    w = torch.tensor(class_counts, dtype=logits.dtype, device=logits.device)
                    w = w.clamp_min(1e-6)
                    w = w.sum() / w
                cls_loss = F.cross_entropy(logits, y.long().view(-1), weight=w)

            out["cls_loss"] = cls_loss
            total = total + alpha * cls_loss

        # --- segmentation ---
        if task in ("segmentation", "multitask") and "seg_logits" in outputs:
            seg = outputs["seg_logits"]          # [B,K,H,W] or [B,K,D,H,W]
            loss_device = seg.device if loss_device is None else loss_device
            mask = targets.get("mask")
            if mask is None:
                lab = targets.get("label", {})
                if isinstance(lab, dict):
                    mask = lab.get("mask")
            if mask is None:
                raise ValueError("Targets must include 'mask' for segmentation.")
            mask = torch.as_tensor(mask, device=seg.device)
            if mask.ndim in (4, 5) and mask.shape[1] > 1:
                mask = mask.argmax(dim=1)
            seg_loss = F.cross_entropy(seg, mask.long())
            out["seg_loss"] = seg_loss
            total = total + beta * seg_loss

        if isinstance(total, torch.Tensor):
            out["loss"] = total
        else:
            dev = loss_device if loss_device is not None else "cpu"
            out["loss"] = torch.tensor(total, device=dev)
        return out

    return _loss_fn


# def make_default_loss(cfg: dict, class_counts: list[int] | None):
#     """
#     Returns a callable: (outputs, targets) -> dict(loss=..., [cls_loss=..., seg_loss=...]).
#     Works with:
#       outputs: {"cls_logits": [B,C?], "seg_logits": [B,K,H,W]?}
#       targets: {"label": [B] or one-hot, "mask": [B,H,W] or one-hot}
#     """
#     task = str(cfg.get("task", "multitask")).lower()
#     C_cls = int(cfg.get("num_classes", 2))
#     K_seg = int(cfg.get("seg_num_classes", cfg.get("num_classes", 2)))  # noqa: F841
#     single_logit = bool(cfg.get("binary_single_logit", True)) and C_cls == 2

#     # --- classification criterion ---
#     cls_crit = None
#     if task in ("classification", "multitask"):
#         if single_logit:
#             # BCE with optional pos_weight = neg/pos
#             pos_weight = None
#             if class_counts and len(class_counts) == 2 and class_counts[1] > 0:
#                 neg, pos = float(class_counts[0]), float(class_counts[1])
#                 pos_weight = torch.tensor([neg / max(pos, 1e-6)], dtype=torch.float32)
#             cls_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         else:
#             w = _class_weights_from_counts(class_counts, C_cls)
#             cls_crit = nn.CrossEntropyLoss(weight=w)

#     # --- segmentation criterion ---
#     seg_crit = None
#     if task in ("segmentation", "multitask"):
#         w = None  # put class weights here if you have them
#         seg_crit = nn.CrossEntropyLoss(weight=w)

#     # weights for multitask
#     alpha = float(cfg.get("alpha", 1.0))  # cls
#     beta = float(cfg.get("beta", 1.0))   # seg

#     def _loss_fn(outputs, targets):
#         total = 0.0
#         out = {}

#         if cls_crit is not None and "cls_logits" in outputs:
#             logits = outputs["cls_logits"]  # [B, C or 1]
#             # labels -> indices [B]
#             y = targets.get("label", targets.get("y", None))
#             if y is None:
#                 raise ValueError("Targets must include 'label' for classification.")
#             y = torch.as_tensor(y, device=logits.device)

#             if single_logit:
#                 # BCE: labels to float and view as logits
#                 yf = y.float()
#                 if yf.ndim == 1:
#                     yf = yf.view(-1, 1)
#                 cls_loss = cls_crit(logits, yf)
#             else:
#                 # CE: labels to long indices
#                 if y.ndim >= 2 and y.shape[-1] > 1:
#                     y = y.argmax(dim=-1)
#                 cls_loss = cls_crit(logits, y.long().view(-1))

#             out["cls_loss"] = cls_loss
#             total = total + alpha * cls_loss

#         if seg_crit is not None and "seg_logits" in outputs:
#             seg = outputs["seg_logits"]  # [B,K,H,W]
#             mask = targets.get("mask")
#             if mask is None:
#                 lab = targets.get("label", {})
#                 if isinstance(lab, dict):
#                     mask = lab.get("mask")
#             if mask is None:
#                 raise ValueError("Targets must include 'mask' for segmentation.")
#             mask = torch.as_tensor(mask, device=seg.device)
#             # one-hot to indices if needed
#             if mask.ndim == 4 and mask.shape[1] > 1:
#                 mask = mask.argmax(dim=1)
#             seg_loss = seg_crit(seg, mask.long())
#             out["seg_loss"] = seg_loss
#             total = total + beta * seg_loss

#         out["loss"] = total if isinstance(total, torch.Tensor) else torch.tensor(total, device=logits.device if 'logits' in locals() else seg.device)
#         return out

#     return _loss_fn


# -------------------------------
# Generic builder adapter
# -------------------------------
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
    if "image_size" in inspect.signature(cls).parameters or "img_size" in inspect.signature(cls).parameters:
        kwargs["image_size"] = img

    try:
        return cls(cfg, **kwargs)  # <-- pass cfg FIRST if __init__(cfg, **kw) exists
    except TypeError:
        try:
            return cls(**kwargs)
        except TypeError:
            return cls()  # last resort


# -------------------------------
# Registry: map name -> callable(cfg)->nn.Module
# -------------------------------
BuildFn = Callable[[dict], nn.Module]


def _simple_cnn_builder(cfg: dict) -> nn.Module:
    m = _build_with_class(SimpleCNNModel, cfg)
    _maybe_init_cls_bias(m, cfg)
    return m


def _densenet121_builder(cfg: dict) -> nn.Module:
    m = _build_with_class(DenseNet121Model, cfg)
    _maybe_init_cls_bias(m, cfg)
    return m


def _vit_builder(cfg: dict) -> nn.Module:
    m = _build_with_class(ViTModel, cfg)
    _maybe_init_cls_bias(m, cfg)
    return m


def _unet_builder(cfg: dict) -> nn.Module:
    return _build_with_class(UNetModel, cfg)


def _swin_unetr_builder(cfg: dict) -> nn.Module:
    return _build_with_class(SwinUNETRModel, cfg)


def _multitask_unet_builder(cfg: dict) -> nn.Module:
    m = _build_with_class(MultitaskUNetModel, cfg)
    _maybe_init_cls_bias(m, cfg)
    return m


# def _unet_builder(cfg: dict) -> nn.Module:
#     from unet import UNetSeg
#     in_ch = int(cfg.get("in_channels", 1))
#     K = _seg_out_dim(cfg)
#     return UNetSeg(in_channels=in_ch, out_channels=K)

# def _swin_unetr_builder(cfg: dict) -> nn.Module:
#     from swin_unetr import SwinUNETRSeg
#     in_ch = int(cfg.get("in_channels", 1))
#     K = _seg_out_dim(cfg)
#     img = cfg.get("input_shape", [256, 256])
#     return SwinUNETRSeg(in_channels=in_ch, out_channels=K, image_size=tuple(img))

# def _multitask_unet_builder(cfg: dict) -> nn.Module:
#     from multitask_unet import MultiTaskUNet
#     in_ch = int(cfg.get("in_channels", 1))
#     C = _cls_out_dim(cfg)
#     K = _seg_out_dim(cfg)
#     m = MultiTaskUNet(in_channels=in_ch, cls_out=C, seg_out=K)
#     _maybe_init_cls_bias(m, cfg)
#     return m


MODEL_REGISTRY: Dict[str, BuildFn] = {
    "simple_cnn": _simple_cnn_builder,
    "densenet121": _densenet121_builder,
    "vit": _vit_builder,
    "unet": _unet_builder,
    "swin_unetr": _swin_unetr_builder,
    "multitask_unet": _multitask_unet_builder,
}


# -------------------------------
# Public API
# -------------------------------
def build_model(cfg: dict) -> nn.Module:
    """
    Resolve and build a model from cfg['architecture']; validate the forward contract with a dummy pass.
    The dummy check can be disabled by setting cfg['skip_model_contract_check']=True.
    """
    name = str(cfg.get("architecture", "")).lower().strip()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[name](cfg)

    if bool(cfg.get("skip_model_contract_check", False)):
        return model

    task = str(cfg.get("task", "classification")).lower()
    need_cls = task in ("classification", "multitask")
    need_seg = task in ("segmentation", "multitask")

    # Quick sanity pass to catch shape/key violations early (dev aid)
    try:
        model.eval()
        with torch.no_grad():
            spatial_dims = int(cfg.get("spatial_dims", 2))
            in_ch = int(cfg.get("in_channels", 1))
            shape = tuple(cfg.get("input_shape", (256, 256)))
            bsz = 2
            if spatial_dims == 3:
                if len(shape) == 3:
                    D, H, W = shape
                elif len(shape) == 4 and shape[0] == in_ch:
                    D, H, W = shape[1:4]
                else:
                    D, H, W = (32, 128, 128)
                x = torch.randn(bsz, in_ch, D, H, W)
            else:
                if len(shape) == 2:
                    H, W = shape
                elif len(shape) == 3 and shape[0] == in_ch:
                    H, W = shape[1:3]
                else:
                    H, W = (256, 256)
                x = torch.randn(bsz, in_ch, H, W)

            out = model(x)
        _enforce_forward_contract(out, need_cls=need_cls, need_seg=need_seg)
    except Exception as e:
        raise RuntimeError(f"Model forward contract violation for '{name}': {e}") from e

    return model


__all__ = ["MODEL_REGISTRY", "build_model"]


# # model_registry.py
# from typing import Dict
# from protocols import ModelRegistryProtocol

# from models.simple_cnn import SimpleCNNModel
# from models.densenet121 import DenseNet121Model
# from models.unet import UNetModel
# from models.multitask_unet import MultitaskUNetModel
# from models.vit import ViTModel
# from models.swin_unetr import SwinUNETRModel


# MODEL_REGISTRY: Dict[str, ModelRegistryProtocol] = {
#     "simple_cnn": SimpleCNNModel,
#     "densenet121": DenseNet121Model,
#     "unet": UNetModel,
#     "multitask_unet": MultitaskUNetModel,
#     "vit": ViTModel,
#     "swin_unetr": SwinUNETRModel,
# }


# __all__ = ["MODEL_REGISTRY"]
