# seg_eval.py
from __future__ import annotations

from typing import Any, Dict, Optional
import torch
from ignite.engine import Engine
from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
from monai.data.meta_tensor import MetaTensor
from metrics_utils import seg_confmat_output_transform


def _infer_device(model) -> torch.device:
    try:
        return getattr(model, "device", next(model.parameters()).device)
    except StopIteration:
        return torch.device("cpu")


def _as_tensor(x):
    return x.as_tensor() if isinstance(x, MetaTensor) else x


def _pick_first_seg_like(obj):
    """
    Find the first tensor that looks like segmentation logits (ndim >= 3)
    inside dict/tuple/list/tensor structures. Never relies on truthiness.
    """
    obj = _as_tensor(obj)

    if torch.is_tensor(obj):
        return obj if obj.ndim >= 3 else None

    if isinstance(obj, dict):
        # try common keys first
        for k in ("seg_out", "seg_logits", "seg", "y_pred", "logits_seg"):
            v = _as_tensor(obj.get(k, None))
            if torch.is_tensor(v) and v.ndim >= 3:
                return v
        # fallback: scan nested values
        for v in obj.values():
            r = _pick_first_seg_like(v)
            if r is not None:
                return r

    if isinstance(obj, (list, tuple)):
        for v in obj:
            r = _pick_first_seg_like(v)
            if r is not None:
                return r

    return None


def _coerce_scalar(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item()) if x.numel() == 1 else float(x.detach().cpu().mean().item())
    return float(x)


@torch.no_grad()
def _step(model, batch, device) -> Dict[str, Any]:
    """
    Normalize a batch -> {'seg_out': logits, 'label': {'mask': y_mask}}
    so it fits metrics_utils.seg_output_transform / seg_confmat_output_transform.
    """
    # inputs
    x = batch.get("image", batch.get("x"))
    if x is None:
        raise KeyError("Expected batch['image'] (or 'x') for segmentation inputs.")
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = _as_tensor(x).to(device, non_blocking=True)

    # ground truth mask (accept several common layouts)
    y_mask = None
    if "mask" in batch:
        y_mask = batch["mask"]
    elif "label" in batch and isinstance(batch["label"], dict):
        y_mask = batch["label"].get("mask", None)
    elif "y" in batch:
        y_mask = batch["y"]
    if y_mask is None:
        raise KeyError("Expected batch to contain 'mask' or label['mask'].")
    if not torch.is_tensor(y_mask):
        y_mask = torch.as_tensor(y_mask)
    y_mask = _as_tensor(y_mask).to(device, non_blocking=True)

    # forward pass (be permissive about how the seg head is exposed)
    if hasattr(model, "segment") and callable(getattr(model, "segment")):
        seg_logits = model.segment(x)  # explicit seg head
    else:
        out = model(x)                  # fallback: main forward
        seg_logits = _pick_first_seg_like(out)
        if seg_logits is None:
            raise TypeError("Could not extract segmentation logits from model output.")
    seg_logits = _as_tensor(seg_logits).float()  # [B,C,H,W] (C can be 1 or >1)

    return {"seg_out": seg_logits, "label": {"mask": y_mask}}


_ENGINE_CACHE: Dict[int, Engine] = {}


def _build_seg_eval_engine(model) -> Engine:
    key = id(model)
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key]
    device = _infer_device(model)

    def _ignite_step(_engine, batch):
        model.eval()
        return _step(model, batch, device)
    eng = Engine(_ignite_step)
    _ENGINE_CACHE[key] = eng
    return eng


def model_eval_seg(
    model,
    val_loader,
    *,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Thin shim over segmentation evaluation.

    Returns:
        {
          'dice': float,
          'iou': float,
          'seg_confmat': torch.Tensor [C,C],
          # optionally add more fields if  pipeline computes them
        }
    """
    engine = _build_seg_eval_engine(model)

    # If num_classes not provided, assume binary (2). For C==1 logits the
    # seg_confmat_output_transform will expand to 2-channel probs automatically.
    nc = int(num_classes) if (num_classes is not None) else 2

    # Confusion matrix over the whole epoch (robust logits->probs + mask handling)
    cm_metric = ConfusionMatrix(num_classes=nc, output_transform=seg_confmat_output_transform)
    dice_metric = DiceCoefficient(cm=cm_metric)
    iou_metric = JaccardIndex(cm=cm_metric)

    # Attach metrics
    cm_metric.attach(engine, "seg_confmat")
    dice_metric.attach(engine, "dice")
    iou_metric.attach(engine, "iou")

    # Run once
    engine.run(val_loader)

    # Collect & coerce
    m = dict(engine.state.metrics)
    out = {
        "dice": _coerce_scalar(m.get("dice", 0.0)),
        "iou": _coerce_scalar(m.get("iou", 0.0)),
        "seg_confmat": m.get("seg_confmat"),  # keep as tensor for downstream logging
    }
    return out
