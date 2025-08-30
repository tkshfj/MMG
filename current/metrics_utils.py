# metrics_utils.py
from __future__ import annotations
import inspect
from typing import Any, Mapping, Optional, Tuple
import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Precision, Recall, ROC_AUC, ConfusionMatrix, Metric


# eval helpers
def to_tensor(x: Any) -> torch.Tensor:
    """Convert MetaTensor/array/scalar to torch.Tensor; raise if not possible."""
    if isinstance(x, MetaTensor):
        x = x.as_tensor()
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception as e:
        raise TypeError(f"to_tensor: cannot convert type {type(x)}") from e


def to_py(x: Any) -> Any:
    """Convert tensors/arrays/mappings to JSON-serializable Python values."""
    x = to_tensor(x) if isinstance(x, (MetaTensor, torch.Tensor)) else x
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, Mapping):
        return {k: to_py(v) for k, v in x.items()}
    try:
        return float(x)
    except Exception:
        return str(x)


def labels_to_1d_indices(y: Any) -> torch.Tensor:
    """Coerce labels to shape [B] int64, handling one-hot and [B,1] forms."""
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if y.ndim >= 2:
        if y.shape[-1] > 1:  # one-hot
            y = y.argmax(dim=-1)
        else:
            y = y.view(y.shape[0], -1).squeeze(-1)
    return y.view(-1).to(torch.long)


def positive_score_from_logits(logits: Any, positive_index: int = 1) -> torch.Tensor:
    """
    Return p(pos) for classification, handling:
      - BCE single-logit:    sigmoid(logit)                     -> [B]
      - CE 2+ logits:        softmax(logits)[:, positive_index] -> [B]
    Expects logits shaped [B,C] or [B]; higher dims should be reduced by cls_output_transform first.
    """
    t = to_tensor(logits).float()
    if t.ndim == 1:
        # [B] -> single-logit BCE
        return torch.sigmoid(t)
    if t.ndim == 2:
        C = t.shape[1]
        if C == 1:
            return torch.sigmoid(t.squeeze(1))
        # CE with C≥2
        return torch.softmax(t, dim=1)[:, int(positive_index)]
    raise ValueError(f"positive_score_from_logits: expected [B] or [B,C], got {tuple(t.shape)}")


# def positive_score_from_logits(logits: Any, positive_index: int = 1) -> torch.Tensor:
#     """
#     Return p(pos) for classification, handling:
#       - BCE single-logit:    sigmoid(logit)                     -> [B]
#       - CE 2+ logits:        softmax(logits)[:, positive_index] -> [B]
#     Expects logits shaped [B,C] or [B]; higher dims should be reduced by cls_output_transform first.
#     """
#     t = to_tensor(logits).float()
#     if t.ndim == 1:
#         # [B] -> single-logit BCE
#         return torch.sigmoid(t)
#     if t.ndim == 2:
#         C = t.shape[1]
#         if C == 1:
#             return torch.sigmoid(t.squeeze(1))
#         # CE with C≥2
#         return torch.softmax(t, dim=1)[:, int(positive_index)]
#     raise ValueError(f"positive_score_from_logits: expected [B] or [B,C], got {tuple(t.shape)}")


def get_mask_from_batch(batch: Mapping[str, Any]) -> Optional[torch.Tensor]:
    """
    Robustly fetch a segmentation mask from common batch layouts:
      - batch["mask"]
      - batch["label"]["mask"]
    Returns a tensor or None.
    """
    y_mask = batch.get("mask")
    if y_mask is not None:
        return y_mask
    lab = batch.get("label")
    if isinstance(lab, Mapping):
        return lab.get("mask")
    return None


def looks_like_cls_logits(t: torch.Tensor) -> bool:
    return torch.is_tensor(t) and (t.ndim in (1, 2))


def extract_from_nested(out: Any, keys: Tuple[str, ...]) -> Optional[torch.Tensor]:
    if torch.is_tensor(out):
        return out if looks_like_cls_logits(out) else None
    if isinstance(out, dict):
        for k in keys:
            v = out.get(k, None)
            if torch.is_tensor(v) and looks_like_cls_logits(v):
                return v
        for v in out.values():
            r = extract_from_nested(v, keys)
            if r is not None:
                return r
    if isinstance(out, (list, tuple)):
        for v in out:
            r = extract_from_nested(v, keys)
            if r is not None:
                return r
    return None


def extract_cls_logits_from_any(out: Any) -> torch.Tensor:
    PREFERRED = ("cls_out", "class_logits", "cls_logits", "logits", "y_pred", "cls", "class")
    found = extract_from_nested(out, PREFERRED)
    if found is None:
        raise KeyError(f"Could not find classification logits; tried keys={PREFERRED}.")
    return found


def extract_seg_logits_from_any(out: Any) -> torch.Tensor:
    KEYS = ("seg_out", "seg_logits", "mask_logits", "segmentation", "y_pred_seg", "logits_seg")

    def _walk(node):
        if torch.is_tensor(node) and node.ndim >= 3:  # [B,C,H,W] or similar
            return node
        if isinstance(node, dict):
            for k in KEYS:
                v = node.get(k, None)
                if torch.is_tensor(v) and v.ndim >= 3:
                    return v
            for v in node.values():
                r = _walk(v)
                if r is not None:
                    return r
        if isinstance(node, (list, tuple)):
            for v in node:
                r = _walk(v)
                if r is not None:
                    return r
        return None

    found = _walk(out)
    if found is None:
        raise KeyError(f"Segmentation logits not found; expected one of keys={KEYS} or a [B,C,H,W] tensor.")
    return found


def seg_confmat(pred_idx: torch.Tensor, true_idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Confusion matrix across pixels. Shape [K, K] with rows=true, cols=pred.
    """
    with torch.no_grad():
        k = int(num_classes)
        t = true_idx.view(-1).long()
        p = pred_idx.view(-1).long()
        idx = t * k + p
        cm = torch.bincount(idx, minlength=k * k).reshape(k, k)
    return cm


def add_confmat(flat: dict, key: str, cm) -> None:
    """Add a confusion matrix as a list plus per-cell scalars to a flat dict."""
    cm_t = cm
    if isinstance(cm_t, MetaTensor):
        cm_t = cm_t.as_tensor()
    if torch.is_tensor(cm_t):
        cm_np = cm_t.detach().cpu().to(torch.int64).numpy()
    elif isinstance(cm_t, np.ndarray):
        cm_np = cm_t.astype(np.int64)
    else:
        cm_np = np.asarray(cm_t, dtype=np.int64)
    flat[key] = cm_np.tolist()
    C = cm_np.shape[0]
    for i in range(C):
        for j in range(C):
            flat[f"{key}_{i}{j}"] = int(cm_np[i, j])


def promote_vec(flat: dict, base: str, v) -> None:
    """Promote a vector to per-index keys and a mean; handle scalars robustly."""
    v_t = to_tensor(v) if isinstance(v, (MetaTensor, torch.Tensor)) else v

    # scalar-like
    if not (hasattr(v_t, "shape") and getattr(v_t, "ndim", 0) >= 1):
        flat[base] = float(to_py(v_t))
        return

    arr = v_t.detach().cpu().numpy() if torch.is_tensor(v_t) else np.asarray(v_t)

    if arr.ndim == 1:
        for i, val in enumerate(arr):
            flat[f"{base}_{i}"] = float(val)
        flat[base] = float(arr.mean()) if arr.size else 0.0
        return

    flat[base] = float(np.asarray(arr).mean())


def to_float_scalar(v: Any, *, strict: bool = False) -> float | None:
    """
    Convert many numeric-like shapes to a float scalar.
    - strict=True: only accept scalar-like (0-D tensor/np scalar/python number)
    - strict=False: also collapse tensors/arrays/lists/dicts via mean()
    """
    # Tensors
    if torch.is_tensor(v):
        if v.numel() == 0:
            return 0.0 if not strict else None
        if strict and v.ndim > 0:
            return None
        return float(v.detach().float().mean().item())

    # NumPy arrays & scalars
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return 0.0 if not strict else None
        if strict and v.ndim > 0:
            return None
        return float(v.mean()) if v.ndim > 0 else float(v.item())
    if isinstance(v, (np.floating, np.integer, float, int)):
        return float(v)

    # Containers (only if non-strict)
    if not strict:
        if isinstance(v, (list, tuple)):
            vals = [to_float_scalar(x, strict=False) for x in v]
            vals = [x for x in vals if x is not None]
            return float(np.mean(vals)) if vals else None
        if isinstance(v, Mapping):
            vals = [to_float_scalar(x, strict=False) for x in v.values()]
            vals = [x for x in vals if x is not None]
            return float(np.mean(vals)) if vals else None

    return None


def coerce_loss_dict(loss_out: Any, *, prefer_key: str = "loss") -> dict[str, torch.Tensor]:
    """
    Normalize various loss returns into a dict of TENSORS and ensure a '{prefer_key}' key.
    Accepts:
      - Tensor (scalar or any shape)                -> {prefer_key: tensor.mean()}
      - Mapping[str, Any] (tensor/number values)    -> keep tensor-like values; if missing '{prefer_key}',
                                                       sum all tensor-like values as total
      - Sequence[Any] (list/tuple)                  -> first tensor/number is total; rest -> loss_1, loss_2, ...
    """
    def _to_t(v):
        if torch.is_tensor(v):
            return v
        if isinstance(v, (float, int, np.floating, np.integer)):
            return torch.as_tensor(v, dtype=torch.float32)
        if isinstance(v, np.ndarray):
            return torch.as_tensor(v)
        return None

    # Tensor
    if torch.is_tensor(loss_out):
        t = loss_out
        if t.numel() > 1:
            t = t.mean()
        return {prefer_key: t}

    # Mapping
    if isinstance(loss_out, Mapping):
        out: dict[str, torch.Tensor] = {}
        for k, v in loss_out.items():
            tv = _to_t(v)
            if tv is not None:
                out[str(k)] = tv
        if not out:
            raise TypeError("coerce_loss_dict: mapping had no tensor-like values.")
        if prefer_key not in out:
            # sum all tensor-like values to form total
            out[prefer_key] = sum(out.values())
        return out

    # Sequence
    if isinstance(loss_out, (list, tuple)) and len(loss_out) > 0:
        first = _to_t(loss_out[0])
        if first is None:
            raise TypeError("coerce_loss_dict: first element of sequence must be tensor/number total.")
        if torch.is_tensor(first) and first.numel() > 1:
            first = first.mean()
        out = {prefer_key: first}
        for i, v in enumerate(loss_out[1:], start=1):
            tv = _to_t(v)
            if tv is not None:
                if torch.is_tensor(tv) and tv.numel() > 1:
                    tv = tv.mean()
                out[f"{prefer_key}_{i}"] = tv
        return out

    raise TypeError(f"coerce_loss_dict: unsupported type {type(loss_out)}")


def train_loss_output_transform(output: Any) -> dict[str, float]:
    """
    Normalize many shapes into {'loss': scalar} for StatsHandler/W&B.
    """
    try:
        ld = coerce_loss_dict(output)
        f = to_float_scalar(ld.get("loss"), strict=True)
        if f is None:
            f = to_float_scalar(ld.get("loss"), strict=False)
        return {"loss": f if f is not None else 0.0}
    except Exception:
        f = to_float_scalar(output, strict=False)
        return {"loss": f if f is not None else 0.0}


# Existing metrics & transforms
def _wrap_output_transform(ot, name="output_transform"):
    """
    Wrap a transform so if it erroneously returns a function (e.g., lambda out: ot),
    we call it one more time with the same output. Also sanity-check the return type.
    """
    def wrapped(output):
        res = ot(output)
        if callable(res):  # someone returned a function object; call it with output
            res = res(output)
        if not (isinstance(res, tuple) and len(res) == 2):
            raise TypeError(f"{name} must return a (y_pred, y) tuple, got {type(res)}")
        return res
    return wrapped


def cls_output_transform(output):
    """
    Normalize to (logits[B,C], labels[B]).
    Keeps single-logit binaries as C=1; reduces higher-dim logits by spatial mean.
    """
    # (y_pred, y) already
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        logits, labels = output[0], output[1]

    # Mapping path
    elif isinstance(output, dict):
        logits = extract_cls_logits_from_any(output)

        labels = None
        lab = output.get("label")
        if isinstance(lab, Mapping):
            for k in ("label", "y", "target", "index", "cls"):
                v = lab.get(k)
                if v is not None:
                    labels = v
                    break
        if labels is None:
            labels = output.get("y")
        if labels is None:
            raise ValueError("cls_output_transform: missing labels (label[...] or y).")

    else:
        raise TypeError(f"cls_output_transform: unsupported output type: {type(output)}")

    # Tensorify & standardize shapes
    logits = to_tensor(logits).float()
    labels = labels_to_1d_indices(labels)  # -> [B]

    # Enforce logits [B,C]
    if logits.ndim == 1:              # [B] -> [B,1]
        logits = logits.unsqueeze(1)
    elif logits.ndim == 3:            # [B,T,C] -> first step
        logits = logits[:, 0, :]
    elif logits.ndim >= 4:            # [B,C,H,W,...] -> spatial mean
        logits = logits.mean(dim=tuple(range(2, logits.ndim)))

    if logits.ndim != 2:
        raise ValueError(f"cls_output_transform: expected [B,C], got {tuple(logits.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"cls_output_transform: batch mismatch logits={tuple(logits.shape)} vs labels={tuple(labels.shape)}")

    return logits, labels


def cls_decision_output_transform(decision: str = "threshold", threshold: float = 0.5, positive_index: int = 1):
    """
    Decision transform for Acc/Prec/Recall/ConfMat.
      - decision='threshold': p(pos) via sigmoid (C==1) or softmax[:, positive_index], then compare to threshold
      - decision='argmax'   : argmax over logits
    Returns (y_pred[B], y_true[B]).
    """
    def _ot(output):
        logits, y = cls_output_transform(output)  # logits [B,C], y [B]
        if str(decision).lower() == "argmax":
            y_hat = logits.argmax(dim=1).long()
        else:  # 'threshold' (default)
            C = logits.shape[1]
            if C == 1:
                # single-logit BCE: sigmoid then threshold
                probs = torch.sigmoid(logits.squeeze(1))          # [B]
                # y_hat = (probs >= float(threshold)).long()
                y_hat = (probs > float(threshold)).long()
            elif C == 2:
                # two-logit: softmax then threshold the positive class
                if not (0 <= int(positive_index) < C):
                    raise ValueError(f"positive_index {positive_index} out of range for C={C}")
                probs = torch.softmax(logits, dim=1)[:, int(positive_index)]  # [B]
                # y_hat = (probs >= float(threshold)).long()
                y_hat = (probs > float(threshold)).long()
            else:
                # multi-class: threshold isn't meaningful; fall back to argmax
                y_hat = logits.argmax(dim=1).long()
            # if C >= 2 and not (0 <= int(positive_index) < C):
            #     raise ValueError(f"positive_index {positive_index} out of range for C={C}")
            # scores = positive_score_from_logits(logits, positive_index=int(positive_index))
            # y_hat = (scores >= float(threshold)).to(torch.long)
        return y_hat.view(-1), y.view(-1).long()
    return _ot


def make_cls_output_transform(num_classes: int):
    import torch.nn.functional as F

    def _ot(output):
        logits, y = cls_output_transform(output)  # use the direct one
        if logits.ndim == 1:  # very rare
            logits = F.one_hot(logits.long(), num_classes=num_classes).float()
        return logits, y
    return _ot


# Classification confusion-matrix transform (logits -> Ignite will argmax)
def cls_confmat_output_transform(output):
    """
    Return (y_pred, y_true) where:
      - y_pred: [B, C] logits/probabilities (no argmax here)
      - y_true: [B] int64 labels
    This matches ignite.metrics.ConfusionMatrix's expected input.
    """
    logits, y_true = cls_output_transform(output)  # logits [B,C], y_true [B] (or close)
    # Handle MONAI MetaTensor or numpy gracefully
    if hasattr(logits, "as_tensor"):
        logits = logits.as_tensor()
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    if hasattr(y_true, "as_tensor"):
        y_true = y_true.as_tensor()
    if not torch.is_tensor(y_true):
        y_true = torch.as_tensor(y_true)
    # Ensure float logits and 2D shape
    logits = logits.float()
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim == 2 and logits.shape[1] == 1:
        # Defensive: expand binary [B,1] logits to two-logit form for CM with C=2
        logits = torch.cat([-logits, logits], dim=1)
    if logits.ndim != 2:
        raise ValueError(f"cls_confmat_output_transform expects [B,C] logits, got {tuple(logits.shape)}")
    # Labels to [B] int64 (handle one-hot if it sneaks in)
    if y_true.ndim >= 2 and y_true.shape[-1] > 1:
        y_true = y_true.argmax(dim=-1)
    y_true = y_true.view(-1).long()
    return logits, y_true


def make_auc_metric(pos_index: int = 1):
    def _ot(output):
        logits, y = cls_output_transform(output)
        C = logits.shape[1]
        if C <= 2:
            scores = positive_score_from_logits(logits, positive_index=int(pos_index))
            y_bin = (y.view(-1) == int(pos_index)).long()
            return scores, y_bin
        probs = torch.softmax(logits, dim=1)
        onehot = torch.nn.functional.one_hot(y.view(-1).to(torch.int64), num_classes=C).to(probs.dtype)
        return probs, onehot
    return ROC_AUC(output_transform=_ot)


def auc_output_transform(output, positive_index=1):
    logits, labels = cls_output_transform(output)
    if logits.ndim == 1 or logits.shape[1] <= 2:
        scores = positive_score_from_logits(logits, positive_index=int(positive_index))
        return scores, labels
    probs = torch.softmax(logits, dim=-1)
    onehot = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=probs.shape[1]).to(probs.dtype)
    return probs, onehot


def seg_output_transform(output):
    pred_logits = output.get("seg_out")
    if pred_logits is None:
        pred_logits = output.get("seg_pred")  # last resort
    if pred_logits is None:
        raise ValueError("seg_output_transform: no seg_out/seg_pred in output")

    lbl = output.get("label") or {}
    mask = lbl.get("mask") if isinstance(lbl, dict) else None
    if mask is None:
        mask = output.get("mask")
    if mask is None:
        raise ValueError("seg_output_transform: no mask in output/label")

    pred_logits = to_tensor(pred_logits).float()
    mask = to_tensor(mask).long()
    if mask.device != pred_logits.device:
        mask = mask.to(pred_logits.device)
    # normalize shapes
    if pred_logits.ndim == 3:    # [C,H,W] -> [1,C,H,W]
        pred_logits = pred_logits.unsqueeze(0)
    if mask.ndim == 4:           # one-hot -> argmax
        mask = mask.argmax(dim=1)
    if mask.ndim == 2:           # [H,W] -> [1,H,W]
        mask = mask.unsqueeze(0)

    return pred_logits, mask


def seg_confmat_output_transform(output, *, threshold: float = 0.5):
    """
    Ignite ConfusionMatrix transform for segmentation.
    Returns:
        y_pred: [B, C, H, W] (one-hot for binary; softmax probs for multi-class)
        y:      [B, H, W] integer mask
    """
    seg_logits, mask = seg_output_transform(output)

    if seg_logits.shape[1] == 1:
        # binary: apply chosen threshold, then one-hot so CM’s argmax matches the threshold
        prob_fg = torch.sigmoid(seg_logits)                       # [B,1,H,W]
        y_hat = (prob_fg >= float(threshold)).long()              # [B,1,H,W]
        y_pred = torch.cat([(y_hat == 0).float(), (y_hat == 1).float()], dim=1)  # [B,2,H,W]
    else:
        y_pred = torch.softmax(seg_logits, dim=1)                 # [B,C,H,W]

    return y_pred, mask


def loss_output_transform(output):
    """For multitask loss metrics: return (pred_dict, target_dict) or (y_pred, y)."""
    if isinstance(output, dict) and ("pred" in output) and ("label" in output):
        return output["pred"], output["label"]
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output
    raise ValueError(f"[loss_output_transform] Unexpected output type: {type(output)}")


def _as_unary_output_transform(ot, default_factory, *, name="output_transform"):
    """
    Resolve `ot` into a unary callable: f(output) -> (y_pred, y).

    Accepts:
      - None                     -> use `default_factory` (unary or 0-arg factory)
      - a 0-arg factory          -> call once to obtain unary
      - a unary transform        -> use as is

    Always returns a wrapped unary via _wrap_output_transform so that:
      - if the unary accidentally returns another function, we call it once more
      - the return type is validated as a 2-tuple
    """
    fn = ot if ot is not None else default_factory

    # If we can inspect the signature, treat 0-arg as a factory and call it once
    try:
        sig = inspect.signature(fn)
        if len(sig.parameters) == 0:
            fn = fn()
    except (TypeError, ValueError):
        # Some callables don't expose a Python signature; try calling with no args.
        try:
            cand = fn()
            if callable(cand):
                fn = cand
        except TypeError:
            pass  # assume it's already unary

    if not callable(fn):
        raise TypeError(f"{name}: expected a callable or 0-arg factory, got {type(fn)}")

    return _wrap_output_transform(fn, name)


def _seg_ce_loss_ot(output):
    """Shape y_pred/y for CrossEntropy over segmentation.
       - logits: [B,C,H,W]; if C==1 -> make 2-channel by [-x, x]
       - target: [B,H,W] (Long indices)
    """
    import torch
    logits, y = seg_output_transform(output)  # logits [B,C,H,W], y [B,H,W] (Long)
    if logits.ndim != 4:
        raise ValueError(f"_seg_ce_loss_ot expects logits [B,C,H,W], got {tuple(logits.shape)}")
    if logits.shape[1] == 1:  # binary -> 2-channel logits for CE
        x = logits[:, 0:1]
        logits = torch.cat([-x, x], dim=1)
    return logits, y.long()


def compute_multi(metrics: dict, w: float = 0.65, auc_key: str = "auc", dice_key: str = "dice") -> float:
    """Weighted harmonic mean of AUC and Dice. Returns 0.0 if either is missing/zero."""
    auc = float(metrics.get(auc_key, 0.0) or 0.0)
    dice = float(metrics.get(dice_key, 0.0) or 0.0)
    if auc <= 0.0 or dice <= 0.0:
        return 0.0
    return 1.0 / (w / auc + (1.0 - w) / dice)


def pick_threshold(p: np.ndarray, y: np.ndarray, method="quantile", beta=1.0):
    if method == "quantile":
        base_rate = float(y.mean())
        return float(np.quantile(p, 1.0 - base_rate))
    ts = np.linspace(0.02, 0.98, 97)
    best_t, best_s = 0.5, -1.0
    b2 = beta * beta
    for t in ts:
        pred = (p >= t).astype(np.int32)
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int(((1 - pred) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f = (1 + b2) * prec * rec / max(b2 * prec + rec, 1e-12)
        if f > best_s:
            best_s, best_t = f, float(t)
    return best_t


class ThresholdBox:
    def __init__(self, value=0.5):
        self.value = float(value)


def cls_decision_output_transform_from_box(thr_box: ThresholdBox, positive_index: int = 1):
    """
    Returns an output_transform that thresholds with thr_box.value at call time.
    """
    def _ot(output):
        # reuse existing path
        return cls_decision_output_transform(
            decision="threshold",
            threshold=float(thr_box.value),
            positive_index=positive_index,
        )(output)
    return _ot


def thresholded_pred_transform(thr_box: ThresholdBox, positive_index=1):
    def _xf(o):
        probs = torch.softmax(o["y_pred"], dim=1)[:, positive_index]
        y = o["y"].view(-1).long()
        preds = (probs >= float(thr_box.value)).long()
        return preds, y
    return _xf


class EvalCollector:
    def __init__(self, positive_index=1):
        self.pos = positive_index
        self.p_buf, self.y_buf = [], []

    def attach(self, engine: Engine):
        @engine.on(Events.STARTED)
        def _reset(_):
            self.p_buf.clear()
            self.y_buf.clear()

        @engine.on(Events.ITERATION_COMPLETED)
        def _collect(e):
            out = e.state.output
            logits = extract_cls_logits_from_any(out)
            labels_any = out["label"] if "label" in out else out.get("y")
            labels = labels_to_1d_indices(labels_any)
            probs = positive_score_from_logits(logits, positive_index=self.pos).detach().cpu()
            self.p_buf.append(probs)
            self.y_buf.append(labels.detach().cpu())

    def numpy(self):
        p = torch.cat(self.p_buf).numpy()
        y = torch.cat(self.y_buf).numpy()
        return p, y


def make_collect_engine(model, device):
    def _step(_, batch):
        model.eval()
        with torch.no_grad():
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            pred = model(x)
            logits = extract_cls_logits_from_any(pred)  # <- support dict outputs
            return {"y_pred": logits, "y": y, "logits": logits, "label": y}
    return Engine(_step)


class CalibratedBinaryReport(Metric):
    def __init__(self, pos_index: int = 1, selector: str = "f1"):
        super().__init__(output_transform=cls_output_transform)  # << inject here
        self.pos = int(pos_index)
        self.selector = selector

    def reset(self):
        self._scores, self._labels = [], []

    def update(self, output):
        logits, y = output
        s = positive_score_from_logits(logits, positive_index=self.pos)
        self._scores.append(s.detach().cpu())
        self._labels.append(y.detach().cpu())

    def compute(self):
        s = torch.cat(self._scores) if self._scores else torch.empty(0)
        y = torch.cat(self._labels) if self._labels else torch.empty(0, dtype=torch.long)
        if s.numel() == 0:
            return {"thr": 0.5, "acc": 0.0, "prec": 0.0, "recall": 0.0, "f1": 0.0, "pos_rate": 0.0,
                    "confmat": torch.zeros(2, 2, dtype=torch.long)}
        # simple threshold search: maximize F1 (vectorized over unique scores)
        cand = torch.unique(s).sort().values
        best = {"f1": -1.0, "thr": 0.5}
        for t in cand[::max(1, len(cand) // 512)]:  # subsample for speed
            pred = (s >= t).long()
            tp = ((pred == 1) & (y == 1)).sum().item()
            fp = ((pred == 1) & (y == 0)).sum().item()
            fn = ((pred == 0) & (y == 1)).sum().item()
            tn = ((pred == 0) & (y == 0)).sum().item()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best["f1"]:
                best = {"f1": f1, "thr": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "prec": prec, "rec": rec}

        tp, fp, fn, tn = best["tp"], best["fp"], best["fn"], best["tn"]
        acc = (tp + tn) / max(1, (tp + fp + fn + tn))
        pos_rate = (tp + fp) / max(1, (tp + fp + fn + tn))
        confmat = torch.tensor([[tn, fp], [fn, tp]], dtype=torch.long)
        return {"thr": best["thr"], "acc": acc, "prec": best["prec"],
                "recall": best["rec"], "f1": best["f1"], "pos_rate": pos_rate,
                "confmat": confmat}


def make_calibrated_cls_metrics(pos_index: int = 1) -> dict:
    # single composite metric; wandb/flattening handler can fan out its dict
    return {"cal_report": CalibratedBinaryReport(pos_index=pos_index, selector="f1")}


def attach_calibrated_threshold_glue(
    evaluator_cal, evaluator_std, thr_box, log_key: str = "cal_thr",
    default: float = 0.5, *, val_loader=None, ema_alpha: float = 0.3, max_jump: float = 0.10
):
    from ignite.engine import Events
    import wandb

    @evaluator_cal.on(Events.COMPLETED)
    def _apply_threshold(engine):
        raw_thr = float(engine.state.metrics.get(log_key, default))
        prev = float(getattr(evaluator_std.state, "cal_thr_prev", default))

        # clamp per-epoch jump, then EMA
        clipped = max(prev - max_jump, min(prev + max_jump, raw_thr))
        new_thr = (1.0 - ema_alpha) * prev + ema_alpha * clipped

        thr_box.value = new_thr
        evaluator_std.state.cal_thr_prev = new_thr

        ep = int(getattr(engine.state, "trainer_epoch", engine.state.epoch))
        # Log only with epoch for validation to keep steps monotonic
        wandb.log({"trainer/epoch": ep, "val/cal_thr": new_thr})


# Metric that returns a scalar threshold each epoch
class CalThreshold(Metric):
    """Collect probs & labels; compute a single scalar threshold."""
    def __init__(self, pos_index: int = 1, method: str = "youden",
                 q_bounds=(0.10, 0.90), rate_tolerance: float = 0.10):
        self.pos = int(pos_index)
        self.method = str(method).lower()
        self.q_lo, self.q_hi = float(q_bounds[0]), float(q_bounds[1])
        self.rate_tol = float(rate_tolerance)
        super().__init__(output_transform=cls_output_transform)

    def reset(self):
        self._scores, self._labels = [], []

    def update(self, output):
        logits, y = output
        s = positive_score_from_logits(logits, positive_index=self.pos)
        self._scores.append(s.detach().cpu())
        self._labels.append(y.detach().cpu())

    def compute(self) -> float:
        if not self._scores:
            return 0.5

        scores = torch.cat(self._scores).numpy()
        labels = torch.cat(self._labels).numpy().astype(np.int32)
        base = float(labels.mean()) if labels.size else 0.5

        # search window & candidate set
        qlo, qhi = np.quantile(scores, [self.q_lo, self.q_hi])
        cand = np.unique(scores)
        cand = cand[(cand >= qlo) & (cand <= qhi)]
        stride = max(1, len(cand) // 512)

        # keep predicted-positive rate within a band around base rate
        lo = max(0.0, base - self.rate_tol)
        hi = min(1.0, base + self.rate_tol)

        picked = None

        if self.method in ("f1", "f1max"):
            best_f1, best_t = -1.0, None
            for t in cand[::stride]:
                pred = (scores >= t)
                prate = float(pred.mean())
                if not (lo <= prate <= hi):
                    continue
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                fn = int(((~pred & (labels == 1)).sum()))
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-12)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            picked = best_t

        elif self.method in ("youden", "j"):
            best_j, best_t = -1.0, None
            P = max(int((labels == 1).sum()), 1)
            N = max(int((labels == 0).sum()), 1)
            for t in cand[::stride]:
                pred = (scores >= t)
                prate = float(pred.mean())
                if not (lo <= prate <= hi):
                    continue
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                tpr = tp / P
                fpr = fp / N
                j = tpr - fpr
                if j > best_j:
                    best_j, best_t = j, float(t)
            picked = best_t

        # Robust fallback if the band filtered everything:
        if picked is None:
            picked = float(np.quantile(scores, 1.0 - base))  # rate-match fallback

        return float(np.clip(picked, qlo, qhi))


def make_calibration_probe(*, pos_index: int = 1, method: str = "youden") -> dict:
    """Return a metric dict that computes a scalar threshold each epoch."""
    return {"cal_thr": CalThreshold(pos_index=pos_index, method=method)}


def _decision_with_box_transform(thr_box, positive_index=1):
    def _ot(output):
        logits, y = cls_output_transform(output)
        scores = positive_score_from_logits(logits, positive_index=int(positive_index))
        y_hat = (scores >= float(thr_box.value)).to(torch.long)
        y = y.view(-1).long().to(y_hat.device)
        return y_hat.view(-1), y.view(-1).long()
    return _ot


def _cm_transform_with_box(thr_box, num_classes=2, positive_index=1):
    """ConfusionMatrix wants (y_pred, y). We feed one-hot of thresholded labels."""
    def _ot(output):
        import torch.nn.functional as F
        y_hat, y_true = _decision_with_box_transform(thr_box, positive_index)(output)  # [B], [B]
        y_onehot = F.one_hot(y_hat.to(torch.int64), num_classes=int(num_classes)).to(torch.float32)  # [B,C]
        # return y_onehot, y_true
        return y_onehot, y_true.to(y_onehot.device)
    return _ot


class _PositiveRate(Metric):
    """Predicted positive rate under the live decision rule."""
    def __init__(self, decision_transform, positive_index=1):
        super().__init__(output_transform=decision_transform)
        self.pidx = int(positive_index)

    def reset(self):
        self.k = 0
        self.n = 0

    def update(self, output):
        y_hat, _ = output
        self.k += int((y_hat == self.pidx).sum().item())
        self.n += int(y_hat.numel())

    def compute(self):
        return float(self.k) / max(1, self.n)


class _GroundTruthPositiveRate(Metric):
    """Ground-truth positive base rate (independent of threshold)."""
    def __init__(self, positive_index=1, output_transform=None):
        super().__init__(output_transform=output_transform or cls_output_transform)
        self.pidx = int(positive_index)

    def reset(self):
        self.k = 0
        self.n = 0

    def update(self, output):
        # When output_transform=cls_output_transform, we get (logits, y)
        _, y = output
        self.k += int((y == self.pidx).sum().item())
        self.n += int(y.numel())

    def compute(self):
        return float(self.k) / max(1, self.n)


def make_std_cls_metrics_with_cal_thr(
    *,
    num_classes: int = 2,
    pos_index: int = 1,
    thr_box: ThresholdBox
) -> dict:
    """
    Standard cls metrics that use a live calibrated threshold (thr_box.value).
    Returns keys compatible with logging: acc, prec, recall, cls_confmat,
    plus pos_rate and gt_pos_rate for the health monitor.
    """

    dec_ot = _decision_with_box_transform(thr_box, positive_index=pos_index)
    cm_ot = _cm_transform_with_box(thr_box, num_classes=num_classes, positive_index=pos_index)

    def _auc_ot(output):
        logits, y = cls_output_transform(output)
        C = logits.shape[1]
        if C <= 2:
            scores = positive_score_from_logits(logits, positive_index=int(pos_index))
            y_bin = (y.view(-1) == int(pos_index)).long()
            return scores, y_bin
        probs = torch.softmax(logits, dim=1)
        onehot = torch.nn.functional.one_hot(y.view(-1).to(torch.int64), num_classes=probs.shape[1]).to(probs.dtype)
        return probs, onehot

    return {
        "acc": Accuracy(output_transform=dec_ot),
        "prec": Precision(output_transform=dec_ot, average="macro"),
        "recall": Recall(output_transform=dec_ot, average="macro"),
        "auc": ROC_AUC(output_transform=_auc_ot),
        "cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cm_ot),
        "pos_rate": _PositiveRate(dec_ot, positive_index=pos_index),
        "gt_pos_rate": _GroundTruthPositiveRate(positive_index=pos_index, output_transform=cls_output_transform),
    }


def make_metrics(
    tasks,
    num_classes,
    loss_fn=None,
    cls_ot=None,            # can be unary or 0-arg factory
    auc_ot=None,            # can be unary or 0-arg factory
    seg_cm_ot=None,         # can be unary or 0-arg factory
    seg_num_classes=None,
    multitask=False,
    cls_decision: str = "argmax",
    cls_threshold: float = 0.5,
    positive_index: int = 1,
    **_,
):
    from ignite.metrics import Accuracy, ConfusionMatrix, Loss, DiceCoefficient, JaccardIndex, Precision, Recall, MetricsLambda
    import torch.nn as nn
    import torch
    # import torch.nn.functional as F

    tasks = set(tasks or [])
    has_cls = "classification" in tasks
    has_seg = "segmentation" in tasks
    mixed_task = (has_cls and has_seg and not multitask)

    metrics = {}

    # resolve base_cls_ot (raw)
    base_cls_ot = _as_unary_output_transform(cls_ot, default_factory=lambda: cls_output_transform, name="cls_ot")
    # decision OT
    decision_ot = cls_decision_output_transform(decision=str(cls_decision).lower(), threshold=float(cls_threshold), positive_index=int(positive_index))

    # Classification metrics
    if has_cls or multitask:
        # Decision OT stays a plain unary (already returns (y_hat, y))
        # decision_ot = cls_decision_output_transform(decision=str(cls_decision).lower(), threshold=float(cls_threshold), positive_index=int(positive_index))

        # def _cm_ot(output):
        #     y_hat, y_true = decision_ot(output)
        #     y_onehot = F.one_hot(y_hat.to(torch.int64), num_classes=int(num_classes)).to(torch.float32)
        #     return y_onehot, y_true

        class _PositiveRate_local(Metric):
            def reset(self):
                self.k = 0
                self.n = 0

            def update(self, output):
                y_hat, _ = decision_ot(output)
                self.k += int((y_hat == int(positive_index)).sum().item())
                self.n += int(y_hat.numel())

            def compute(self):
                return float(self.k) / max(1, self.n)

        class _GroundTruthPositiveRate_local(Metric):
            def reset(self):
                self.k = 0
                self.n = 0

            def update(self, output):
                _, y = base_cls_ot(output)  # uses resolved unary
                self.k += int((y == int(positive_index)).sum().item())
                self.n += int(y.numel())

            def compute(self):
                return float(self.k) / max(1, self.n)

        metrics.update({
            "acc": Accuracy(output_transform=decision_ot),
            "prec": Precision(output_transform=decision_ot, average=False),
            "recall": Recall(output_transform=decision_ot, average=False),
            "auc": make_auc_metric(pos_index=int(positive_index)),
            # "cls_confmat": ConfusionMatrix(num_classes=int(num_classes), output_transform=_cm_ot),
            "cls_confmat": ConfusionMatrix(
                num_classes=int(num_classes),
                output_transform=lambda o: (
                    torch.nn.functional.one_hot(
                        decision_ot(o)[0].to(torch.int64), num_classes=int(num_classes)).to(torch.float32), decision_ot(o)[1])),
            "pos_rate": _PositiveRate_local(),
            "gt_pos_rate": _GroundTruthPositiveRate_local(),
        })

    # Segmentation metrics
    if has_seg or multitask:
        seg_nc = max(2, int(seg_num_classes) if seg_num_classes is not None else 2)

        # If the user provides seg_cm_ot, accept factory or unary.
        # Otherwise, use robust default (already unary).
        seg_cm_transform = _as_unary_output_transform(
            seg_cm_ot,
            default_factory=lambda: seg_confmat_output_transform,
            name="seg_cm_ot",
        )
        _seg_cm = ConfusionMatrix(num_classes=seg_nc, output_transform=seg_cm_transform)

        metrics.update({
            "dice": DiceCoefficient(cm=_seg_cm),
            "iou": JaccardIndex(cm=_seg_cm),
            "seg_confmat": _seg_cm,
        })

    # Loss metrics
    if mixed_task:
        metrics["cls_loss"] = Loss(loss_fn=loss_fn or nn.CrossEntropyLoss(), output_transform=base_cls_ot)
        metrics["seg_loss"] = Loss(loss_fn=nn.CrossEntropyLoss(), output_transform=_seg_ce_loss_ot)
    elif multitask:
        if loss_fn is None:
            raise ValueError("multitask=True requires a combined loss_fn")
        metrics["loss"] = Loss(loss_fn=loss_fn, output_transform=loss_output_transform)
    else:
        if has_cls and not has_seg:
            metrics["loss"] = Loss(loss_fn=loss_fn or nn.CrossEntropyLoss(), output_transform=base_cls_ot)
        elif has_seg and not has_cls:
            metrics["loss"] = Loss(loss_fn=nn.CrossEntropyLoss() if loss_fn is None else loss_fn, output_transform=_seg_ce_loss_ot)

    if multitask and ("auc" in metrics) and ("dice" in metrics):
        w = float(_.get("multi_weight", 0.65))
        eps = 1e-8
        metrics["multi"] = MetricsLambda(
            lambda auc, dice: 1.0 / (w / (auc + eps) + (1.0 - w) / (dice + eps)),
            metrics["auc"], metrics["dice"]
        )

    return metrics


__all__ = [
    "to_tensor", "to_py", "labels_to_1d_indices", "get_mask_from_batch",
    "looks_like_cls_logits", "extract_from_nested", "extract_cls_logits_from_any",
    "extract_seg_logits_from_any", "seg_confmat", "add_confmat", "promote_vec",
    "make_metrics", "cls_output_transform", "auc_output_transform", "seg_output_transform",
    "make_auc_metric", "cls_confmat_output_transform", "seg_confmat_output_transform",
    "loss_output_transform", "cls_decision_output_transform",
    "ThresholdBox", "cls_decision_output_transform_from_box", "compute_multi",
    "pick_threshold", "EvalCollector", "make_collect_engine", "make_calibrated_cls_metrics",
    "attach_calibrated_threshold_glue", "CalThreshold", "make_calibration_probe",
    "make_std_cls_metrics_with_cal_thr", "to_float_scalar", "train_loss_output_transform", "coerce_loss_dict",
]
