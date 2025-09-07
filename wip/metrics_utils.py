# metrics_utils.py
from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Union
from dataclasses import dataclass
from functools import partial
import torch
import numpy as np
from monai.data.meta_tensor import MetaTensor
from ignite.metrics import Accuracy, Precision, Recall, ROC_AUC, Metric, MetricsLambda
from ignite.metrics import DiceCoefficient, JaccardIndex, EpochMetric
from ignite.metrics.confusion_matrix import ConfusionMatrix
from utils.safe import to_tensor, labels_to_1d_indices, to_float_scalar, to_py


# Helpers
def get_threshold_from_state(engine) -> float:
    # Single source of truth for threshold
    thr = getattr(engine.state, "threshold", None)
    return float(thr if thr is not None else 0.5)


def probs_from_logits(
    logits: Any, *, num_classes: int = 2, positive_index: int = 1
) -> torch.Tensor:
    """
    Unified conversion:
    - binary → p(pos) ∈ [N]
    - multiclass → softmax probabilities ∈ [N, C]
    """
    t = to_tensor(logits).float()
    if int(num_classes) <= 2:
        return positive_score_from_logits(t, positive_index=positive_index).view(-1)
    return torch.softmax(t, dim=-1)


def make_cm_rates(cm: ConfusionMatrix) -> dict[str, Metric]:
    """Return MetricsLambda for predicted-positive and GT-positive rates from a ConfusionMatrix metric."""

    def _pred_pos_rate(m):
        tn, fp, fn, tp = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        total = tn + fp + fn + tp
        return (fp + tp) / (total + (total == 0))

    def _gt_pos_rate(m):
        tn, fp, fn, tp = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        total = tn + fp + fn + tp
        return (fn + tp) / (total + (total == 0))

    return {
        "pos_rate": MetricsLambda(_pred_pos_rate, cm),
        "gt_pos_rate": MetricsLambda(_gt_pos_rate, cm),
    }


# Debug stats metric (global min/median/max over an epoch)
class PosProbStats(Metric):
    """
    Computes min/median/max/mean over per-sample positive-class probabilities.
    Expects output_transform to yield (probs, y_true).
    """
    def __init__(self, output_transform=None, device=None):
        if output_transform is None:
            def _identity(x):
                return x
            output_transform = _identity
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._vals = []

    def update(self, output):
        probs, _ = output
        self._vals.append(to_tensor(probs).detach().reshape(-1).to("cpu"))

    def compute(self):
        if not self._vals:
            return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
        v = torch.cat(self._vals, dim=0)
        return {
            "min": v.min().item(),
            "median": v.median().item(),
            "max": v.max().item(),
            "mean": v.mean().item(),
        }


# Metric attachers
def attach_classification_metrics(evaluator, cfg: Mapping[str, Any]) -> None:
    def thr_getter() -> float:
        return get_threshold_from_state(evaluator)

    n_cls = int(cfg.get("num_classes", 2))
    pos_idx = int(cfg.get("positive_index", 1))

    metrics = make_cls_val_metrics(
        num_classes=n_cls,
        decision="threshold",
        threshold=thr_getter,
        positive_index=pos_idx,
    )
    # for name, metric in pack.items():
    #     tag = "confmat" if name == "cls_confmat" else name
    #     metric.attach(evaluator, f"val/{tag}")
    metrics["auc"].attach(evaluator, "val/auc")
    metrics["cls_confmat"].attach(evaluator, "val/confmat")
    if n_cls == 2:  # only when valid
        metrics["pos_rate"].attach(evaluator, "val/pos_rate")

    auc_ot = partial(cls_proba_output_transform, positive_index=pos_idx, num_classes=n_cls)
    PosProbStats(output_transform=auc_ot).attach(evaluator, "val/posprob_stats")


# Helpers
def _first_not_none(m: Mapping[str, Any], keys: tuple[str, ...]):
    """Return the first present key whose value is not None."""
    for k in keys:
        if k in m:
            v = m[k]
            if v is not None:
                return v
    return None


def positive_score_from_logits(
    logits: Any,
    *,
    positive_index: int = 1,
    # New: accept flags or an explicit mode
    binary_single_logit: Optional[bool] = None,
    binary_bce_from_two_logits: Optional[bool] = None,
    mode: Literal["auto", "bce_single", "ce_softmax", "bce_two_logit"] = "auto",
) -> torch.Tensor:
    """
    Return p(pos) for binary classification.

    Modes (selected either by flags or explicitly via `mode`):
      - "bce_single":     sigmoid(logit)                            -> [B]
      - "ce_softmax":     softmax(logits, -1)[..., pos_idx]         -> [B]
      - "bce_two_logit":  sigmoid(logits[..., pos_idx])             -> [B]

    Auto-resolution rules (if mode="auto"):
      1) If binary_single_logit is True          -> bce_single
      2) elif binary_bce_from_two_logits is True -> bce_two_logit
      3) else:
           - if last dim == 1 or tensor is 1D    -> bce_single
           - else                                -> ce_softmax

    Accepts logits shaped [..., C] or [...] where C∈{1,2+}; higher dims already
    reduced by caller for classification (i.e., [B] or [B,C]).
    """
    t = to_tensor(logits).float()
    # normalize shape references to last dim
    last_dim = t.shape[-1] if t.ndim > 0 else 1

    # Resolve mode if auto
    if mode == "auto":
        if binary_single_logit is True:
            mode = "bce_single"
        elif binary_bce_from_two_logits is True:
            mode = "bce_two_logit"
        else:
            if t.ndim == 1 or last_dim == 1:
                mode = "bce_single"
            else:
                mode = "ce_softmax"

    # Apply chosen mapping
    if mode == "bce_single":
        # supports [B] or [B,1] ... → [B]
        if t.ndim >= 2 and last_dim == 1:
            t = t.squeeze(-1)
        return torch.sigmoid(t)

    if mode == "bce_two_logit":
        # expect 2 logits (or at least a channel dimension)
        if t.ndim == 1:
            # edge case: if already the positive logit flattened
            return torch.sigmoid(t)
        return torch.sigmoid(t[..., int(positive_index)])

    if mode == "ce_softmax":
        # softmax over channel dim; then select positive channel
        probs = torch.softmax(t, dim=-1)
        return probs[..., int(positive_index)]

    raise ValueError(
        f"positive_score_from_logits: unsupported shape {tuple(t.shape)} for mode='{mode}'"
    )


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


def extract_from_nested(out: Any, keys: Tuple[str, ...]) -> Optional[torch.Tensor]:

    def looks_like_cls_logits(t: torch.Tensor) -> bool:
        return torch.is_tensor(t) and (t.ndim in (1, 2))

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


def _extract_logits_and_labels(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        y_pred_like, y_true_like = output[0], output[1]

        # logits
        if isinstance(y_pred_like, Mapping):
            logits = extract_cls_logits_from_any(y_pred_like)
        else:
            logits = y_pred_like

        # labels
        if isinstance(y_true_like, Mapping):
            labels_any = _first_not_none(y_true_like, ("label", "y", "target"))
        else:
            labels_any = y_true_like

    elif isinstance(output, Mapping):
        # logits (no tensor truthiness)
        logits = extract_cls_logits_from_any(output)

        # top-level labels (no `or` chains)
        labels_any = _first_not_none(output, ("label", "y", "target"))
        if isinstance(labels_any, Mapping):
            labels_any = _first_not_none(labels_any, ("label", "y", "target"))

    else:
        raise TypeError(f"Unsupported output type for output_transform: {type(output)}")

    if logits is None:
        raise KeyError("Missing logits for classification.")
    if labels_any is None:
        raise KeyError("Missing labels for classification.")

    y_pred = torch.as_tensor(logits)
    y_true = torch.as_tensor(labels_any)

    # Normalize labels to 1D indices
    try:
        y_true = labels_to_1d_indices(y_true)
    except Exception:
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = y_true.argmax(dim=-1)
        else:
            y_true = y_true.view(-1)

    return y_pred.float(), y_true.long().view(-1)


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


def coerce_loss_dict(loss_out: Any, *, prefer_key: str = "loss") -> dict[str, torch.Tensor]:
    import numpy as np

    def _to_t(v):
        if torch.is_tensor(v):
            return v
        if isinstance(v, (float, int, np.floating, np.integer)):
            return torch.tensor(float(v), dtype=torch.float32)
        if isinstance(v, np.ndarray):
            return torch.as_tensor(v, dtype=torch.float32)
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
            raise TypeError("coerce_loss_dict: mapping had no tensor/number values.")
        if prefer_key not in out:
            out[prefer_key] = sum(out.values())
        return out

    # Sequence
    if isinstance(loss_out, (list, tuple)) and len(loss_out) > 0:
        first = _to_t(loss_out[0])
        if first is None:
            raise TypeError("coerce_loss_dict: first element must be tensor/number total.")
        if first.numel() > 1:
            first = first.mean()
        out = {prefer_key: first}
        for i, v in enumerate(loss_out[1:], start=1):
            tv = _to_t(v)
            if tv is not None:
                if tv.numel() > 1:
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


def _resolve_threshold(thr: Union[float, Callable[[], float]]) -> float:
    return float(thr() if callable(thr) else thr)


class SelectIndex(Metric):
    def __init__(self, base, index: int):
        # Allow passing a metric factory (callable) or an instance
        if callable(base):
            base = base()  # instantiate
        if not isinstance(base, Metric):
            raise TypeError(f"SelectIndex base must be an ignite Metric, got {type(base)}")
        self.base = base
        self.index = int(index)
        super().__init__()  # will call self.reset()

    def reset(self):
        self.base.reset()

    def update(self, output):
        self.base.update(output)

    def compute(self):
        vec = self.base.compute()
        # vec expected shape: [C] (per-class). Support tensor/list/ndarray.
        # import torch, numpy as np
        if torch.is_tensor(vec):
            vec = vec.detach().cpu()
            return float(vec[self.index].item())
        if isinstance(vec, np.ndarray):
            return float(vec[self.index].item())
        return float(vec[self.index])


# class SelectIndex(Metric):
#     """
#     Wrap a per-class Metric (average=False) and return the value at `index`.
#     """
#     def __init__(self, base: Metric, index: int):
#         super().__init__()
#         self.base = base
#         self.index = int(index)

#     def reset(self):
#         self.base.reset()

#     def update(self, output):
#         self.base.update(output)

#     def compute(self):
#         v = self.base.compute()           # tensor/list/ndarray
#         t = torch.as_tensor(v)
#         if t.numel() == 0:
#             return 0.0
#         idx = max(0, min(self.index, t.numel() - 1))
#         return float(t.reshape(-1)[idx].item())


# Standard classification output transforms
@dataclass(frozen=True)
class ClsOTs:
    """Pack of classification output transforms."""
    base: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]        # (prob_pos, y)
    thresholded: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]  # (y_hat, y)
    cm: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]          # ([B,C] scores, y)


def make_default_cls_output_transforms(
    *,
    decision: str = "threshold",
    threshold: Union[float, Callable[[], float]] = 0.5,
    positive_index: int = 1,
) -> ClsOTs:
    """
    Default, model-agnostic classification transforms with a single source of truth
    for the decision rule and threshold.
    """

    def base(output):
        logits, y_true = cls_output_transform(output)
        prob = positive_score_from_logits(logits, positive_index=positive_index).view(-1)
        # prob = _prob_pos_from_logits(logits)
        return prob, y_true.long().view(-1)

    def thresholded(output):
        prob, y_true = base(output)
        thr = _resolve_threshold(threshold)
        y_hat = (prob >= thr).long()
        return y_hat, y_true

    # CM: use the threshold-aware transform we already standardized
    cm = partial(
        cls_confmat_output_transform_thresholded,
        decision=decision,
        threshold=threshold,
        positive_index=positive_index,
    )

    return ClsOTs(base=base, thresholded=thresholded, cm=cm)


def cls_output_transform(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels)."""
    logits, y = _extract_logits_and_labels(output)
    return logits.float(), y.long().view(-1)


def cls_decision_output_transform(
    output: Any, *,
    decision: str = "threshold",
    # threshold: float = 0.5,
    threshold: Union[float, Callable[[], float]] = 0.5,
    positive_index: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pred_indices, labels) for Acc/Prec/Rec/ConfMat."""
    logits, y = cls_output_transform(output)
    dec = str(decision).lower()
    if dec == "argmax" or (logits.ndim >= 2 and logits.shape[-1] > 2):
        y_hat = logits.argmax(dim=-1).long()
    else:
        p_pos = positive_score_from_logits(logits, positive_index=positive_index).view(-1)
        thr = _resolve_threshold(threshold)
        y_hat = (p_pos >= float(thr)).long()
    return y_hat.view(-1), y.view(-1)


def cls_proba_output_transform(
    output: Any, *, positive_index: int = 1, num_classes: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (probs, labels) for ROC-AUC (binary->[N], multi->[N,C])."""
    logits, y = cls_output_transform(output)
    return probs_from_logits(logits, num_classes=int(num_classes), positive_index=int(positive_index)), y


def cls_confmat_output_transform_thresholded(
    output,
    *,
    decision: str = "threshold",
    threshold: float | Callable[[], float] = 0.5,
    positive_index: int = 1,
):
    # Reuse robust parsing
    logits, y_true = cls_output_transform(output)

    # ensure tensor
    if hasattr(logits, "as_tensor"):
        logits = logits.as_tensor()
    logits = torch.as_tensor(logits, dtype=torch.float32)

    if hasattr(y_true, "as_tensor"):
        y_true = y_true.as_tensor()
    y_true = torch.as_tensor(y_true, dtype=torch.long).view(-1)

    # multiclass or explicit argmax path: pass [B,C] scores; CM will argmax
    if decision == "argmax" or (logits.ndim >= 2 and logits.shape[-1] > 2):
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        return logits.float(), y_true

    # binary/threshold path: build 2-channel one-hot so CM’s argmax == thresholded decision
    thr = _resolve_threshold(threshold)
    # get P(pos)
    p_pos = positive_score_from_logits(logits, positive_index=positive_index).view(-1)
    y_hat = (p_pos >= thr).float()  # [B] in {0,1}
    y_pred = torch.stack([1.0 - y_hat, y_hat], dim=1)  # [B,2]
    return y_pred, y_true


def make_cls_val_metrics(
    num_classes: int,
    decision: str = "threshold",
    threshold: Union[float, Callable[[], float]] = 0.5,
    positive_index: int = 1,
) -> Dict[str, Any]:
    # Reuse the canonical pack to avoid drift
    ots = make_default_cls_output_transforms(
        decision=decision, threshold=threshold, positive_index=positive_index
    )
    acc = Accuracy(output_transform=ots.thresholded)
    # prec = Precision(output_transform=ots.thresholded, average=True)
    # rec = Recall(output_transform=ots.thresholded, average=True)
    # precision/recall: macro and per-class → positive-class selection
    prec_macro = Precision(output_transform=ots.thresholded, average=True)
    prec_vec = Precision(output_transform=ots.thresholded, average=False)
    prec_pos = SelectIndex(prec_vec, positive_index)

    rec_macro = Recall(output_transform=ots.thresholded, average=True)
    rec_vec = Recall(output_transform=ots.thresholded, average=False)
    rec_pos = SelectIndex(rec_vec, positive_index)
    auc = ROC_AUC(output_transform=ots.base)
    cm = ConfusionMatrix(num_classes=int(num_classes), output_transform=ots.cm)
    result = {
        "acc": acc,
        "precision_macro": prec_macro,
        "precision_pos": prec_pos,
        "recall_macro": rec_macro,
        "recall_pos": rec_pos,
        "auc": auc,
        "cls_confmat": cm,
        # "acc": acc, "prec": prec, "recall": rec, "auc": auc, "cls_confmat": cm,
        # "pos_rate": rates["pos_rate"],
        # "gt_pos_rate": rates["gt_pos_rate"],
    }
    if int(num_classes) == 2:
        result.update(make_cm_rates(cm))
    return result


def make_std_cls_metrics_with_cal_thr(
    num_classes: int,
    *,
    decision: str = "threshold",
    positive_index: int = 1,
    thr_getter: Callable[[], float] | float = 0.5,
    **kwargs,
):
    if "pos_index" in kwargs:   # back-compat
        positive_index = int(kwargs.pop("pos_index"))

    decision_ot = partial(cls_decision_output_transform, decision=decision, threshold=thr_getter, positive_index=positive_index)
    proba_ot = partial(cls_proba_output_transform, positive_index=positive_index, num_classes=num_classes)
    cm_ot_cal = partial(cls_confmat_output_transform_thresholded, decision=decision, threshold=thr_getter, positive_index=positive_index)

    acc = Accuracy(output_transform=decision_ot)
    prec_macro = Precision(output_transform=decision_ot, average=True)
    prec_vec = Precision(output_transform=decision_ot, average=False)
    prec_pos = SelectIndex(prec_vec, positive_index)

    rec_macro = Recall(output_transform=decision_ot, average=True)
    rec_vec = Recall(output_transform=decision_ot, average=False)
    rec_pos = SelectIndex(rec_vec, positive_index)

    cm = ConfusionMatrix(num_classes=int(num_classes), output_transform=cm_ot_cal)

    metrics: dict[str, Metric] = {
        "acc": acc,
        "precision_macro": prec_macro,
        "precision_pos": prec_pos,
        "recall_macro": rec_macro,
        "recall_pos": rec_pos,
        "cls_confmat": cm,
    }

    # metrics: dict[str, Metric] = {
    #     "acc": Accuracy(output_transform=decision_ot),
    #     "prec": Precision(output_transform=decision_ot, average=(int(num_classes) <= 2)),
    #     "recall": Recall(output_transform=decision_ot, average=(int(num_classes) <= 2)),
    #     "cls_confmat": ConfusionMatrix(num_classes=int(num_classes), output_transform=cm_ot_cal),
    # }

    if int(num_classes) <= 2:
        metrics["auc"] = ROC_AUC(output_transform=proba_ot)
    else:
        def _sk_auc(y_pred, y_true):
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
        metrics["auc"] = EpochMetric(_sk_auc, output_transform=proba_ot)

    if int(num_classes) == 2:
        metrics.update(make_cm_rates(cm))

    return metrics


def _parse_seg_logits_and_target(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (seg_logits [B,C,H,W] with C>=1, y_true indices [B,H,W]).
    Accepts common dict/tuple shapes.
    """
    if isinstance(output, (list, tuple)) and len(output) == 2:
        preds, targets = output
        out = preds if isinstance(preds, Mapping) else {"seg_out": preds}
        lbl = targets if isinstance(targets, Mapping) else {"mask": targets}
    elif isinstance(output, Mapping):
        out = output
        lbl = out.get("label", {})
    else:
        raise ValueError(f"_parse_seg_logits_and_target: unexpected type {type(output)}")

    seg_logits = extract_seg_logits_from_any(out)
    y = None
    if isinstance(lbl, Mapping):
        y = _first_not_none(lbl, ("mask", "y", "label"))
    if y is None:
        y = out.get("mask", None)
    if y is None:
        raise ValueError("_parse_seg_logits_and_target: no mask in output/label")

    seg_logits = to_tensor(seg_logits).float()
    y = to_tensor(y)

    # Normalize target to indices [B,H,W]
    if y.ndim == 2:
        y = y.unsqueeze(0)
    elif y.ndim == 4:
        y = y.argmax(dim=1) if y.shape[1] > 1 else y[:, 0]
    elif y.ndim != 3:
        raise ValueError(f"_parse_seg_logits_and_target: bad target shape {tuple(y.shape)}")
    return seg_logits, y.long()


def seg_confmat_output_transform_default(output: Any, *, threshold: float = 0.5):
    """
    Output transform for segmentation ConfusionMatrix.
    Returns (y_pred_indices[B,H,W], y_true_indices[B,H,W]).
    - Multiclass: argmax over channel C
    - Binary (C==1 or no channel dim): sigmoid + threshold
    Accepts common prediction dicts: {"seg_logits": ...} or {"seg": ...} or {"pred": ...}
    Accepts y_true as indices [B,H,W] or one-hot [B,C,H,W].
    """
    seg, y = _parse_seg_logits_and_target(output)

    # Discretize prediction
    if seg.ndim == 4 and seg.size(1) > 1:
        # multiclass
        y_hat = seg.argmax(dim=1).long()
    else:
        # binary (C==1 or missing C)
        s = seg if seg.ndim == 3 else seg.squeeze(1)  # [B,H,W]
        y_hat = (torch.sigmoid(s) >= float(threshold)).long()

    return y_hat, y.long()


def seg_output_transform(
    output,
    *,
    threshold: float = 0.5,
    # num_classes: Optional[int] = None,
):

    # parse container
    if isinstance(output, (list, tuple)) and len(output) == 2:
        preds, targets = output
        out = preds if isinstance(preds, Mapping) else {"seg_out": preds}
        lbl = targets if isinstance(targets, Mapping) else {"mask": targets}
    elif isinstance(output, Mapping):
        out = output
        lbl = out.get("label", {})
    else:
        raise ValueError(f"seg_output_transform: unexpected output type {type(output)}")

    # fetch tensors safely (no `or` on tensors)
    pred = _first_not_none(out, ("seg_logits", "seg_out", "seg_pred", "pred"))
    if pred is None:
        raise ValueError("seg_output_transform: no seg_out/seg_logits/seg_pred/pred in output")

    mask = _first_not_none(lbl, ("mask", "y", "label")) if isinstance(lbl, Mapping) else None
    if mask is None:
        mask = out.get("mask", None)
    if mask is None:
        raise ValueError("seg_output_transform: no mask in output/label")

    pred = to_tensor(pred).float()
    y = to_tensor(mask)

    # normalize shapes
    # Accept [H,W], [B,H,W], [B,1,H,W], [B,C,H,W]
    if pred.ndim == 2:               # [H, W]
        pred = pred.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif pred.ndim == 3:             # [B, H, W]
        pred = pred.unsqueeze(1)               # [B,1,H,W]
    elif pred.ndim != 4:
        raise ValueError("seg_output_transform: pred must be [B,C,H,W]")

    if y.ndim == 2:
        y = y.unsqueeze(0)
    elif y.ndim == 4:
        y = y.argmax(dim=1) if y.shape[1] > 1 else y[:, 0]
    elif y.ndim != 3:
        raise ValueError(f"seg_output_transform: unsupported target shape {tuple(y.shape)}")
    y = y.long()

    C = pred.shape[1]
    if C == 1:
        y_pred = (pred.sigmoid() >= float(threshold)).long().squeeze(1)
    else:
        y_pred = pred.argmax(dim=1).long()

    return y_pred, y


def seg_confmat_output_transform(output, *, threshold: float = 0.5):
    """
    ConfusionMatrix transform for segmentation.
    Returns:
      y_pred: [B, K, H, W] (one-hot for binary at threshold; softmax probs for multi-class)
      y:      [B, H, W] integer mask
    """

    seg_logits, y = _parse_seg_logits_and_target(output)

    C = seg_logits.shape[1]
    if C == 1:
        prob_fg = torch.sigmoid(seg_logits)                  # [B,1,H,W]
        y_hat = (prob_fg >= float(threshold)).float()        # [B,1,H,W] in {0,1}
        y_pred = torch.cat([1.0 - y_hat, y_hat], dim=1)      # [B,2,H,W]
    else:
        y_pred = torch.softmax(seg_logits, dim=1)            # [B,C,H,W]

    return y_pred, y


def make_seg_val_metrics(*, num_classes: int, threshold: float = 0.5, ignore_index: Optional[int] = None):
    _ot = partial(seg_confmat_output_transform, threshold=threshold)
    cm = ConfusionMatrix(num_classes=num_classes, output_transform=_ot)
    try:
        dice = DiceCoefficient(cm, ignore_index=ignore_index) if ignore_index is not None else DiceCoefficient(cm)
    except TypeError:
        dice = DiceCoefficient(cm)
    try:
        iou = JaccardIndex(cm, ignore_index=ignore_index) if ignore_index is not None else JaccardIndex(cm)
    except TypeError:
        iou = JaccardIndex(cm)
    return {"seg_confmat": cm, "seg_dice": dice, "seg_iou": iou}


def make_metrics(
    *,
    tasks,
    num_classes: int,
    cls_decision: str = "threshold",
    cls_threshold=0.5,  # float or callable
    positive_index: int = 1,
    seg_num_classes: int | None = None,
    seg_threshold: float = 0.5,
    loss_fn=None,
    cls_ot=None,   # thresholded (y_hat, y)
    auc_ot=None,   # base (prob, y)
    seg_cm_ot=None,
    multitask: bool = False,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    # If caller didn't provide output transforms, build the standard pack
    if cls_ot is None or auc_ot is None:
        ots = make_default_cls_output_transforms(
            decision=cls_decision,
            threshold=cls_threshold,
            positive_index=positive_index,
        )
        # thresholded -> (y_hat, y) for Acc/Prec/Rec
        cls_ot = ots.thresholded
        # base -> (p_pos, y) for AUC
        auc_ot = ots.base

    # classification
    if "classification" in tasks or multitask:
        acc = Accuracy(output_transform=cls_ot)
        # prec = Precision(output_transform=cls_ot, average=True)
        # rec = Recall(output_transform=cls_ot, average=True)
        prec_macro = Precision(output_transform=cls_ot, average=True)
        prec_vec = Precision(output_transform=cls_ot, average=False)
        prec_pos = SelectIndex(prec_vec, positive_index)

        rec_macro = Recall(output_transform=cls_ot, average=True)
        rec_vec = Recall(output_transform=cls_ot, average=False)
        rec_pos = SelectIndex(rec_vec, positive_index)
        auc = ROC_AUC(output_transform=auc_ot)
        # ConfusionMatrix must receive a callable output_transform, not a call
        cm_ot = partial(
            cls_confmat_output_transform_thresholded,
            decision=cls_decision,
            threshold=cls_threshold,
            positive_index=positive_index,
        )
        cm = ConfusionMatrix(num_classes=int(num_classes), output_transform=cm_ot)
        # metrics.update({"acc": acc, "prec": prec, "recall": rec, "auc": auc, "cls_confmat": cm})
        metrics.update({
            "acc": acc,
            "precision_macro": prec_macro,
            "precision_pos": prec_pos,
            "recall_macro": rec_macro,
            "recall_pos": rec_pos,
            "auc": auc,
            "cls_confmat": cm,
        })
        if int(num_classes) == 2:
            metrics.update(make_cm_rates(cm))

    # segmentation
    if "segmentation" in tasks or multitask:
        # ConfusionMatrix over seg classes
        _seg_cm_ot = seg_cm_ot or partial(seg_confmat_output_transform_default, threshold=float(seg_threshold))
        seg_cm = ConfusionMatrix(num_classes=int(seg_num_classes or num_classes), output_transform=_seg_cm_ot)
        dice = DiceCoefficient(cm=seg_cm)
        iou = JaccardIndex(cm=seg_cm)

        metrics.update({"dice": dice, "iou": iou, "seg_confmat": seg_cm})

    # loss metrics wiring

    # Combined score (if both present)
    if ("classification" in tasks and "segmentation" in tasks or multitask) and "auc" in metrics and "dice" in metrics:
        eps = 1e-8
        w = 0.65  # keep default; or pull from cfg where we call this

        def _multi_score(auc, dice, _w=w, _eps=eps):
            return 1.0 / (_w / (auc + _eps) + (1.0 - _w) / (dice + _eps))
        metrics["multi"] = MetricsLambda(_multi_score, metrics["auc"], metrics["dice"])

    return metrics


__all__ = [
    "cls_output_transform",
    "make_cls_val_metrics",
    "make_metrics",
    "extract_cls_logits_from_any",
    "extract_seg_logits_from_any",
    "positive_score_from_logits",
    "attach_classification_metrics",
    "coerce_loss_dict"
]
