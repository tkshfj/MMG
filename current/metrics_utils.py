# metrics_utils.py
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple, Union
from dataclasses import dataclass
from functools import partial
import torch
import numpy as np
from monai.data.meta_tensor import MetaTensor
from ignite.metrics import Accuracy, Precision, Recall, ConfusionMatrix, ROC_AUC, MetricsLambda, Metric, EpochMetric
from utils.safe import to_tensor, to_float_scalar, to_py
from utils.posprob import PosProbCfg

LESION_IDX = 0
BG_IDX = 1


# Helpers
def get_threshold_from_state(engine) -> float:
    # Single source of truth for threshold
    thr = getattr(engine.state, "threshold", None)
    return float(thr if thr is not None else 0.5)


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
    def __init__(self, output_transform=None, device: "str | torch.device" = "cpu"):
        if output_transform is None:
            def _identity(x):
                return x
            output_transform = _identity
        # super().__init__(output_transform=output_transform, device=device)
        # Ignite expects a concrete device; avoid passing None.
        if device is None:
            device = "cpu"
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
def attach_classification_metrics(
    evaluator,
    cfg: Mapping[str, Any],
    *,
    thr_getter: Callable[[], float] | None = None,
    ppcfg: PosProbCfg | None = None,
) -> None:
    """
    Attach standard classification metrics, all routed through a single PosProbCfg.
    If ppcfg is None, we pull it from evaluator.state.ppcfg (set in engine_utils).
    """
    if thr_getter is None:
        thr_getter = lambda: get_threshold_from_state(evaluator)  # noqa: E731

    n_cls = int(cfg.get("num_classes", 2))
    pos_idx = int(cfg.get("positive_index", 1))
    pp = ppcfg or getattr(getattr(evaluator, "state", object()), "ppcfg", None) or PosProbCfg(positive_index=pos_idx)

    metrics = make_cls_val_metrics(
        num_classes=n_cls,
        ppcfg=pp,
        positive_index=pos_idx,
        cls_decision="threshold",
        cls_threshold=thr_getter,
        device=getattr(getattr(evaluator, "state", object()), "device", torch.device("cpu")),
    )

    # Attach with conventional names
    for k, m in metrics.items():
        m.attach(evaluator, k)

    # Threshold-free distribution debug (prob stats)
    auc_ot = cls_proba_output_transform(ppcfg=pp, positive_index=pos_idx, num_classes=n_cls)
    PosProbStats(output_transform=auc_ot, device="cpu").attach(evaluator, "posprob_stats")


# Helpers
def _first_not_none(m: Mapping[str, Any], keys: tuple[str, ...]):
    """Return the first present key whose value is not None."""
    for k in keys:
        if k in m:
            v = m[k]
            if v is not None:
                return v
    return None


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


def _extract_logits_and_labels(output: Any, ppcfg: PosProbCfg) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accept dict/tuple outputs and return (logits, labels[N]).
    """
    # Get logits container
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        y_pred, y_any = output[0], output[1]
    elif isinstance(output, Mapping):
        y_pred = output
        # prefer top-level 'y', else fallbacks inside 'label'
        y_any = output.get("y", None)
        if y_any is None:
            lab = output.get("label", None)
            if isinstance(lab, Mapping):
                y_any = lab.get("label") or lab.get("y") or lab.get("target")
            else:
                y_any = lab
        if y_any is None:
            # last resorts
            y_any = output.get("target", output.get("labels"))
    else:
        # best-effort (not typical)
        y_pred, y_any = output, output

    logits = ppcfg.pick_logits(y_pred).float()

    if y_any is None:
        raise KeyError("No labels found in output: expected keys like 'y' or 'label{label|y|target}'")
    labels = ppcfg.pick_labels(y_any).view(-1).long()
    return logits, labels


def _cls_probs_and_labels_from_engine_output(output: Any, ppcfg: PosProbCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Always:
      - extract logits & labels once
      - map to probabilities via SoT:
          * binary: [N] pos-prob using ppcfg.logits_to_pos_prob
          * multiclass (C>2): [N,C] softmax
    """
    logits, labels = _extract_logits_and_labels(output, ppcfg)
    if logits.ndim >= 2 and logits.shape[-1] > 2:
        probs = torch.softmax(logits.float(), dim=-1)              # [N,C]
    else:
        probs = ppcfg.logits_to_pos_prob(logits).reshape(-1)       # [N]
    return probs, labels


def _cls_cm_output_transform(
    output: Any,
    *,
    ppcfg: PosProbCfg,
    threshold: float | callable,
    positive_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    - Binary: threshold pos-prob → return 2-col one-hot [N,2]
    - Multiclass: return softmax probs [N,C] (CM/Accuracy will argmax)
    """
    logits, labels = _extract_logits_and_labels(output, ppcfg)
    if logits.ndim >= 2 and logits.shape[-1] > 2:
        probs = torch.softmax(logits.float(), dim=-1)              # [N,C]
        return probs, labels
    thr = float(threshold()) if callable(threshold) else float(threshold)
    p_pos = ppcfg.logits_to_pos_prob(logits).view(-1)
    yhat = (p_pos >= thr).long()
    two_col = torch.stack([1 - yhat, yhat], dim=1)                 # [N,2]
    return two_col, labels


def make_cls_val_metrics(
    *,
    num_classes: int,
    ppcfg: PosProbCfg,
    positive_index: int = 1,
    cls_decision: str = "threshold",
    cls_threshold: float | callable = 0.5,
    device: Optional[torch.device] = None,
) -> dict[str, Metric]:
    dev = device or torch.device("cpu")
    auc = ROC_AUC(
        output_transform=lambda out: _cls_probs_and_labels_from_engine_output(out, ppcfg),
        device=dev,
    )
    cm = ConfusionMatrix(
        num_classes=int(num_classes),
        output_transform=lambda out: _cls_cm_output_transform(
            out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)
        ),
        device=dev,
    )

    def _tpr(m):
        tp, fn = m[1, 1], m[1, 0]
        return tp / (tp + fn + 1e-9)

    def _tnr(m):
        tn, fp = m[0, 0], m[0, 1]
        return tn / (tn + fp + 1e-9)

    bal_acc = MetricsLambda(lambda a, b: 0.5 * (a + b), MetricsLambda(_tpr, cm), MetricsLambda(_tnr, cm))
    acc = Accuracy(output_transform=lambda out: _cls_cm_output_transform(
        out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)
    ), device=dev)
    prec_macro = Precision(average=True, output_transform=lambda out: _cls_cm_output_transform(
        out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)
    ), device=dev)
    rec_macro = Recall(average=True, output_transform=lambda out: _cls_cm_output_transform(
        out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)
    ), device=dev)
    prec_vec = Precision(average=False, output_transform=lambda out: _cls_cm_output_transform(
        out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)
    ), device=dev)
    rec_vec = Recall(average=False, output_transform=lambda out: _cls_cm_output_transform(
        out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)
    ), device=dev)

    return {
        "auc": auc,
        "acc": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "precision_pos": MetricsLambda(lambda v: v[int(positive_index)], prec_vec),
        "recall_pos": MetricsLambda(lambda v: v[int(positive_index)], rec_vec),
        "cls_confmat": cm,
        "bal_acc": bal_acc,
    }


def cls_output_transform(*, ppcfg: PosProbCfg) -> Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]:
    """Factory: returns callable that yields (logits, labels)."""
    def _ot(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        return _extract_logits_and_labels(output, ppcfg)
    return _ot


def seg_labels_from_logits(output):
    """
    Returns hard labels for metrics (no thresholds for 2-ch).
    output = (y_pred_logits, y_true) with shapes:
      logits: [B, C, H, W], y_true: [B, H, W]
    """
    y_pred_logits, y_true = output
    if y_pred_logits.ndim != 4:
        raise ValueError(f"Expected logits [B,C,H,W], got {tuple(y_pred_logits.shape)}")
    C = y_pred_logits.shape[1]
    if C == 2:
        y_hat = y_pred_logits.argmax(dim=1)  # [B,H,W] in {0,1}
    elif C == 1:
        # Single-logit (rare for setup): threshold at 0.5
        y_hat = (y_pred_logits.sigmoid().squeeze(1) >= 0.5).long()
    else:
        y_hat = y_pred_logits.argmax(dim=1)
    return y_hat, y_true


def seg_binary_lesion(output):
    """
    Map labels to binary: lesion=1, background=0 for CM-based metrics.
    """
    y_hat, y_true = seg_labels_from_logits(output)
    y_hat_bin = (y_hat == LESION_IDX).long()
    y_true_bin = (y_true == LESION_IDX).long()
    return y_hat_bin, y_true_bin


def make_seg_lesion_metrics() -> dict[str, MetricsLambda]:
    """
    Dice/IoU for lesion only, derived from 2x2 ConfusionMatrix.
    """
    cm = ConfusionMatrix(num_classes=2, output_transform=seg_binary_lesion)

    def _dice_from_cm(m):
        TP = m[1, 1].to(torch.float32)
        FP = m[0, 1].to(torch.float32)
        FN = m[1, 0].to(torch.float32)
        return (2 * TP) / (2 * TP + FP + FN + 1e-8)

    def _iou_from_cm(m):
        TP = m[1, 1].to(torch.float32)
        FP = m[0, 1].to(torch.float32)
        FN = m[1, 0].to(torch.float32)
        return TP / (TP + FP + FN + 1e-8)

    dice_lesion = MetricsLambda(_dice_from_cm, cm)
    iou_lesion = MetricsLambda(_iou_from_cm, cm)

    return {
        "dice_lesion": dice_lesion,
        "iou_lesion": iou_lesion,
        # Optional: background metrics (for sanity only)
        "dice_bg": MetricsLambda(lambda m: (2 * m[0, 0]) / (2 * m[0, 0] + m[1, 0] + m[0, 1] + 1e-8), cm),
        "iou_bg": MetricsLambda(lambda m: m[0, 0] / (m[0, 0] + m[1, 0] + m[0, 1] + 1e-8), cm),
        "seg_confmat": cm,
    }


# Standard classification output transforms
@dataclass(frozen=True)
class ClsOTs:
    """Pack of classification output transforms."""
    base: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]  # (prob_pos, y)
    thresholded: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]  # (y_hat, y)
    cm: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]  # ([B,C] scores, y)


@dataclass(frozen=True)
class SegOTs:
    """
    Pack of segmentation output transforms.
    - index: returns discrete indices (y_hat_idx[B,H,W], y_idx[B,H,W])
    - cm:    returns scores for ConfusionMatrix (probs[B,K,H,W] or 2-col one-hot, y_idx)
    """
    index: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]
    cm: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]


def make_seg_output_transform(
    *,
    threshold: float | Callable[[], float] = 0.5,
) -> SegOTs:
    """
    Factory to build both segmentation OTs, with dynamic or fixed threshold support.
    - For multi-class (C>1): index OT does softmax→argmax; CM OT returns softmax probs.
    - For binary (C==1):     index OT uses sigmoid≥thr; CM OT returns 2-col one-hot scores.
    """
    def _thr() -> float:
        return float(threshold()) if callable(threshold) else float(threshold)

    def _index_ot(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        # indices for metrics expecting discrete labels
        return seg_confmat_output_transform_default(output, threshold=_thr())

    def _cm_ot(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        # scores for ConfusionMatrix (it will argmax internally)
        return seg_confmat_output_transform(output, threshold=_thr())

    return SegOTs(index=_index_ot, cm=_cm_ot)


def cls_decision_output_transform(
    *,
    ppcfg: PosProbCfg,
    decision: str = "threshold",     # "threshold" or "argmax"
    threshold: Union[float, Callable[[], float]] = 0.5,
    positive_index: int = 1,
) -> Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]:
    dec = decision.lower()

    def _ot(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, y = _extract_logits_and_labels(output, ppcfg)
        if dec == "argmax" or (logits.ndim >= 2 and logits.shape[-1] > 2):
            y_hat = logits.argmax(dim=-1).long()
        else:
            p_pos = ppcfg.logits_to_pos_prob(logits).view(-1)
            thr = (threshold() if callable(threshold) else float(threshold))
            y_hat = (p_pos >= thr).long()
        return y_hat.view(-1), y.view(-1)
    return _ot


def make_default_cls_output_transforms(
    *,
    ppcfg: PosProbCfg,
    num_classes: int,
    decision: str = "threshold",
    threshold: Union[float, Callable[[], float]] = 0.5,
    positive_index: int = 1,
) -> ClsOTs:
    base = cls_proba_output_transform(ppcfg=ppcfg, num_classes=num_classes, positive_index=positive_index)

    def thresholded(output):
        probs, y_true = base(output)
        thr = (threshold() if callable(threshold) else float(threshold))
        if probs.ndim == 1:
            y_hat = (probs >= thr).long()
        else:
            y_hat = probs.argmax(dim=1).long()
        return y_hat.view(-1), y_true.view(-1)

    cm = cls_confmat_output_transform_thresholded(
        ppcfg=ppcfg,
        decision=decision,
        threshold=threshold,
        positive_index=positive_index,
        num_classes=num_classes,
    )
    return ClsOTs(base=base, thresholded=thresholded, cm=cm)


def make_std_cls_metrics_with_cal_thr(
    num_classes: int,
    *,
    decision: str = "threshold",
    positive_index: int = 1,
    thr_getter: Callable[[], float] | float = 0.5,
    ppcfg: PosProbCfg | None = None,
    score_provider: PosProbCfg | None = None,
    device: torch.device | None = None,
    **kwargs,
):
    if "pos_index" in kwargs:   # back-compat
        positive_index = int(kwargs.pop("pos_index"))
    # Resolve a single PosProbCfg source-of-truth
    pp = ppcfg or score_provider or PosProbCfg(positive_index=int(positive_index))
    decision_ot = cls_decision_output_transform(ppcfg=pp, decision=decision, threshold=thr_getter, positive_index=positive_index)
    proba_ot = cls_proba_output_transform(ppcfg=pp, num_classes=num_classes, positive_index=positive_index)
    cm_ot_cal = cls_confmat_output_transform_thresholded(ppcfg=pp, decision=decision, threshold=thr_getter, positive_index=positive_index, num_classes=num_classes)  # noqa: E501

    acc = Accuracy(output_transform=decision_ot)  # -> (int01, int01)
    prec_macro = Precision(average=True, output_transform=decision_ot, is_multilabel=False)
    prec_vec = Precision(average=False, output_transform=decision_ot, is_multilabel=False)
    rec_macro = Recall(average=True, output_transform=decision_ot, is_multilabel=False)
    rec_vec = Recall(average=False, output_transform=decision_ot, is_multilabel=False)

    cm = ConfusionMatrix(num_classes=int(num_classes), output_transform=cm_ot_cal)
    metrics: dict[str, Metric] = {
        "acc": acc,
        "precision_macro": prec_macro,
        "prec": prec_vec,  # expose full vector
        "recall_macro": rec_macro,
        "recall": rec_vec,  # expose full vector
        "cls_confmat": cm,
    }

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


def cls_proba_output_transform(
    *,
    ppcfg: PosProbCfg,
    num_classes: int,
    positive_index: int = 1,
) -> Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]:
    return lambda output: _cls_probs_and_labels_from_engine_output(output, ppcfg)


def cls_confmat_output_transform_thresholded(
    *,
    ppcfg: PosProbCfg,
    decision: str = "threshold",             # same semantics as above
    threshold: Union[float, Callable[[], float]] = 0.5,
    positive_index: int = 1,
    num_classes: int = 2,
) -> Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Factory: ConfusionMatrix OT.
    - Multiclass: returns probs [N,C] (CM will argmax internally).
    - Binary: thresholds to indices and returns 2-col one-hot scores [N,2].
    """
    dec = decision.lower()

    def _ot(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, y = _extract_logits_and_labels(output, ppcfg)
        # Multiclass path or explicit argmax -> provide [N,C] scores
        if dec == "argmax" or (logits.ndim >= 2 and logits.shape[-1] > 2) or int(num_classes) > 2:
            probs = torch.softmax(logits, dim=-1)
            return probs, y
        # Binary path -> apply threshold, then emit 2-col one-hot predictions
        p_pos = ppcfg.logits_to_pos_prob(logits).view(-1)
        thr = (threshold() if callable(threshold) else float(threshold))
        yhat = (p_pos >= thr).long()
        two_col = torch.stack([1 - yhat, yhat], dim=1)  # [N,2]
        return two_col, y
    return _ot


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


def seg_output_transform(output, *, threshold: float = 0.5):
    """
    Output transform for segmentation metrics.

    - Accepts model outputs in several container shapes and names.
    - Returns (y_pred, y) where:
        * If C == 2: y_pred = argmax over channel dim (no thresholding)
        * If C == 1: y_pred = (sigmoid >= threshold)
    - Targets are coerced to int64 class indices [B,H,W] (handles one-hot masks).
    """
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

    # fetch tensors safely (no boolean 'or' on tensors)
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

    if y.ndim == 2:                  # [H, W]
        y = y.unsqueeze(0)           # [1,H,W]
    elif y.ndim == 4:                # [B,C,H,W] (one-hot)
        y = y.argmax(dim=1) if y.shape[1] > 1 else y[:, 0]
    elif y.ndim != 3:                # [B,H,W]
        raise ValueError(f"seg_output_transform: unsupported target shape {tuple(y.shape)}")
    y = y.long()

    C = int(pred.shape[1])

    # IMPORTANT:
    #   * For 2-channel logits, use argmax (class 0=lesion, class 1=background)
    #   * Threshold ONLY applies to single-logit binary heads
    if C == 1:
        y_pred = (pred.sigmoid() >= float(threshold)).long().squeeze(1)  # [B,H,W]
    else:
        y_pred = pred.argmax(dim=1).long()  # [B,H,W], no threshold, lesion=0

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


# Per-class metrics from a ConfusionMatrix
def _dice_vec_from_cm(cm: ConfusionMatrix) -> MetricsLambda:
    def _dice_per_class(m: torch.Tensor) -> torch.Tensor:
        # m: [K, K], rows=true, cols=pred
        m = m.to(dtype=torch.float32)
        tp = torch.diag(m)
        fp = m.sum(dim=0) - tp
        fn = m.sum(dim=1) - tp
        eps = 1e-8
        return (2.0 * tp) / (2.0 * tp + fp + fn + eps)  # [K]
    return MetricsLambda(_dice_per_class, cm)


def _iou_vec_from_cm(cm: ConfusionMatrix) -> MetricsLambda:
    def _iou_per_class(m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=torch.float32)
        tp = torch.diag(m)
        fp = m.sum(dim=0) - tp
        fn = m.sum(dim=1) - tp
        eps = 1e-8
        return tp / (tp + fp + fn + eps)  # [K]
    return MetricsLambda(_iou_per_class, cm)


def _mean_excluding_index() -> Callable[[torch.Tensor, int], torch.Tensor]:
    # Returns a lambda usable by MetricsLambda to mean over classes except ignore_index
    def _mean_idx(v: torch.Tensor, idx: int) -> torch.Tensor:
        if v.numel() == 0:
            return torch.tensor(0.0, dtype=v.dtype, device=v.device)
        mask = torch.ones_like(v, dtype=torch.bool)
        if 0 <= int(idx) < v.numel():
            mask[int(idx)] = False
        denom = mask.sum().clamp_min(1).to(v.dtype)
        return (v[mask].sum() / denom)
    return _mean_idx


def make_seg_val_metrics(
    *,
    num_classes: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Metric]:
    # Confusion matrix over pixels
    _ot = partial(seg_confmat_output_transform, threshold=threshold)
    cm_kwargs = dict(num_classes=int(num_classes), output_transform=_ot)
    if device is not None:
        cm_kwargs["device"] = device
    cm = ConfusionMatrix(**cm_kwargs)

    # Per-class vectors
    dice_vec = _dice_vec_from_cm(cm)  # [K]
    iou_vec = _iou_vec_from_cm(cm)    # [K]

    # Macro (respect ignore_index if provided)
    if ignore_index is None:
        dice_macro = MetricsLambda(lambda v: v.mean(), dice_vec)
        iou_macro = MetricsLambda(lambda v: v.mean(), iou_vec)
    else:
        mean_ex = _mean_excluding_index()
        dice_macro = MetricsLambda(mean_ex, dice_vec, int(ignore_index))
        iou_macro = MetricsLambda(mean_ex, iou_vec, int(ignore_index))

    metrics: dict[str, Metric] = {
        "seg_confmat": cm,
        "dice": dice_macro,
        "iou": iou_macro,
    }

    # Per-class metrics (ensures dice_1 exists for schedulers/sweeps)
    K = int(num_classes)
    for k in range(K):
        metrics[f"dice_{k}"] = MetricsLambda(lambda v, k=k: v[k], dice_vec)
        metrics[f"iou_{k}"] = MetricsLambda(lambda v, k=k: v[k], iou_vec)

    return metrics


def make_metrics(
    *,
    tasks,
    num_classes: int,
    cls_decision: str = "threshold",
    cls_threshold=0.5,  # float or callable
    positive_index: int = 1,
    seg_num_classes: int | None = None,
    seg_threshold: float = 0.5,   # Ignored for 2-ch seg (we argmax)
    loss_fn=None,
    cls_ot=None,   # thresholded (y_hat, y)
    auc_ot=None,   # base (prob, y)
    seg_cm_ot=None,  # Deprecated for 2-ch path; kept for API compat
    multitask: bool = False,
    ppcfg=None,  # PosProbCfg | None
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Returns a dict[name -> Metric] with BARE names; caller can prefix 'val/' at log time.

    Changes:
    - Segmentation now exposes lesion-only metrics: dice_lesion / iou_lesion
      (background ignored). We alias 'dice' -> 'dice_lesion' for backward compat.
    - For 2-channel logits, predictions use argmax over channels (no sigmoid/threshold).
    """
    dev = device or torch.device("cpu")
    # Guard & defaults
    if ppcfg is None:
        # Lazy import if we keep PosProbCfg elsewhere
        try:
            from posprob import PosProbCfg
            ppcfg = PosProbCfg(positive_index=int(positive_index))
        except Exception:
            ppcfg = type("PP", (), {"positive_index": int(positive_index)})()

    tasks = set(tasks or [])
    has_cls = ("classification" in tasks) or bool(multitask)
    has_seg = ("segmentation" in tasks) or bool(multitask)

    metrics: dict[str, Any] = {}

    # Classification metrics
    if has_cls:
        if cls_ot is None or auc_ot is None:
            # Build default output transforms
            from .metrics_utils_cls import make_default_cls_output_transforms, cls_confmat_output_transform_thresholded
            ots = make_default_cls_output_transforms(
                ppcfg=ppcfg, num_classes=num_classes,
                decision=cls_decision, threshold=cls_threshold, positive_index=positive_index,
            )
            cls_ot = ots.thresholded
            auc_ot = ots.base
        else:
            # Also need CM OT if caller provided custom OTs
            from .metrics_utils_cls import cls_confmat_output_transform_thresholded

        acc = Accuracy(output_transform=cls_ot, device=dev)
        prec_vec = Precision(average=False, output_transform=cls_ot, is_multilabel=False, device=dev)
        rec_vec = Recall(average=False, output_transform=cls_ot, is_multilabel=False, device=dev)
        prec_mac = Precision(average=True, output_transform=cls_ot, is_multilabel=False, device=dev)
        rec_mac = Recall(average=True, output_transform=cls_ot, is_multilabel=False, device=dev)
        auc = ROC_AUC(output_transform=auc_ot, device=dev)

        cm_ot = cls_confmat_output_transform_thresholded(
            ppcfg=ppcfg, decision=cls_decision, threshold=cls_threshold,
            positive_index=positive_index, num_classes=num_classes
        )
        cm = ConfusionMatrix(num_classes=int(num_classes), output_transform=cm_ot, device=dev)

        metrics.update({
            "acc": acc,
            "precision_macro": prec_mac,
            "recall_macro": rec_mac,
            "precision_pos": MetricsLambda(lambda v: v[int(positive_index)], prec_vec),
            "recall_pos": MetricsLambda(lambda v: v[int(positive_index)], rec_vec),
            "auc": auc,
            "cls_confmat": cm,
        })

        if int(num_classes) == 2:
            # If we have existing helper:
            try:
                from .metrics_utils_cls import make_cm_rates
                metrics.update(make_cm_rates(cm))
            except Exception:
                pass  # optional

    # Segmentation metrics (lesion-only)
    if has_seg:
        # New: background ignored; lesion is class 0
        # This internally builds a 2x2 CM on (lesion vs background) via seg_binary_lesion.
        seg_metrics = make_seg_lesion_metrics(device=dev)
        metrics.update(seg_metrics)

        # Backward-compat: keep 'dice'/'iou' keys pointing to lesion metrics unless already set.
        if "dice_lesion" in seg_metrics and "dice" not in metrics:
            metrics["dice"] = seg_metrics["dice_lesion"]
        if "iou_lesion" in seg_metrics and "iou" not in metrics:
            metrics["iou"] = seg_metrics["iou_lesion"]

        # Note: we intentionally ignore seg_threshold for 2-ch logits (we use argmax).
        # If we have truly single-logit binary seg, handle it in seg_labels_from_logits().

    # Optional: combined multi-task headline
    if has_cls and has_seg and ("auc" in metrics) and ("dice_lesion" in metrics):
        eps, w = 1e-8, 0.65
        metrics["multi"] = MetricsLambda(
            lambda a, d, _w=w, _e=eps: 1.0 / (_w / (a + _e) + (1 - _w) / (d + _e)),
            metrics["auc"], metrics["dice_lesion"]
        )

    return metrics


__all__ = [
    "cls_output_transform",
    "cls_proba_output_transform",
    "cls_decision_output_transform",
    "cls_confmat_output_transform_thresholded",
    "make_cls_val_metrics",
    "make_metrics",
    "extract_cls_logits_from_any",
    "extract_seg_logits_from_any",
    "attach_classification_metrics",
    "coerce_loss_dict",
    "SegOTs",
    "make_seg_output_transform",
    "seg_confmat_output_transform_default",
    "seg_confmat_output_transform",
]

# def make_metrics(
#     *,
#     tasks,
#     num_classes: int,
#     cls_decision: str = "threshold",
#     cls_threshold=0.5,  # float or callable
#     positive_index: int = 1,
#     seg_num_classes: int | None = None,
#     seg_threshold: float = 0.5,
#     loss_fn=None,
#     cls_ot=None,   # thresholded (y_hat, y)
#     auc_ot=None,   # base (prob, y)
#     seg_cm_ot=None,
#     multitask: bool = False,
#     ppcfg: PosProbCfg | None = None,
#     device: torch.device | None = None,
# ) -> dict[str, Any]:
#     dev = device or torch.device("cpu")
#     pp = ppcfg or PosProbCfg(positive_index=int(positive_index))
#     metrics: dict[str, Any] = {}

#     if cls_ot is None or auc_ot is None:
#         ots = make_default_cls_output_transforms(
#             ppcfg=pp, num_classes=num_classes,
#             decision=cls_decision, threshold=cls_threshold, positive_index=positive_index,
#         )
#         cls_ot = ots.thresholded
#         auc_ot = ots.base

#     if "classification" in tasks or multitask:
#         acc = Accuracy(output_transform=cls_ot, device=dev)
#         prec_vec = Precision(average=False, output_transform=cls_ot, is_multilabel=False, device=dev)
#         rec_vec = Recall(average=False, output_transform=cls_ot, is_multilabel=False, device=dev)
#         prec_mac = Precision(average=True, output_transform=cls_ot, is_multilabel=False, device=dev)
#         rec_mac = Recall(average=True, output_transform=cls_ot, is_multilabel=False, device=dev)
#         auc = ROC_AUC(output_transform=auc_ot, device=dev)

#         cm_ot = cls_confmat_output_transform_thresholded(
#             ppcfg=pp, decision=cls_decision, threshold=cls_threshold,
#             positive_index=positive_index, num_classes=num_classes
#         )
#         cm = ConfusionMatrix(num_classes=int(num_classes), output_transform=cm_ot, device=dev)

#         metrics.update({
#             "acc": acc,
#             "precision_macro": prec_mac,
#             "recall_macro": rec_mac,
#             "precision_pos": MetricsLambda(lambda v: v[int(positive_index)], prec_vec),
#             "recall_pos": MetricsLambda(lambda v: v[int(positive_index)], rec_vec),
#             "auc": auc,
#             "cls_confmat": cm,
#         })
#         if int(num_classes) == 2:
#             metrics.update(make_cm_rates(cm))

#     if "segmentation" in tasks or multitask:
#         _seg_cm_ot = seg_cm_ot or partial(seg_confmat_output_transform_default, threshold=float(seg_threshold))
#         seg_cm = ConfusionMatrix(num_classes=int(seg_num_classes or num_classes), output_transform=_seg_cm_ot, device=dev)
#         dice = DiceCoefficient(cm=seg_cm)
#         iou = JaccardIndex(cm=seg_cm)
#         metrics.update({"dice": dice, "iou": iou, "seg_confmat": seg_cm})

#     if ("classification" in tasks and "segmentation" in tasks or multitask) and "auc" in metrics and "dice" in metrics:
#         eps, w = 1e-8, 0.65
#         metrics["multi"] = MetricsLambda(lambda a, d, _w=w, _e=eps: 1.0 / (_w / (a + _e) + (1 - _w) / (d + _e)), metrics["auc"], metrics["dice"])
#     return metrics

# --- ensure these exist in this file (shown earlier) ---
# seg_labels_from_logits(output) -> (y_hat, y_true)
# seg_binary_lesion(output) -> (y_hat_bin, y_true_bin)
# make_seg_lesion_metrics(device: torch.device | None = None) -> dict[str, Metric]

# def _resolve_threshold(thr: Union[float, Callable[[], float]]) -> float:
#     return float(thr()) if callable(thr) else float(thr)


# def attach_classification_metrics(
#         evaluator, cfg: Mapping[str, Any], *,
#         thr_getter: Callable[[], float] | None = None,
#         score_provider: PosProbCfg | None = None,
#         ppcfg: PosProbCfg | None
# ) -> None:
#     if thr_getter is None:
#         def thr_getter():
#             # fallback if caller didn't pass the two-pass box
#             # return float(getattr(evaluator.state, "threshold", cfg.get("cls_threshold", 0.5)))
#             return get_threshold_from_state(evaluator)

#     n_cls = int(cfg.get("num_classes", 2))
#     pos_idx = int(cfg.get("positive_index", 1))
#     pp = ppcfg or score_provider or PosProbCfg(positive_index=pos_idx)

#     metrics = make_cls_val_metrics(
#         num_classes=n_cls,
#         decision="threshold",
#         threshold=thr_getter,
#         positive_index=pos_idx,
#         ppcfg=pp,
#         score_provider=score_provider,
#     )
#     metrics["auc"].attach(evaluator, "auc")
#     metrics["cls_confmat"].attach(evaluator, "cls_confmat")
#     if "bal_acc" in metrics:
#         metrics["bal_acc"].attach(evaluator, "bal_acc")
#     if n_cls == 2:
#         # attach predicted positive rate, derived from the same CM → same threshold
#         from metrics_utils import make_cm_rates
#         rates = make_cm_rates(metrics["cls_confmat"])
#         rates["pos_rate"].attach(evaluator, "pos_rate")
#         # metrics["gt_pos_rate"].attach(evaluator, "val/gt_pos_rate")
#     # helpful, threshold-free distribution debug
#     # from functools import partial
#     # auc_ot = partial(cls_proba_output_transform, positive_index=pos_idx, num_classes=n_cls)
#     auc_ot = cls_proba_output_transform(ppcfg=pp, positive_index=pos_idx, num_classes=n_cls)
#     # PosProbStats(output_transform=auc_ot).attach(evaluator, "val/posprob_stats")
#     PosProbStats(output_transform=auc_ot, device="cpu").attach(evaluator, "posprob_stats")


# def _cls_probs_and_labels_from_engine_output(output: Any, ppcfg: PosProbCfg) -> tuple[torch.Tensor, torch.Tensor]:
#     logits, labels = _extract_logits_and_labels(output, ppcfg)
#     if logits.ndim >= 2 and logits.shape[-1] > 2:
#         probs = torch.softmax(logits.float(), dim=-1)        # [N,C]
#     else:
#         probs = ppcfg.logits_to_pos_prob(logits).reshape(-1)  # [N]
#     return probs, labels


# def _cls_cm_output_transform(output: Any, *, ppcfg: PosProbCfg, threshold: float | callable, positive_index: int) -> tuple[torch.Tensor, torch.Tensor]:
#     probs, labels = _cls_probs_and_labels_from_engine_output(output, ppcfg)
#     thr = float(threshold()) if callable(threshold) else float(threshold)
#     if probs.ndim == 1:
#         # build 2-column probs so CM/Acc/Prec/Rec work
#         yhat = (probs >= thr).long()
#         two_col = torch.stack([1 - yhat, yhat], dim=1)
#         return two_col, labels
#     else:
#         # multi-class → argmax
#         yhat = probs.argmax(dim=1)
#         two_col = torch.nn.functional.one_hot(yhat, num_classes=probs.shape[1]).to(probs.dtype)
#         return two_col, labels


# def make_cls_val_metrics(
#     *,
#     num_classes: int,
#     ppcfg: PosProbCfg,
#     positive_index: int = 1,
#     cls_decision: str = "threshold",
#     cls_threshold: float | callable = 0.5,
#     device: Optional[torch.device] = None,
# ) -> dict[str, Metric]:
#     from ignite.metrics import Accuracy, Precision, Recall, ROC_AUC, ConfusionMatrix, MetricsLambda
#     dev = device or torch.device("cpu")
#     # AUC uses raw probabilities (binary [N] or multi [N,C])
#     auc = ROC_AUC(
#         output_transform=lambda out: _cls_probs_and_labels_from_engine_output(out, ppcfg),
#         device=dev,
#     )
#     cm = ConfusionMatrix(
#         num_classes=int(num_classes),
#         output_transform=lambda out: _cls_cm_output_transform(out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)),
#         device=dev,
#     )

#     def _tpr(m):
#         tp, fn = m[1, 1], m[1, 0]
#         return tp / (tp + fn + 1e-9)

#     def _tnr(m):
#         tn, fp = m[0, 0], m[0, 1]
#         return tn / (tn + fp + 1e-9)

#     bal_acc = MetricsLambda(
#         lambda a, b: 0.5 * (a + b),
#         MetricsLambda(_tpr, cm),
#         MetricsLambda(_tnr, cm),
#     )
#     acc = Accuracy(
#         output_transform=lambda out: _cls_cm_output_transform(out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)),
#         device=dev)
#     prec_macro = Precision(
#         average=True,
#         output_transform=lambda out: _cls_cm_output_transform(out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)),
#         device=dev)
#     rec_macro = Recall(
#         average=True,
#         output_transform=lambda out: _cls_cm_output_transform(out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)),
#         device=dev)
#     prec_vec = Precision(
#         average=False,
#         output_transform=lambda out: _cls_cm_output_transform(out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)),
#         device=dev)
#     rec_vec = Recall(
#         average=False,
#         output_transform=lambda out: _cls_cm_output_transform(out, ppcfg=ppcfg, threshold=cls_threshold, positive_index=int(positive_index)),
#         device=dev)
#     return {
#         "auc": auc,
#         "acc": acc,
#         "precision_macro": prec_macro,
#         "recall_macro": rec_macro,
#         "precision_pos": MetricsLambda(lambda v: v[int(positive_index)], prec_vec),
#         "recall_pos": MetricsLambda(lambda v: v[int(positive_index)], rec_vec),
#         "cls_confmat": cm,
#         "bal_acc": bal_acc,
#     }


# def seg_output_transform(output, *, threshold: float = 0.5,):
#     # parse container
#     if isinstance(output, (list, tuple)) and len(output) == 2:
#         preds, targets = output
#         out = preds if isinstance(preds, Mapping) else {"seg_out": preds}
#         lbl = targets if isinstance(targets, Mapping) else {"mask": targets}
#     elif isinstance(output, Mapping):
#         out = output
#         lbl = out.get("label", {})
#     else:
#         raise ValueError(f"seg_output_transform: unexpected output type {type(output)}")

#     # fetch tensors safely (no `or` on tensors)
#     pred = _first_not_none(out, ("seg_logits", "seg_out", "seg_pred", "pred"))
#     if pred is None:
#         raise ValueError("seg_output_transform: no seg_out/seg_logits/seg_pred/pred in output")

#     mask = _first_not_none(lbl, ("mask", "y", "label")) if isinstance(lbl, Mapping) else None
#     if mask is None:
#         mask = out.get("mask", None)
#     if mask is None:
#         raise ValueError("seg_output_transform: no mask in output/label")

#     pred = to_tensor(pred).float()
#     y = to_tensor(mask)

#     # normalize shapes
#     # Accept [H,W], [B,H,W], [B,1,H,W], [B,C,H,W]
#     if pred.ndim == 2:               # [H, W]
#         pred = pred.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
#     elif pred.ndim == 3:             # [B, H, W]
#         pred = pred.unsqueeze(1)               # [B,1,H,W]
#     elif pred.ndim != 4:
#         raise ValueError("seg_output_transform: pred must be [B,C,H,W]")

#     if y.ndim == 2:
#         y = y.unsqueeze(0)
#     elif y.ndim == 4:
#         y = y.argmax(dim=1) if y.shape[1] > 1 else y[:, 0]
#     elif y.ndim != 3:
#         raise ValueError(f"seg_output_transform: unsupported target shape {tuple(y.shape)}")
#     y = y.long()

#     C = pred.shape[1]
#     if C == 1:
#         y_pred = (pred.sigmoid() >= float(threshold)).long().squeeze(1)
#     else:
#         y_pred = pred.argmax(dim=1).long()

#     return y_pred, y


# def make_seg_val_metrics(*, num_classes: int, threshold: float = 0.5, ignore_index: Optional[int] = None):
#     _ot = partial(seg_confmat_output_transform, threshold=threshold)
#     cm = ConfusionMatrix(num_classes=num_classes, output_transform=_ot)
#     try:
#         dice = DiceCoefficient(cm, ignore_index=ignore_index) if ignore_index is not None else DiceCoefficient(cm)
#     except TypeError:
#         dice = DiceCoefficient(cm)
#     try:
#         iou = JaccardIndex(cm, ignore_index=ignore_index) if ignore_index is not None else JaccardIndex(cm)
#     except TypeError:
#         iou = JaccardIndex(cm)
#     # return {"seg_confmat": cm, "seg_dice": dice, "seg_iou": iou}
#     return {"seg_confmat": cm, "dice": dice, "iou": iou}
