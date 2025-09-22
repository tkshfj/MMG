# metrics_utils.py
import torch
from monai.data.meta_tensor import MetaTensor


# Helpers
def _to_tensor(x):
    """Convert MetaTensor/array/scalar to torch.Tensor; raise if not possible."""
    if isinstance(x, MetaTensor):
        x = x.as_tensor()
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception as e:
        raise TypeError(f"_to_tensor: cannot convert type {type(x)}") from e


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


# Classification transforms
def cls_output_transform(output):
    """ Normalize to (logits[B, C] float, labels[B] long). """
    import torch

    def _debug_once_dict(d, where):
        if not getattr(cls_output_transform, "_debug_dumped", False):
            def walk(k, v, depth=0):
                t = type(v).__name__
                if isinstance(v, dict):
                    print("  " * depth + f"- {k}: dict({list(v.keys())[:8]})")
                    for kk, vv in list(v.items())[:8]:
                        walk(f"{k}.{kk}", vv, depth + 1)
                elif isinstance(v, torch.Tensor):
                    print("  " * depth + f"- {k}: Tensor{tuple(v.shape)}, dtype={v.dtype}")
                else:
                    print("  " * depth + f"- {k}: {t}")
            print(f"[cls_output_transform] dict structure at {where}:")
            for k, v in d.items():
                walk(k, v)
            cls_output_transform._debug_dumped = True

    def _get_nested(d, path):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _to_tensor(x):
        from monai.data.meta_tensor import MetaTensor
        if isinstance(x, MetaTensor):
            x = x.as_tensor()
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def _pick_cls_from_seq(seq):
        # Prefer 2D [B,C] tensors (classification head)
        two_d = [t for t in seq if isinstance(t, torch.Tensor) and t.ndim == 2]
        if two_d:
            return two_d[0]
        # Next: ViT-style [B,T,C] → take token 0 (or mean over T)
        three_d = [t for t in seq if isinstance(t, torch.Tensor) and t.ndim == 3]
        if three_d:
            x = three_d[0]
            return x[:, 0, :] if x.shape[1] >= 1 else x.mean(dim=1)
        # Do NOT pick 4D [B,C,H,W] (likely segmentation); skip it
        return None

    def _extract_logits(x):
        # Tensors direct
        if isinstance(x, torch.Tensor):
            return x.float()

        if isinstance(x, dict):
            # If 'pred' exists, mine it first
            if "pred" in x:
                p = x["pred"]
                # pred is a tensor
                if isinstance(p, torch.Tensor):
                    if p.ndim == 2:  # [B,C]
                        return p.float()
                    if p.ndim == 3:  # [B,T,C]
                        return (p[:, 0, :]).float() if p.shape[1] >= 1 else p.mean(dim=1).float()
                    # 4D is probably segmentation; do not use as cls logits
                # pred is a tuple/list: pick best candidate
                if isinstance(p, (tuple, list)):
                    cand = _pick_cls_from_seq(p)
                    if cand is not None:
                        return cand.float()
                # pred is a dict: try common keys
                if isinstance(p, dict):
                    for k in ("class_logits", "logits", "y_pred", "scores", "output"):
                        if k in p and isinstance(p[k], torch.Tensor):
                            t = p[k]
                            if t.ndim == 2:  # [B,C]
                                return t.float()
                            if t.ndim == 3:  # [B,T,C]
                                return (t[:, 0, :]).float() if t.shape[1] >= 1 else t.mean(dim=1).float()

            # Fall back to common top-level keys (NOT generic tensors like image)
            for keys in (("pred", "class_logits"), ("pred", "logits"), ("class_logits",), ("logits",), ("y_pred",), ("scores",), ("output",)):
                v = _get_nested(x, keys)
                if isinstance(v, torch.Tensor):
                    if v.ndim == 2:
                        return v.float()
                    if v.ndim == 3:
                        return (v[:, 0, :]).float() if v.shape[1] >= 1 else v.mean(dim=1).float()

            # No acceptable logits found
            _debug_once_dict(x, "logits-miss")
            raise KeyError("[cls_output_transform] Could not find classification logits in dict.")

        # Anything else: try tensor coercion (rare)
        return _to_tensor(x).float()

    def _extract_labels(x):
        # tensors -> use directly
        if isinstance(x, torch.Tensor):
            y = x
        elif isinstance(x, dict):
            # common paths
            for keys in (
                ("label", "label"),
                ("label", "y"),
                ("label",),
                ("y",),
                ("target",),
                ("labels",),
            ):
                v = _get_nested(x, keys)
                if v is not None:
                    y = _to_tensor(v)
                    break
            else:
                # accept 'label' dict with alternative keys
                if "label" in x and isinstance(x["label"], dict):
                    for k in ("index", "cls", "target"):
                        if k in x["label"]:
                            y = _to_tensor(x["label"][k])
                            break
                # last resort: first 1D/0D integer-ish tensor
                if "y" not in locals():
                    for k, v in x.items():
                        if isinstance(v, torch.Tensor) and (v.dtype.is_floating_point is False):
                            y = v
                            break
                    else:
                        _debug_once_dict(x, "labels-miss")
                        raise KeyError("[cls_output_transform] Could not find labels in dict.")
        else:
            y = _to_tensor(x)

        # normalize labels -> [B]
        if y.ndim >= 2 and y.shape[-1] > 1:
            y = y.argmax(dim=-1)
        if y.ndim >= 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if y.ndim == 0:
            y = y.view(1)
        return y.long().view(-1)

    def _finalize(logits, labels):
        # Keep batch B; reduce only non-class dims
        if logits.ndim == 3:            # [B,T,C] -> [B,C]
            logits = logits[:, 0, :] if logits.shape[1] >= 1 else logits.mean(dim=1)
        elif logits.ndim >= 4:          # [B,C,H,W,...] -> [B,C]
            dims = tuple(range(2, logits.ndim))
            logits = logits.mean(dim=dims)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if logits.ndim != 2:
            raise ValueError(f"[cls_output_transform] Expected [B,C], got {tuple(logits.shape)}")

        labels = labels.long().view(-1)
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(f"[cls_output_transform] Incompatible shapes after collapse: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}")

        if not getattr(cls_output_transform, "_printed", False):
            print(f"[cls_output_transform] -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
            cls_output_transform._printed = True
        return logits.float(), labels

    # accepted outer shapes
    # list[dict] decollated
    if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], dict):
        preds, labs = [], []
        for s in output:
            preds.append(_extract_logits(s))
            labs.append(_extract_labels(s))
        logits = torch.cat([p.unsqueeze(0) if p.ndim == 1 else p for p in preds], dim=0)
        labels = torch.cat([lab.view(-1) for lab in labs], dim=0)
        return _finalize(logits, labels)

    # ((y_pred, y),)
    if isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], (list, tuple)) and len(output[0]) >= 2:
        return _finalize(_extract_logits(output[0][0]), _extract_labels(output[0][1]))

    # (y_pred, y, *extras)
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        return _finalize(_extract_logits(output[0]), _extract_labels(output[1]))

    # dict (batched)
    if isinstance(output, dict):
        return _finalize(_extract_logits(output), _extract_labels(output))

    # unknown
    tname = type(output).__name__
    prev = str(output)
    if len(prev) > 200:
        prev = prev[:197] + "..."
    raise ValueError(f"[cls_output_transform] Unexpected output structure: type={tname}, preview={prev}")


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


def auc_output_transform(output, positive_index=1):
    """
    AUROC transform:
      - binary:  scores[B], labels[B]
      - multiclass (C>2): probs[B,C], one-hot labels[B,C] (OvR)
    """
    import torch
    import torch.nn.functional as F

    logits, labels = cls_output_transform(output)  # logits [B,C], labels [B]
    if logits.ndim == 1:
        # [B] logits (rare) -> binary
        scores = torch.sigmoid(logits)
        return scores, labels

    B, C = logits.shape
    if C == 1:
        scores = torch.sigmoid(logits.squeeze(-1))         # [B]
        return scores, labels
    if C == 2:
        scores = torch.softmax(logits, dim=-1)[:, positive_index]  # [B]
        return scores, labels

    # C > 2: multiclass -> one-vs-rest
    probs = torch.softmax(logits, dim=-1)                  # [B,C]
    onehot = F.one_hot(labels.to(torch.int64), num_classes=C).to(probs.dtype)  # [B,C]
    return probs, onehot


# Segmentation transforms
def _unpack_engine_output(output):
    """Return (y_pred, y_true) from common Ignite/MONAI outputs."""
    match output:
        case (y_pred, y_true, *_) | [y_pred, y_true, *_]:
            return y_pred, y_true
        case {"y_pred": y_pred, "y": y_true}:
            return y_pred, y_true
        case {"pred": y_pred, "label": y_true}:  # MONAI Supervised* default
            return y_pred, y_true
        case _:
            return output, None


def seg_output_transform(output):
    """Return (seg_logits [B,C,H,W] float, mask [B,H,W] long)."""
    y_pred, y_true = _unpack_engine_output(output)

    def get(d, k):
        return d.get(k) if isinstance(d, dict) else None

    def coalesce(*vals):
        return next((v for v in vals if v is not None), None)

    # logits
    y_pred = coalesce(
        get(y_pred, "seg_out"), get(y_pred, "logits"), get(y_pred, "seg"),
        get(get(output, "pred"), "seg_out"), get(get(output, "pred"), "logits"), get(get(output, "pred"), "seg"),
        get(get(output, "y_pred"), "seg_out"), get(get(output, "y_pred"), "logits"), get(get(output, "y_pred"), "seg"),
        y_pred if not isinstance(y_pred, dict) else None,
    )

    # mask
    y_true = coalesce(
        get(y_true, "mask"), get(y_true, "seg"), get(y_true, "y"),
        get(get(output, "label"), "mask"), get(get(output, "label"), "seg"), get(get(output, "label"), "y"),
        get(output, "mask"),
        y_true if not isinstance(y_true, dict) else None,
    )

    if y_pred is None or y_true is None:
        raise ValueError("seg_output_transform: expected pred['seg_out'] and label['mask'] (or top-level 'mask').")

    seg_logits = _to_tensor(y_pred).float()
    mask = _to_tensor(y_true)

    if seg_logits.ndim == 3:
        seg_logits = seg_logits.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    if mask.ndim == 4:
        mask = mask[:, 0] if mask.shape[1] == 1 else mask.argmax(dim=1)
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)  # [1,H,W] -> [H,W]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # [H,W] -> [1,H,W]

    return seg_logits, mask.long()


def seg_confmat_output_transform(output):
    """
    Ignite ConfusionMatrix transform for segmentation.
    Returns:
        y_pred: [B, C, H, W] probs/logits (ConfusionMatrix will argmax if 4D)
        y:      [B, H, W] integer mask
    """
    seg_logits, mask = seg_output_transform(output)
    # Convert logits to probs for stability
    if seg_logits.shape[1] == 1:
        # seg_logits = torch.stack([-seg_logits[:, 0], seg_logits[:, 0]], dim=1)
        # return seg_logits, mask.long()
        prob_fg = torch.sigmoid(seg_logits)
        prob_bg = 1.0 - prob_fg
        y_pred = torch.cat([prob_bg, prob_fg], dim=1)
    else:
        y_pred = torch.softmax(seg_logits, dim=1)
    return y_pred, mask


def _seg_cm_normalizer(ot):
    """Convert logits to probs (2 channels if binary) and ensure y indices."""
    def f(output):
        y_pred, y = ot(output)
        # y_pred: ensure probs with 2 channels if binary
        if y_pred.ndim == 4:
            if y_pred.shape[1] == 1:
                prob_fg = torch.sigmoid(y_pred)
                prob_bg = 1.0 - prob_fg
                y_pred = torch.cat([prob_bg, prob_fg], dim=1)
            else:
                y_pred = torch.softmax(y_pred, dim=1)
        # y: ensure [B,H,W] indices
        if y.ndim == 4:
            y = y[:, 0] if y.shape[1] == 1 else y.argmax(dim=1)
        return y_pred, y.long()
    return f


def loss_output_transform(output):
    """For multitask loss metrics: return (pred_dict, target_dict) or (y_pred, y)."""
    if isinstance(output, dict) and ("pred" in output) and ("label" in output):
        return output["pred"], output["label"]
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output
    raise ValueError(f"[loss_output_transform] Unexpected output type: {type(output)}")


# metrics utils
def _coerce_factory(ot, default):
    """
    Accepts:
      - None -> returns default
      - a transform factory (zero-arg callable returning a transform) -> calls once
      - a ready-to-use transform (callable taking `output`) -> returns as-is
    """
    if ot is None:
        return default
    try:
        res = ot()
        return res if callable(res) else ot
    except TypeError:
        return ot


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


def _build_seg_cm_ot():
    """ConfusionMatrix OT with probs (2 channels if binary) + index masks."""
    base = _coerce_factory(None, seg_confmat_output_transform)   # -> callable
    return _wrap_output_transform(_seg_cm_normalizer(base), "seg_cm_ot")


def compute_val_multi(metrics: dict, w: float = 0.65,
                      auc_key: str = "val_auc", dice_key: str = "val_dice") -> float:
    """Weighted harmonic mean of AUC and Dice. Returns 0.0 if either is missing/zero."""
    auc = float(metrics.get(auc_key, 0.0) or 0.0)
    dice = float(metrics.get(dice_key, 0.0) or 0.0)
    if auc <= 0.0 or dice <= 0.0:
        return 0.0
    return 1.0 / (w / auc + (1.0 - w) / dice)


def make_metrics(
    tasks,
    num_classes,
    loss_fn=None,
    cls_ot=None,
    auc_ot=None,
    seg_cm_ot=None,
    seg_num_classes=None,
    multitask=False,
    **_,
):
    """
    Build metrics for the active tasks.
      tasks: iterable of {"classification","segmentation"}
      num_classes: classification C
      seg_num_classes: segmentation classes (default 2)
      multitask: attach unified val_loss via loss_output_transform (requires combined loss_fn)
    """
    from ignite.metrics import Accuracy, ConfusionMatrix, Loss, DiceCoefficient, JaccardIndex, Precision, Recall, ROC_AUC
    from ignite.metrics import MetricsLambda
    import torch.nn as nn

    tasks = set(tasks or [])
    has_cls = "classification" in tasks
    has_seg = "segmentation" in tasks
    mixed_task = (has_cls and has_seg and not multitask)

    metrics = {}

    # classification metrics
    if has_cls or multitask:
        _cls_ot = _wrap_output_transform(_coerce_factory(cls_ot, cls_output_transform), "cls_ot")
        _auc_ot = _wrap_output_transform(_coerce_factory(auc_ot, lambda out: auc_output_transform(out)), "auc_ot")

        metrics.update({
            "val_acc": Accuracy(output_transform=_cls_ot),
            "val_prec": Precision(output_transform=_cls_ot, average=False),
            "val_recall": Recall(output_transform=_cls_ot, average=False),
            "val_auc": ROC_AUC(output_transform=_auc_ot),
            # Use dedicated OT for CM so shapes are guaranteed [B,C] vs [B]
            "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_confmat_output_transform),
        })

    # segmentation metrics
    if has_seg or multitask:
        seg_nc = max(2, int(seg_num_classes) if seg_num_classes is not None else 2)
        _seg_cm = ConfusionMatrix(
            num_classes=seg_nc,
            output_transform=(_wrap_output_transform(
                _seg_cm_normalizer(_coerce_factory(seg_cm_ot, seg_confmat_output_transform)), "seg_cm_ot") if seg_cm_ot is not None else _build_seg_cm_ot()))
        metrics.update({
            "val_dice": DiceCoefficient(cm=_seg_cm),
            "val_iou": JaccardIndex(cm=_seg_cm),
            "val_seg_confmat": _seg_cm,
        })

    # loss metrics
    if mixed_task:
        # Per-task validation losses (independent heads)
        metrics["val_cls_loss"] = Loss(
            loss_fn=loss_fn or nn.CrossEntropyLoss(),
            output_transform=_wrap_output_transform(_coerce_factory(cls_ot, cls_output_transform), "cls_ot"),
        )
        # Seg loss: use CE with indices; binary (C=1) is handled by _seg_ce_loss_ot
        metrics["val_seg_loss"] = Loss(
            loss_fn=nn.CrossEntropyLoss(),
            output_transform=_wrap_output_transform(_coerce_factory(None, _seg_ce_loss_ot), "seg_ce_loss_ot"),
        )
    elif multitask:
        if loss_fn is None:
            raise ValueError("multitask=True requires a combined loss_fn")
        metrics["val_loss"] = Loss(loss_fn=loss_fn, output_transform=loss_output_transform)
    else:
        if has_cls and not has_seg:
            metrics["val_loss"] = Loss(
                loss_fn=loss_fn or nn.CrossEntropyLoss(),
                output_transform=_wrap_output_transform(_coerce_factory(cls_ot, cls_output_transform), "cls_ot"),
            )
        elif has_seg and not has_cls:
            metrics["val_loss"] = Loss(
                # CE over indices works for binary (via _seg_ce_loss_ot) and multiclass
                loss_fn=nn.CrossEntropyLoss() if loss_fn is None else loss_fn,
                output_transform=_wrap_output_transform(_coerce_factory(None, _seg_ce_loss_ot), "seg_ce_loss_ot"),
            )

    if multitask and ("val_auc" in metrics) and ("val_dice" in metrics):
        # Weighted harmonic mean of AUC and Dice (vector)
        w = float(_.get("multi_weight", 0.65))
        eps = 1e-8
        metrics["val_multi"] = MetricsLambda(
            lambda auc, dice: 1.0 / (w / (auc + eps) + (1.0 - w) / (dice + eps)),
            metrics["val_auc"], metrics["val_dice"]
        )

        # Foreground only (common in med-seg, scaler)
        # dice_fg = MetricsLambda(lambda d: d[1], metrics["val_dice"])  # pick class-1
        # w, eps = 0.65, 1e-8
        # metrics["val_multi"] = MetricsLambda(
        #     lambda auc, d: 1.0 / (w / (auc + eps) + (1.0 - w) / (d + eps)),
        #     metrics["val_auc"], dice_fg
        # )

        # Mean Dice over classes (macro, scalar)
        # dice_mean = MetricsLambda(lambda d: d.mean(), metrics["val_dice"])
        # metrics["val_multi"] = MetricsLambda(
        #     lambda auc, d: 1.0 / (w / (auc + eps) + (1.0 - w) / (d + eps)),
        #     metrics["val_auc"], dice_mean
        # )

    return metrics


__all__ = [
    "make_metrics", "cls_output_transform", "auc_output_transform", "seg_output_transform",
    "cls_confmat_output_transform", "seg_confmat_output_transform", "loss_output_transform"
]
