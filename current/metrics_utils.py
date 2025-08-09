# metrics_utils.py
import logging
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
        maybe = ot()
        if callable(maybe):
            return maybe
    except TypeError:
        pass
    return ot


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

    # tolerant extractors
    def _extract_logits(x):
        # Tensors pass-through
        if isinstance(x, torch.Tensor):
            return x.float()

        # Dict: try common paths first
        if isinstance(x, dict):
            # common direct paths
            for keys in (
                ("pred", "class_logits"),
                ("pred", "logits"),
                ("class_logits",),
                ("logits",),
                ("y_pred",),
                ("output",),
                ("scores",),
            ):
                v = _get_nested(x, keys)
                if v is not None:
                    return _to_tensor(v).float()

            # accept pred as tensor or nested dict with alternative keys
            if "pred" in x:
                v = x["pred"]
                if isinstance(v, torch.Tensor):
                    return v.float()
                if isinstance(v, dict):
                    for k in ("label", "y_pred", "cls", "out", "score", "logit"):
                        if k in v:
                            return _to_tensor(v[k]).float()
                    # last resort: first tensor-like value inside pred
                    for k, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            return vv.float()

            # last resort: first tensor-like value anywhere at top level
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    return v.float()

            _debug_once_dict(x, "logits-miss")
            raise KeyError("[cls_output_transform] Could not find logits in dict.")

        # Fallback: try to coerce
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

        def _collapse_to_BC(x):
            # Already [B, C]
            if x.ndim == 2:
                return x
            # ViT-style [B, T, C] (tokens): prefer class token 0, else mean over tokens
            if x.ndim == 3:
                # one-time debug
                if not getattr(cls_output_transform, "_dbg_collapse3", False):
                    print(f"[cls_output_transform] collapsing 3D logits {tuple(x.shape)} -> select token0/mean")
                    cls_output_transform._dbg_collapse3 = True
                try:
                    return x[:, 0, :]  # class token
                except Exception:
                    return x.mean(dim=1)
            # CNN-style [B, C, H, W, ...]: global average pool over spatial dims
            if x.ndim >= 4:
                if not getattr(cls_output_transform, "_dbg_collapseN", False):
                    print(f"[cls_output_transform] collapsing ND logits {tuple(x.shape)} -> GAP over dims 2..N")
                    cls_output_transform._dbg_collapseN = True
                if x.shape[1] >= 2:
                    dims = tuple(range(2, x.ndim))
                    return x.mean(dim=dims)
                else:
                    # If channel isn’t at dim=1, assume last dim is C and average the middle
                    # [B, *, C] where * can be any product of spatial dims
                    x = x.view(x.shape[0], -1, x.shape[-1])
                    return x.mean(dim=1)
            # Fallback: [C] -> [1, C]
            if x.ndim == 1:
                return x.unsqueeze(0)
            raise ValueError(f"[cls_output_transform] Unexpected logits rank: {x.ndim}")

        logits = _collapse_to_BC(logits).float()
        labels = labels.long().view(-1)

        if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"[cls_output_transform] Incompatible shapes after collapse: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
            )

        if not getattr(cls_output_transform, "_printed", False):
            print(f"[cls_output_transform] -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
            cls_output_transform._printed = True
        return logits, labels

    # def _finalize(logits, labels):
    #     if logits.ndim == 1:
    #         logits = logits.unsqueeze(0)
    #     if logits.ndim > 2 and logits.shape[-1] >= 2:
    #         logits = logits.reshape(-1, logits.shape[-1])
    #     if logits.ndim != 2:
    #         raise ValueError(f"[cls_output_transform] Expected logits [B, C], got {tuple(logits.shape)}")
    #     if not getattr(cls_output_transform, "_printed", False):
    #         print(f"[cls_output_transform] -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
    #         cls_output_transform._printed = True
    #     return logits.float(), labels.long().view(-1)

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


# def auc_output_transform(output, positive_index=1):
#     """
#     AUROC transform: returns (scores[B], labels[B]) where scores are prob of the positive class.
#     """
#     logits, labels = cls_output_transform(output)
#     if logits.ndim == 2 and logits.shape[-1] == 2:
#         scores = torch.softmax(logits, dim=-1)[:, positive_index]
#     elif logits.ndim == 2 and logits.shape[-1] == 1:
#         scores = torch.sigmoid(logits.squeeze(-1))
#     elif logits.ndim == 1:
#         scores = torch.sigmoid(logits)
#     else:
#         raise ValueError(f"[auc_output_transform] Unexpected logits shape: {tuple(logits.shape)}")
#     return scores, labels


# Segmentation transforms
def seg_output_transform(output):
    """
    Return (seg_logits, mask):
      seg_logits: float [B, C, H, W] (or [B, 1, H, W] for binary)
      mask:      long  [B, H, W] (class indices, not one-hot)
    Accepts (y_pred, y) or dict with pred['seg_out'], label['mask'].
    """
    if isinstance(output, (tuple, list)) and len(output) == 2:
        y_pred, y_true = output
    else:
        y_pred = output["pred"]["seg_out"]
        y_true = output["label"]["mask"]

    seg_logits = _to_tensor(y_pred).float()
    mask = _to_tensor(y_true)

    # logits -> [B, C, H, W]
    if seg_logits.ndim == 3:         # [C,H,W] -> [1,C,H,W]
        seg_logits = seg_logits.unsqueeze(0)

    # mask -> [B, H, W] (indices)
    if mask.ndim == 4:               # [B,C,H,W] or [B,1,H,W]
        mask = mask[:, 0] if mask.shape[1] == 1 else mask.argmax(dim=1)
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)        # [1,H,W] -> [H,W]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)      # [H,W] -> [1,H,W]

    return seg_logits, mask.long()


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
        prob_fg = torch.sigmoid(seg_logits)
        prob_bg = 1.0 - prob_fg
        y_pred = torch.cat([prob_bg, prob_fg], dim=1)
    else:
        y_pred = torch.softmax(seg_logits, dim=1)
    return y_pred, mask


def seg_discrete_output_transform(output, activation="sigmoid", threshold=0.5):
    """
    For metrics that expect discrete masks.
    """
    logits, y = seg_output_transform(output)
    if activation == "sigmoid":          # binary / multi-label: (B,1,...)
        probs = torch.sigmoid(logits)
        y_pred = (probs > threshold).long()
    elif activation == "softmax":        # multi-class: (B,C,...)
        probs = torch.softmax(logits, dim=1)
        y_pred = probs.argmax(dim=1)     # (B, H, W)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return y_pred, y.long()


def loss_output_transform(output):
    """
    For multitask loss metrics: return (pred_dict, target_dict) or (y_pred, y).
    """
    if isinstance(output, dict) and ("pred" in output) and ("label" in output):
        return output["pred"], output["label"]
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output
    raise ValueError(f"[loss_output_transform] Unexpected output type: {type(output)}")


# Metric bundles
def get_segmentation_metrics():
    import torch.nn as nn
    return {"loss": nn.BCEWithLogitsLoss()}


def get_classification_metrics(loss_fn=None):
    import torch.nn as nn
    return {"loss": nn.CrossEntropyLoss()}


def make_metrics(
    tasks,
    num_classes,
    loss_fn=None,
    cls_ot=None,
    auc_ot=None,
    seg_cm_ot=None,
    seg_num_classes=None,
    multitask=False,
    **legacy,  # accept old kwarg names
):
    """
    Build a metrics dict suitable for Ignite .attach(...).

    Args:
        tasks: list/iterable containing "classification" and/or "segmentation"
        num_classes: classification classes
        loss_fn: optional callable
        cls_ot: classification output_transform (factory or callable). If None, uses cls_output_transform
        auc_ot: auc output_transform (factory or callable). If None, uses auc_output_transform
        seg_cm_ot: segmentation confusion-matrix output_transform. If None, uses seg_confmat_output_transform
        seg_num_classes: classes for segmentation confmat; default 2 if None
        multitask: if True, wires a combined val_loss via loss_output_transform
    """
    from ignite.metrics import Accuracy, ConfusionMatrix, Loss, DiceCoefficient, JaccardIndex, Precision, Recall, ROC_AUC
    import torch.nn as nn

    # Resolve and wrap transforms safely
    cls_ot = _wrap_output_transform(_coerce_factory(cls_ot, cls_output_transform), "cls_ot")
    auc_ot = _wrap_output_transform(_coerce_factory(auc_ot, lambda out: auc_output_transform(out)), "auc_ot")

    # Build a normalized seg CM transform (probs + index labels)
    _seg_cm_base = _coerce_factory(seg_cm_ot, seg_confmat_output_transform)
    seg_cm_ot = _wrap_output_transform(_seg_cm_normalizer(_seg_cm_base), "seg_cm_ot")

    metrics = {}

    # Classification metrics
    if ("classification" in tasks) or multitask:
        metrics.update({
            "val_acc": Accuracy(output_transform=cls_ot),
            "val_prec": Precision(output_transform=cls_ot, average=False),
            "val_recall": Recall(output_transform=cls_ot, average=False),
            "val_auc": ROC_AUC(output_transform=auc_ot),
            "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_ot),
        })

    # Segmentation metrics (via ConfusionMatrix)
    if ("segmentation" in tasks) or multitask:
        # Default to 2 for binary segmentation if not provided; enforce >1
        seg_nc = int(seg_num_classes) if seg_num_classes is not None else 2
        if seg_nc <= 1:
            print("[make_metrics] seg_num_classes <= 1 detected; using 2 for binary segmentation CM.")
            seg_nc = 2

        seg_cm = ConfusionMatrix(num_classes=seg_nc, output_transform=seg_cm_ot)
        metrics.update({
            "val_dice": DiceCoefficient(cm=seg_cm),
            "val_iou": JaccardIndex(cm=seg_cm),
            "val_seg_confmat": seg_cm,
        })

    # Loss metrics (single-task / mixed-task / multitask combined)
    has_cls = "classification" in tasks
    has_seg = "segmentation" in tasks
    mixed_task = (has_cls and has_seg and not multitask)

    if mixed_task:
        # Per-task losses when using two independent heads
        metrics["val_cls_loss"] = Loss(
            loss_fn=loss_fn or nn.CrossEntropyLoss(),
            output_transform=cls_ot,
        )
        metrics["val_seg_loss"] = Loss(
            loss_fn=loss_fn or nn.BCEWithLogitsLoss(),
            output_transform=seg_output_transform,
        )
        # Intentionally omit unified `val_loss` in mixed-task mode.
    elif multitask:
        if loss_fn is None:
            raise ValueError("multitask=True requires a combined loss_fn")
        metrics["val_loss"] = Loss(loss_fn=loss_fn, output_transform=loss_output_transform)
    else:
        if has_cls and not has_seg:
            metrics["val_loss"] = Loss(
                loss_fn=loss_fn or nn.CrossEntropyLoss(),
                output_transform=cls_ot,
            )
        elif has_seg and not has_cls:
            metrics["val_loss"] = Loss(
                loss_fn=loss_fn or nn.BCEWithLogitsLoss(),
                output_transform=seg_output_transform,
            )

    return metrics


# Optional: classification confmat transform (pred indices)
def cls_confmat_output_transform(output):
    """
    Output transform for classification confusion matrix that returns predicted indices.
    """
    y_pred, y_true = cls_output_transform(output)
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(dim=1)
    y_pred = y_pred.view(-1).long()
    y_true = y_true.view(-1).long()
    print(f"[DEBUG] Confmat Transform - y_pred.shape: {y_pred.shape}, y_true.shape: {y_true.shape}")
    return y_pred, y_true


# Attach helper
def attach_metrics(evaluator, model, config=None, val_loader=None):
    """Attach metrics to evaluator using model registry's get_metrics() method."""
    metrics = model.get_metrics()
    attached = []
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
        attached.append(name)
    logging.info(f"Attached metrics: {', '.join(attached)}")
    if hasattr(evaluator, "metrics"):
        print("[attach_metrics] Registered metrics:", list(evaluator.metrics.keys()))
    elif hasattr(evaluator, "_metrics"):
        print("[attach_metrics] Registered metrics:", list(evaluator._metrics.keys()))
    else:
        print("[attach_metrics] Registered metrics not available.")
    return attached


__all__ = [
    # factories/bundles
    "make_metrics", "get_classification_metrics", "get_segmentation_metrics",
    # core transforms
    "cls_output_transform", "auc_output_transform", "seg_output_transform",
    "seg_confmat_output_transform", "seg_discrete_output_transform",
    "loss_output_transform", "cls_confmat_output_transform",
]
