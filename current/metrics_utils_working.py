# # metrics_utils.py
# import logging
# import torch
# from monai.data.meta_tensor import MetaTensor


# # Helpers
# def _to_tensor(x):
#     return x.as_tensor() if isinstance(x, MetaTensor) else x


# def _coerce_factory(ot, default):
#     """
#     Accepts:
#       - None -> returns default
#       - a transform factory (zero-arg callable returning a transform) -> calls once
#       - a ready-to-use transform (callable taking `output`) -> returns as-is
#     """
#     if ot is None:
#         return default
#     # Try calling with zero args: if it returns a callable, treat as factory.
#     try:
#         maybe = ot()
#         if callable(maybe):
#             return maybe
#     except TypeError:
#         pass
#     return ot


# def _wrap_output_transform(ot, name="output_transform"):
#     """
#     Wrap a transform so if it erroneously returns a function (e.g., lambda out: ot),
#     we call it one more time with the same output. Also sanity-check the return type.
#     """
#     def wrapped(output):
#         res = ot(output)
#         if callable(res):  # someone returned a function object; call it with output
#             res = res(output)
#         if not (isinstance(res, tuple) and len(res) == 2):
#             raise TypeError(f"{name} must return a (y_pred, y) tuple, got {type(res)}")
#         return res
#     return wrapped


# # Default transforms (safe & unified)
# def cls_output_transform(output):
#     """
#     Return (logits[B, C], labels[B]) with labels long.
#     Handles:
#       - (y_pred, y)
#       - (y_pred, y, ...)
#       - ((y_pred, y),)
#       - list[dict] decollated samples with pred['class_logits'], label['label']
#       - dict with pred['class_logits'], label['label']
#     """
#     import torch

#     # Tuple/List cases first
#     if isinstance(output, (tuple, list)):
#         # (y_pred, y)
#         if len(output) == 2 and all(
#             isinstance(x, (torch.Tensor, MetaTensor, dict, tuple, list)) for x in output
#         ):
#             y_pred, y_true = output

#             # If first item is itself a (y_pred, y) pair, unwrap
#             if isinstance(y_pred, (tuple, list)) and len(y_pred) == 2 and not isinstance(y_true, (torch.Tensor, MetaTensor)):
#                 y_pred, y_true = y_pred

#         # (y_pred, y, extra...) — take first two if they are tensors
#         elif len(output) >= 2 and all(isinstance(x, (torch.Tensor, MetaTensor)) for x in output[:2]):
#             y_pred, y_true = output[0], output[1]

#         # ((y_pred, y),) nested single element
#         elif len(output) == 1 and isinstance(output[0], (tuple, list)) and len(output[0]) == 2:
#             y_pred, y_true = output[0]

#         # list[dict] decollated
#         elif len(output) > 0 and isinstance(output[0], dict):
#             preds, labels = [], []
#             for s in output:
#                 p = s["pred"]["class_logits"]
#                 l = s["label"]["label"]
#                 p = _to_tensor(p).float()
#                 if p.ndim == 1:
#                     p = p.unsqueeze(0)      # [C] -> [1,C]
#                 preds.append(p)
#                 labels.append(_to_tensor(l).long().view(-1))  # [1]
#             logits = torch.cat(preds, dim=0)                   # [B,C]
#             labels = torch.cat(labels, dim=0).view(-1)         # [B]
#             # one-time debug
#             if not hasattr(cls_output_transform, "_printed"):
#                 print(f"[cls_output_transform] list[dict] -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
#                 cls_output_transform._printed = True
#             return logits, labels

#         else:
#             # Fall through and try dict path below
#             pass

#         # If we set y_pred/y_true above, finish here
#         if "y_pred" in locals() and "y_true" in locals():
#             # If inner was a dict (rare), pull tensors
#             if isinstance(y_pred, dict):
#                 y_pred = y_pred.get("label", y_pred.get("logits", y_pred.get("y_pred")))
#             logits = _to_tensor(y_pred).float()
#             labels = _to_tensor(y_true).long().view(-1)
#             if logits.ndim == 1:
#                 logits = logits.unsqueeze(0)
#             if logits.ndim != 2:
#                 raise ValueError(f"[cls_output_transform] Expected logits [B, C], got {tuple(logits.shape)}")
#             if not hasattr(cls_output_transform, "_printed"):
#                 print(f"[cls_output_transform] tuple/list -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
#                 cls_output_transform._printed = True
#             return logits, labels

#     # Dict (batched) case
#     if isinstance(output, dict):
#         y_pred = output["pred"]["class_logits"]
#         y_true = output["label"]["label"]
#         logits = _to_tensor(y_pred).float()
#         labels = _to_tensor(y_true).long().view(-1)
#         if logits.ndim == 1:
#             logits = logits.unsqueeze(0)
#         if logits.ndim != 2:
#             raise ValueError(f"[cls_output_transform] Expected logits [B, C], got {tuple(logits.shape)}")
#         if not hasattr(cls_output_transform, "_printed"):
#             print(f"[cls_output_transform] dict -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
#             cls_output_transform._printed = True
#         return logits, labels

#     # 3) Nothing matched
#     raise ValueError(f"[cls_output_transform] Unexpected output structure: type={type(output)}, value summary={str(type(output))}")


# def auc_output_transform(output, positive_index=1):
#     """
#     AUROC transform: returns (scores[B], labels[B]) where scores are prob of positive class.
#     """
#     logits, labels = cls_output_transform(output)
#     if logits.ndim == 2 and logits.shape[-1] == 2:
#         scores = torch.softmax(logits, dim=-1)[:, positive_index]
#     elif logits.ndim == 1:
#         scores = torch.sigmoid(logits)
#     elif logits.ndim == 2 and logits.shape[-1] == 1:
#         scores = torch.sigmoid(logits.squeeze(-1))
#     else:
#         raise ValueError(f"[aus_output_transform] Unexpected logits shape: {logits.shape}")
#     return scores, labels


# def seg_output_transform(output):
#     """
#     Return (seg_logits, mask):
#       seg_logits: float [B, C, H, W] (or [B, 1, H, W] for binary)
#       mask:      long  [B, H, W] (class indices, not one-hot)
#     Accepts (y_pred, y) or dict with pred['seg_out'], label['mask'].
#     """
#     if isinstance(output, (tuple, list)) and len(output) == 2:
#         y_pred, y_true = output
#     else:
#         y_pred = output["pred"]["seg_out"]
#         y_true = output["label"]["mask"]

#     seg_logits = _to_tensor(y_pred).float()
#     mask = _to_tensor(y_true)

#     # logits -> [B, C, H, W]
#     if seg_logits.dim() == 3:         # [C,H,W] -> [1,C,H,W]
#         seg_logits = seg_logits.unsqueeze(0)

#     # mask -> [B, H, W] (indices)
#     if mask.dim() == 4:               # [B,C,H,W] (one-hot) or [B,1,H,W]
#         mask = mask[:, 0] if mask.shape[1] == 1 else mask.argmax(dim=1)
#     elif mask.dim() == 3 and mask.shape[0] == 1:
#         mask = mask.squeeze(0)        # [1,H,W] -> [H,W]
#     if mask.dim() == 2:
#         mask = mask.unsqueeze(0)      # [H,W] -> [1,H,W]

#     return seg_logits, mask.long()


# def _seg_cm_normalizer(ot):
#     import torch

#     def f(output):
#         y_pred, y = ot(output)

#         # y_pred: ensure probs with 2 channels if binary
#         if y_pred.dim() == 4:
#             if y_pred.shape[1] == 1:
#                 prob_fg = torch.sigmoid(y_pred)
#                 prob_bg = 1.0 - prob_fg
#                 y_pred = torch.cat([prob_bg, prob_fg], dim=1)
#             else:
#                 # multi-class: convert logits to probs
#                 if y_pred.dtype.is_floating_point:
#                     y_pred = torch.softmax(y_pred, dim=1)

#         # y: ensure [B,H,W] indices (collapse channel if one-hot)
#         if y.dim() == 4:
#             y = y[:, 0] if y.shape[1] == 1 else y.argmax(dim=1)

#         return y_pred, y.long()
#     return f


# def seg_confmat_output_transform(output):
#     """
#     Ignite ConfusionMatrix transform for segmentation.
#     Returns:
#         y_pred: [B, C, H, W] probs/logits (ConfusionMatrix will argmax if 4D)
#         y:      [B, H, W] integer mask
#     """
#     seg_logits, mask = seg_output_transform(output)
#     # Convert logits to probs for stability
#     if seg_logits.shape[1] == 1:
#         # binary -> 2-channel probs
#         prob_fg = torch.sigmoid(seg_logits)
#         prob_bg = 1.0 - prob_fg
#         y_pred = torch.cat([prob_bg, prob_fg], dim=1)
#     else:
#         y_pred = torch.softmax(seg_logits, dim=1)
#     return y_pred, mask


# def seg_discrete_output_transform(output, activation="sigmoid", threshold=0.5):
#     """
#     If a metric expects discrete masks, add activation + thresholding here.
#     """
#     logits, y = seg_output_transform(output)
#     if activation == "sigmoid":          # binary / multi-label: (B,1,...)
#         probs = torch.sigmoid(logits)
#         y_pred = (probs > threshold).long()
#     elif activation == "softmax":        # multi-class: (B,C,...)
#         probs = torch.softmax(logits, dim=1)
#         y_pred = probs.argmax(dim=1)     # (B, H, W)
#     else:
#         raise ValueError(f"Unknown activation: {activation}")
#     return y_pred, y.long()


# def loss_output_transform(output):
#     """
#     For multitask loss metrics: return (pred_dict, target_dict) or (y_pred, y).
#     """
#     if isinstance(output, dict) and ("pred" in output) and ("label" in output):
#         return output["pred"], output["label"]
#     if isinstance(output, (tuple, list)) and len(output) == 2:
#         return output
#     raise ValueError(f"[loss_output_transform] Unexpected output type: {type(output)}")


# # Metric bundles
# def get_segmentation_metrics():
#     import torch.nn as nn
#     return {"loss": nn.BCEWithLogitsLoss()}


# def get_classification_metrics(loss_fn=None):
#     import torch.nn as nn
#     return {"loss": nn.CrossEntropyLoss()}


# def make_metrics(
#     tasks,
#     num_classes,
#     loss_fn=None,
#     cls_ot=None,
#     auc_ot=None,
#     seg_cm_ot=None,
#     seg_num_classes=None,
#     multitask=False,
#     **legacy,  # <— accept old kwarg names
# ):
#     """
#     Build a metrics dict suitable for Ignite .attach(...).

#     Args:
#         tasks: list/iterable containing "classification" and/or "segmentation"
#         num_classes: classification classes
#         loss_fn: optional callable
#         cls_ot: classification output_transform (factory or callable). If None, uses cls_output_transform
#         auc_ot: auc output_transform (factory or callable). If None, uses aus_output_transform
#         seg_cm_ot: segmentation confusion-matrix output_transform. If None, uses seg_confmat_output_transform
#         seg_num_classes: classes for segmentation confmat; defaults to num_classes if None
#         multitask: if True, wires a combined val_loss via loss_output_transform
#     """
#     from ignite.metrics import Accuracy, ConfusionMatrix, Loss, DiceCoefficient, JaccardIndex, Precision, Recall, ROC_AUC
#     import torch.nn as nn

#     # Resolve and wrap transforms safely
#     cls_ot = _wrap_output_transform(_coerce_factory(cls_ot, cls_output_transform), "cls_ot")
#     auc_ot = _wrap_output_transform(_coerce_factory(auc_ot, lambda out: auc_output_transform(out)), "auc_ot")

#     # Build a normalized seg CM transform
#     _seg_cm_base = _coerce_factory(seg_cm_ot, seg_confmat_output_transform)
#     seg_cm_ot = _wrap_output_transform(_seg_cm_normalizer(_seg_cm_base), "seg_cm_ot")
#     # cls_ot = _wrap_output_transform(_coerce_factory(cls_ot, cls_output_transform), "cls_ot")
#     # auc_ot = _wrap_output_transform(_coerce_factory(auc_ot, lambda out: aus_output_transform(out)), "auc_ot")
#     # seg_cm_ot = _wrap_output_transform(_coerce_factory(seg_cm_ot, seg_confmat_output_transform), "seg_cm_ot")

#     metrics = {}

#     # Classification metrics
#     if ("classification" in tasks) or multitask:
#         metrics.update({
#             "val_acc": Accuracy(output_transform=cls_ot),
#             "val_prec": Precision(output_transform=cls_ot, average=False),
#             "val_recall": Recall(output_transform=cls_ot, average=False),
#             "val_auc": ROC_AUC(output_transform=auc_ot),
#             "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_ot),
#         })

#     # Segmentation metrics (via ConfusionMatrix)
#     if ("segmentation" in tasks) or multitask:

#         # seg_nc = int(seg_nc) if seg_nc is not None else 2
#         # if seg_nc <= 1:
#         #     print("[make_metrics] seg_num_classes <= 1 detected; using 2 for binary segmentation CM.")
#         #     seg_nc = 2
#         seg_nc = seg_num_classes if seg_num_classes is not None else num_classes
#         seg_cm = ConfusionMatrix(num_classes=seg_nc, output_transform=seg_cm_ot)
#         metrics.update({
#             "val_dice": DiceCoefficient(cm=seg_cm),
#             "val_iou": JaccardIndex(cm=seg_cm),
#             "val_seg_confmat": seg_cm,
#         })

#     # Loss metrics (single-task / mixed-task / multitask combined)
#     # Mixed-task detection (both tasks requested, but NOT a multitask head)
#     has_cls = "classification" in tasks
#     has_seg = "segmentation" in tasks
#     mixed_task = (has_cls and has_seg and not multitask)

#     if mixed_task:
#         # Per-task losses when using two independent heads
#         metrics["val_cls_loss"] = Loss(
#             loss_fn=loss_fn or nn.CrossEntropyLoss(),
#             output_transform=cls_ot,
#         )
#         metrics["val_seg_loss"] = Loss(
#             loss_fn=loss_fn or nn.BCEWithLogitsLoss(),
#             output_transform=seg_output_transform,
#         )
#         # NOTE: we intentionally do NOT set `val_loss` in mixed-task mode to avoid ambiguity.

#     elif multitask:
#         # Combined loss for true multitask models
#         if loss_fn is None:
#             raise ValueError("multitask=True requires a combined loss_fn")
#         metrics["val_loss"] = Loss(loss_fn=loss_fn, output_transform=loss_output_transform)

#     else:
#         # Single-task: expose a single `val_loss` for the present task
#         if has_cls and not has_seg:
#             metrics["val_loss"] = Loss(
#                 loss_fn=loss_fn or nn.CrossEntropyLoss(),
#                 output_transform=cls_ot,
#             )
#         elif has_seg and not has_cls:
#             metrics["val_loss"] = Loss(
#                 loss_fn=loss_fn or nn.BCEWithLogitsLoss(),
#                 output_transform=seg_output_transform,
#             )

#     return metrics


# # Optional: classification confmat transform (pred indices)
# # (not used by default since ConfusionMatrix handles logits)
# def cls_confmat_output_transform(output):
#     """
#     Output transform for classification confusion matrix that returns predicted indices.
#     """
#     y_pred, y_true = cls_output_transform(output)
#     if y_pred.ndim > 1:
#         y_pred = y_pred.argmax(dim=1)
#     y_pred = y_pred.view(-1).long()
#     y_true = y_true.view(-1).long()
#     print(f"[DEBUG] Confmat Transform - y_pred.shape: {y_pred.shape}, y_true.shape: {y_true.shape}")
#     return y_pred, y_true


# # Attach Metrics helper
# def attach_metrics(evaluator, model, config=None, val_loader=None):
#     """Attach metrics to evaluator using model registry's get_metrics() method."""
#     metrics = model.get_metrics()
#     attached = []
#     for name, metric in metrics.items():
#         metric.attach(evaluator, name)
#         attached.append(name)
#     logging.info(f"Attached metrics: {', '.join(attached)}")
#     if hasattr(evaluator, "metrics"):
#         print("[attach_metrics] Registered metrics:", list(evaluator.metrics.keys()))
#     elif hasattr(evaluator, "_metrics"):
#         print("[attach_metrics] Registered metrics:", list(evaluator._metrics.keys()))
#     else:
#         print("[attach_metrics] Registered metrics not available.")
#     return attached


# __all__ = [
#     # factories/bundles
#     "make_metrics", "get_classification_metrics", "get_segmentation_metrics",
#     # core transforms
#     "cls_output_transform", "auc_output_transform", "seg_output_transform",
#     "seg_confmat_output_transform", "seg_discrete_output_transform",
#     "loss_output_transform", "cls_confmat_output_transform",
# ]

# metrics_utils.py
import logging
import torch
from monai.data.meta_tensor import MetaTensor


# Helpers
def _to_tensor(x):
    return x.as_tensor() if isinstance(x, MetaTensor) else x


def _coerce_factory(ot, default):
    """
    Accepts:
      - None -> returns default
      - a transform factory (zero-arg callable returning a transform) -> calls once
      - a ready-to-use transform (callable taking `output`) -> returns as-is
    """
    if ot is None:
        return default
    # Try calling with zero args: if it returns a callable, treat as factory.
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


# Default transforms (safe & unified)
def cls_output_transform(output):
    """
    Return (logits[B, C], labels[B]) with labels long.
    Handles:
      - (y_pred, y)
      - (y_pred, y, ...)
      - ((y_pred, y),)
      - list[dict] decollated samples with pred['class_logits'], label['label']
      - dict with pred['class_logits'], label['label']
    """
    import torch

    # Tuple/List cases first
    if isinstance(output, (tuple, list)):
        # (y_pred, y)
        if len(output) == 2 and all(
            isinstance(x, (torch.Tensor, MetaTensor, dict, tuple, list)) for x in output
        ):
            y_pred, y_true = output

            # If first item is itself a (y_pred, y) pair, unwrap
            if isinstance(y_pred, (tuple, list)) and len(y_pred) == 2 and not isinstance(y_true, (torch.Tensor, MetaTensor)):
                y_pred, y_true = y_pred

        # (y_pred, y, extra...) — take first two if they are tensors
        elif len(output) >= 2 and all(isinstance(x, (torch.Tensor, MetaTensor)) for x in output[:2]):
            y_pred, y_true = output[0], output[1]

        # ((y_pred, y),) nested single element
        elif len(output) == 1 and isinstance(output[0], (tuple, list)) and len(output[0]) == 2:
            y_pred, y_true = output[0]

        # list[dict] decollated
        elif len(output) > 0 and isinstance(output[0], dict):
            preds, labels = [], []
            for s in output:
                p = s["pred"]["class_logits"]
                l = s["label"]["label"]
                p = _to_tensor(p).float()
                if p.ndim == 1:
                    p = p.unsqueeze(0)      # [C] -> [1,C]
                preds.append(p)
                labels.append(_to_tensor(l).long().view(-1))  # [1]
            logits = torch.cat(preds, dim=0)                   # [B,C]
            labels = torch.cat(labels, dim=0).view(-1)         # [B]
            # one-time debug
            if not hasattr(cls_output_transform, "_printed"):
                print(f"[cls_output_transform] list[dict] -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
                cls_output_transform._printed = True
            return logits, labels

        else:
            # Fall through and try dict path below
            pass

        # If we set y_pred/y_true above, finish here
        if "y_pred" in locals() and "y_true" in locals():
            # If inner was a dict (rare), pull tensors
            if isinstance(y_pred, dict):
                y_pred = y_pred.get("label", y_pred.get("logits", y_pred.get("y_pred")))
            logits = _to_tensor(y_pred).float()
            labels = _to_tensor(y_true).long().view(-1)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.ndim != 2:
                raise ValueError(f"[cls_output_transform] Expected logits [B, C], got {tuple(logits.shape)}")
            if not hasattr(cls_output_transform, "_printed"):
                print(f"[cls_output_transform] tuple/list -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
                cls_output_transform._printed = True
            return logits, labels

    # Dict (batched) case
    if isinstance(output, dict):
        y_pred = output["pred"]["class_logits"]
        y_true = output["label"]["label"]
        logits = _to_tensor(y_pred).float()
        labels = _to_tensor(y_true).long().view(-1)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim != 2:
            raise ValueError(f"[cls_output_transform] Expected logits [B, C], got {tuple(logits.shape)}")
        if not hasattr(cls_output_transform, "_printed"):
            print(f"[cls_output_transform] dict -> logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")
            cls_output_transform._printed = True
        return logits, labels

    # 3) Nothing matched
    raise ValueError(f"[cls_output_transform] Unexpected output structure: type={type(output)}, value summary={str(type(output))}")


def auc_output_transform(output, positive_index=1):
    """
    AUROC transform: returns (scores[B], labels[B]) where scores are prob of positive class.
    """
    logits, labels = cls_output_transform(output)
    if logits.ndim == 2 and logits.shape[-1] == 2:
        scores = torch.softmax(logits, dim=-1)[:, positive_index]
    elif logits.ndim == 1:
        scores = torch.sigmoid(logits)
    elif logits.ndim == 2 and logits.shape[-1] == 1:
        scores = torch.sigmoid(logits.squeeze(-1))
    else:
        raise ValueError(f"[aus_output_transform] Unexpected logits shape: {logits.shape}")
    return scores, labels


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
    if seg_logits.dim() == 3:         # [C,H,W] -> [1,C,H,W]
        seg_logits = seg_logits.unsqueeze(0)

    # mask -> [B, H, W] (indices)
    if mask.dim() == 4:               # [B,C,H,W] (one-hot) or [B,1,H,W]
        mask = mask[:, 0] if mask.shape[1] == 1 else mask.argmax(dim=1)
    elif mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)        # [1,H,W] -> [H,W]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)      # [H,W] -> [1,H,W]

    return seg_logits, mask.long()


# def seg_output_transform(output):
#     """
#     Return (seg_logits, mask) for segmentation loss or postprocessing:
#       seg_logits: float tensor [B, C, H, W] (or [B, 1, H, W] for binary)
#       mask:      long tensor  [B, H, W]
#     Accepts tuple (y_pred, y) or dict with pred['seg_out'], label['mask'].
#     """
#     if isinstance(output, (tuple, list)) and len(output) == 2:
#         y_pred, y_true = output
#     else:
#         y_pred = output["pred"]["seg_out"]
#         y_true = output["label"]["mask"]

#     seg_logits = _to_tensor(y_pred).float()
#     mask = _to_tensor(y_true)

#     # Normalize shapes to [B, C, H, W] and [B, H, W]
#     if seg_logits.dim() == 3:     # [C, H, W] -> [1, C, H, W]
#         seg_logits = seg_logits.unsqueeze(0)
#     if mask.dim() == 3 and mask.shape[0] == 1:  # [1, H, W] -> [H, W]
#         mask = mask.squeeze(0)
#     if mask.dim() == 2:           # [H, W] -> [1, H, W]
#         mask = mask.unsqueeze(0)

#     return seg_logits, mask.long()


def _seg_cm_normalizer(ot):
    import torch

    def f(output):
        y_pred, y = ot(output)

        # y_pred: ensure probs with 2 channels if binary
        if y_pred.dim() == 4:
            if y_pred.shape[1] == 1:
                prob_fg = torch.sigmoid(y_pred)
                prob_bg = 1.0 - prob_fg
                y_pred = torch.cat([prob_bg, prob_fg], dim=1)
            else:
                # multi-class: convert logits to probs
                if y_pred.dtype.is_floating_point:
                    y_pred = torch.softmax(y_pred, dim=1)

        # y: ensure [B,H,W] indices (collapse channel if one-hot)
        if y.dim() == 4:
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
        # binary -> 2-channel probs
        prob_fg = torch.sigmoid(seg_logits)
        prob_bg = 1.0 - prob_fg
        y_pred = torch.cat([prob_bg, prob_fg], dim=1)
    else:
        y_pred = torch.softmax(seg_logits, dim=1)
    return y_pred, mask


def seg_discrete_output_transform(output, activation="sigmoid", threshold=0.5):
    """
    If a metric expects discrete masks, add activation + thresholding here.
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
    **legacy,  # <— accept old kwarg names
):
    """
    Build a metrics dict suitable for Ignite .attach(...).

    Args:
        tasks: list/iterable containing "classification" and/or "segmentation"
        num_classes: classification classes
        loss_fn: optional callable
        cls_ot: classification output_transform (factory or callable). If None, uses cls_output_transform
        auc_ot: auc output_transform (factory or callable). If None, uses aus_output_transform
        seg_cm_ot: segmentation confusion-matrix output_transform. If None, uses seg_confmat_output_transform
        seg_num_classes: classes for segmentation confmat; defaults to num_classes if None
        multitask: if True, wires a combined val_loss via loss_output_transform
    """
    from ignite.metrics import Accuracy, ConfusionMatrix, Loss, DiceCoefficient, JaccardIndex, Precision, Recall, ROC_AUC
    import torch.nn as nn

    # Map legacy names if the new ones are None
    # if cls_ot is None and "cls_output_transform" in legacy:
    #     cls_ot = legacy["cls_output_transform"]
    # if auc_ot is None and "auc_output_transform" in legacy:
    #     auc_ot = legacy["auc_output_transform"]
    # if seg_cm_ot is None and "seg_confmat_output_transform" in legacy:
    #     seg_cm_ot = legacy["seg_confmat_output_transform"]

    # Resolve and wrap transforms safely
    cls_ot = _wrap_output_transform(_coerce_factory(cls_ot, cls_output_transform), "cls_ot")
    auc_ot = _wrap_output_transform(_coerce_factory(auc_ot, lambda out: auc_output_transform(out)), "auc_ot")

    # Build a normalized seg CM transform
    _seg_cm_base = _coerce_factory(seg_cm_ot, seg_confmat_output_transform)
    seg_cm_ot = _wrap_output_transform(_seg_cm_normalizer(_seg_cm_base), "seg_cm_ot")
    # cls_ot = _wrap_output_transform(_coerce_factory(cls_ot, cls_output_transform), "cls_ot")
    # auc_ot = _wrap_output_transform(_coerce_factory(auc_ot, lambda out: aus_output_transform(out)), "auc_ot")
    # seg_cm_ot = _wrap_output_transform(_coerce_factory(seg_cm_ot, seg_confmat_output_transform), "seg_cm_ot")

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

        # seg_nc = int(seg_nc) if seg_nc is not None else 2
        # if seg_nc <= 1:
        #     print("[make_metrics] seg_num_classes <= 1 detected; using 2 for binary segmentation CM.")
        #     seg_nc = 2
        seg_nc = seg_num_classes if seg_num_classes is not None else num_classes
        seg_cm = ConfusionMatrix(num_classes=seg_nc, output_transform=seg_cm_ot)
        metrics.update({
            "val_dice": DiceCoefficient(cm=seg_cm),
            "val_iou": JaccardIndex(cm=seg_cm),
            "val_seg_confmat": seg_cm,
        })

    # Loss metrics (single-task / mixed-task / multitask combined)
    # Mixed-task detection (both tasks requested, but NOT a multitask head)
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
        # NOTE: we intentionally do NOT set `val_loss` in mixed-task mode to avoid ambiguity.

    elif multitask:
        # Combined loss for true multitask models
        if loss_fn is None:
            raise ValueError("multitask=True requires a combined loss_fn")
        metrics["val_loss"] = Loss(loss_fn=loss_fn, output_transform=loss_output_transform)

    else:
        # Single-task: expose a single `val_loss` for the present task
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

    # metrics = {}

    # # Classification metrics
    # if ("classification" in tasks) or multitask:
    #     metrics.update({
    #         "val_acc": Accuracy(output_transform=cls_ot),
    #         "val_prec": Precision(output_transform=cls_ot, average=False),
    #         "val_recall": Recall(output_transform=cls_ot, average=False),
    #         "val_auc": ROC_AUC(output_transform=auc_ot),
    #         "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_ot),
    #     })

    #     # Single-task classification loss metric (when not multitask)
    #     if not multitask and (("segmentation" not in tasks) or ("classification" in tasks and "segmentation" not in tasks)):
    #         metrics["val_loss"] = Loss(loss_fn=loss_fn or nn.CrossEntropyLoss(), output_transform=cls_ot)

    # # Segmentation metrics (via ConfusionMatrix)
    # if ("segmentation" in tasks) or multitask:
    #     seg_nc = seg_num_classes if seg_num_classes is not None else num_classes
    #     seg_cm = ConfusionMatrix(num_classes=seg_nc, output_transform=seg_cm_ot)
    #     metrics.update({
    #         "val_dice": DiceCoefficient(cm=seg_cm),
    #         "val_iou": JaccardIndex(cm=seg_cm),
    #         "val_seg_confmat": seg_cm,
    #     })

    #     # Single-task segmentation loss metric
    #     if not multitask and (("classification" not in tasks) or ("segmentation" in tasks and "classification" not in tasks)):
    #         metrics["val_loss"] = Loss(loss_fn=loss_fn or nn.BCEWithLogitsLoss(), output_transform=seg_output_transform)

    # # Multitask combined loss metric
    # if multitask:
    #     if loss_fn is None:
    #         raise ValueError("multitask=True requires a combined loss_fn")
    #     metrics["val_loss"] = Loss(loss_fn=loss_fn, output_transform=loss_output_transform)

    # return metrics


# Optional: classification confmat transform (pred indices)
# (not used by default since ConfusionMatrix handles logits)
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


# Attach Metrics helper
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


# # metrics_utils.py
# import logging
# import torch
# from monai.data.meta_tensor import MetaTensor


# def get_segmentation_metrics():
#     import torch.nn as nn
#     return {"loss": nn.BCEWithLogitsLoss()}


# def get_classification_metrics(loss_fn=None):
#     import torch.nn as nn
#     return {
#         "loss": nn.CrossEntropyLoss(),
#     }


# def make_metrics(
#     tasks,
#     num_classes,
#     loss_fn=None,
#     cls_output_transform=None,
#     auc_output_transform=None,
#     seg_confmat_output_transform=None,
#     multitask=False,
# ):
#     from ignite.metrics import Accuracy, ConfusionMatrix, Loss, DiceCoefficient, JaccardIndex
#     from ignite.metrics import Precision, Recall
#     from ignite.metrics.roc_auc import ROC_AUC
#     import torch.nn as nn
#     # from monai.metrics import DiceMetric, MeanIoU

#     metrics = {}

#     # Classification metrics
#     if "classification" in tasks or multitask:
#         metrics.update({
#             "val_acc": Accuracy(output_transform=cls_output_transform),
#             "val_prec": Precision(output_transform=cls_output_transform, average=False),
#             "val_recall": Recall(output_transform=cls_output_transform, average=False),
#             "val_auc": ROC_AUC(output_transform=auc_output_transform),
#             "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform),
#         })

#     # Segmentation metrics (via ConfusionMatrix)
#     if "segmentation" in tasks or multitask:
#         seg_cm = ConfusionMatrix(num_classes=num_classes, output_transform=seg_confmat_output_transform)
#         metrics.update({
#             "val_dice": DiceCoefficient(cm=seg_cm),
#             "val_iou": JaccardIndex(cm=seg_cm),
#             "val_seg_confmat": seg_cm,
#             # If Dice expects discrete labels, use seg_discrete_output_transform instead.
#             # "dice": DiceMetric(
#             #     include_background=False,
#             #     reduction="mean",
#             #     get_not_nans=False,
#             #     output_transform=lambda o: (torch.sigmoid(seg_output_transform(o)[0]), seg_output_transform(o)[1].long())),
#             # or discrete version:
#             # "dice": DiceMetric(include_background=False, reduction="mean",
#             #                    output_transform=lambda o: seg_discrete_output_transform(o, "sigmoid", 0.5)),
#             # "miou": MeanIoU(include_background=False, output_transform=lambda o: seg_discrete_output_transform(o, "softmax", 0.5))
#         })

#         # Loss metric wiring
#         if multitask:
#             # multitask_loss expects (pred_dict, target_dict)
#             metrics["val_loss"] = Loss(loss_fn=loss_fn, output_transform=loss_output_transform)
#         else:
#             # single-task loss: pick the right transform automatically
#             if "classification" in tasks and "segmentation" not in tasks:
#                 metrics["val_loss"] = Loss(
#                     loss_fn=loss_fn or nn.CrossEntropyLoss(),
#                     output_transform=cls_output_transform
#                 )
#             elif "segmentation" in tasks and "classification" not in tasks:
#                 metrics["val_loss"] = Loss(
#                     loss_fn=loss_fn or nn.BCEWithLogitsLoss(),
#                     output_transform=seg_output_transform
#                 )
#             else:
#                 # If both tasks are present but NOT multitask loss, you can expose per-task losses instead:
#                 metrics["val_cls_loss"] = Loss(
#                     loss_fn=nn.CrossEntropyLoss(), output_transform=cls_output_transform
#                 )
#                 metrics["val_seg_loss"] = Loss(
#                     loss_fn=nn.BCEWithLogitsLoss(), output_transform=seg_output_transform
#                 )

#     return metrics


# def _to_tensor(x):
#     return x.as_tensor() if isinstance(x, MetaTensor) else x


# # Classification
# def cls_output_transform(output):
#     # works for engine.state.output being a dict (this case) or a (y_pred, y) tuple
#     if isinstance(output, tuple) and len(output) == 2:
#         y_pred, y = output
#     else:
#         y_pred = output["pred"]["class_logits"]
#         y = output["label"]["label"]
#     return _to_tensor(y_pred).float(), _to_tensor(y).long()


# # For AUROC specifically, return positive-class score
# def auc_output_transform(output, positive_index=1):
#     logits, y = cls_output_transform(output)
#     if logits.ndim == 2 and logits.shape[-1] == 2:
#         scores = torch.softmax(logits, dim=-1)[..., positive_index]
#     elif logits.ndim == 1:
#         scores = torch.sigmoid(logits)
#     elif logits.ndim == 2 and logits.shape[-1] == 1:
#         scores = torch.sigmoid(logits.squeeze(-1))
#     else:
#         raise ValueError(f"[AUC] Unexpected logits shape: {logits.shape}")
#     return scores, y


# # Segmentation
# def seg_output_transform(output):
#     if isinstance(output, tuple) and len(output) == 2:
#         y_pred, y = output
#     else:
#         y_pred = output["pred"]["seg_out"]
#         y = output["label"]["mask"]
#     return _to_tensor(y_pred).float(), _to_tensor(y)


# # If a metric expects discrete masks, add activation+thresholding here
# def seg_discrete_output_transform(output, activation="sigmoid", threshold=0.5):
#     logits, y = seg_output_transform(output)
#     # probs
#     if activation == "sigmoid":          # binary / multi-label: (B,1,...)
#         probs = torch.sigmoid(logits)
#         y_pred = (probs > threshold).long()
#     elif activation == "softmax":        # multi-class: (B,C,...)
#         probs = torch.softmax(logits, dim=1)
#         y_pred = probs.argmax(dim=1)     # (B, H, W[, D])
#     else:
#         raise ValueError(f"Unknown activation: {activation}")
#     return y_pred, y.long()


# # Optional: for computing multitask loss as a metric
# # Returns (pred_dict, target_dict) so can pass `multitask_loss`
# def loss_output_transform(output):
#     if isinstance(output, dict) and ("pred" in output) and ("label" in output):
#         return output["pred"], output["label"]
#     if isinstance(output, tuple) and len(output) == 2:
#         return output  # (y_pred, y)
#     raise ValueError(f"[loss_output_transform] Unexpected output type: {type(output)}")


# def cls_confmat_output_transform(output):
#     """
#     Output transform for classification confusion matrix.
#     Expects batched output. Returns (predicted_class, true_class) as 1D tensors.
#     """
#     import torch

#     y_pred, y_true = cls_output_transform(output)

#     # Assert that both are tensors and have matching batch size
#     if not isinstance(y_pred, torch.Tensor):
#         raise TypeError(f"y_pred must be a tensor, got {type(y_pred)}")
#     if not isinstance(y_true, torch.Tensor):
#         raise TypeError(f"y_true must be a tensor, got {type(y_true)}")
#     assert y_pred.shape[0] == y_true.shape[0], f"Batch size mismatch: {y_pred.shape}, {y_true.shape}"

#     # Convert y_pred (logits) to class indices if necessary
#     if y_pred.ndim > 1:
#         y_pred = y_pred.argmax(dim=1)
#     y_pred = y_pred.view(-1).long()
#     y_true = y_true.view(-1).long()

#     print(f"[DEBUG] Confmat Transform - y_pred.shape: {y_pred.shape}, y_true.shape: {y_true.shape}")

#     return y_pred, y_true


# def seg_confmat_output_transform(output):
#     """
#     Returns:
#       probs [B, C, H, W]  (float)
#       mask  [B, H, W]     (long)
#     For binary (C=1) logits, expands to 2 channels: [bg, fg].
#     """
#     import torch

#     if not isinstance(output, (list, tuple)):
#         # when engine uses batch dict (no decollation)
#         output = [output]

#     probs_list, masks_list = [], []
#     for sample in output:
#         pred = sample["pred"]["seg_out"] if isinstance(sample.get("pred"), dict) else sample["pred"][1]
#         mask = sample["label"]["mask"]

#         # ensure tensors
#         pred = pred.float()
#         if isinstance(mask, torch.Tensor) and mask.ndim == 4 and mask.shape[1] == 1:
#             mask = mask.squeeze(1)
#         mask = mask.long()

#         # shape handling
#         # pred: [B?, C, H, W] or [C, H, W]
#         if pred.ndim == 3:  # [C, H, W] -> [1, C, H, W]
#             pred = pred.unsqueeze(0)

#         C = pred.shape[1]
#         if C == 1:
#             # binary: make 2-channel probs
#             prob_fg = torch.sigmoid(pred)               # [B,1,H,W]
#             prob_bg = 1.0 - prob_fg
#             probs = torch.cat([prob_bg, prob_fg], dim=1)  # [B,2,H,W]
#         else:
#             probs = torch.softmax(pred, dim=1)          # [B,C,H,W]

#         probs_list.append(probs)
#         masks_list.append(mask)

#     probs_batch = torch.cat(probs_list, dim=0) if len(probs_list) > 1 else probs_list[0]
#     masks_batch = torch.cat(masks_list, dim=0) if len(masks_list) > 1 else masks_list[0]

#     return probs_batch, masks_batch


# def seg_confmat_output_transform(output):
#     """
#     Ignite ConfusionMatrix transform for segmentation.
#     Returns:
#         y_pred: [B, C, H, W] probabilities/logits (ConfusionMatrix will argmax if 4D)
#         y:      [B, H, W] integer mask
#     Uses pred["seg_out"] and label["mask"] explicitly.
#     Supports a single dict per batch or a list of sample dicts (decollated).
#     """
#     def _as_batch(x):
#         # Ensure [B, ...]
#         return x if x.dim() >= 4 else x.unsqueeze(0)

#     def _extract(sample):
#         seg_logits = _to_tensor(sample["pred"]["seg_out"]).float()  # [B?, C, H, W] or [C, H, W]
#         mask = _to_tensor(sample["label"]["mask"])                   # [B?, H, W] or [1, H, W]
#         if seg_logits.dim() == 3:   # [C, H, W] -> [1, C, H, W]
#             seg_logits = seg_logits.unsqueeze(0)
#         if mask.dim() == 3 and mask.shape[0] == 1:  # [1, H, W] -> [H, W]
#             mask = mask.squeeze(0)
#         if mask.dim() == 2:         # [H, W] -> [1, H, W]
#             mask = mask.unsqueeze(0)
#         return seg_logits, mask.long()

#     # Handle a single dict (batched tensors) OR a decollated list of dicts
#     if isinstance(output, dict):
#         y_pred, y = _extract(output)
#         return y_pred, y

#     if isinstance(output, list) and output and isinstance(output[0], dict):
#         preds, masks = zip(*[_extract(s) for s in output])
#         # Concatenate along batch
#         y_pred = torch.cat(preds, dim=0)
#         y = torch.cat(masks, dim=0)
#         return y_pred, y

#     raise ValueError("seg_confmat_output_transform expects dict or list[dict]")


# # Attach Metrics
# def attach_metrics(evaluator, model, config=None, val_loader=None):
#     """Attach metrics to evaluator using model registry's get_metrics() method."""
#     # Get metrics dictionary from the model registry
#     metrics = model.get_metrics()
#     # Attach each metric by key
#     attached = []
#     for name, metric in metrics.items():
#         metric.attach(evaluator, name)
#         attached.append(name)
#     # Debug/Info logs
#     logging.info(f"Attached metrics: {', '.join(attached)}")
#     if hasattr(evaluator, "metrics"):
#         print("[attach_metrics] Registered metrics:", list(evaluator.metrics.keys()))
#     elif hasattr(evaluator, "_metrics"):
#         print("[attach_metrics] Registered metrics:", list(evaluator._metrics.keys()))
#     else:
#         print("[attach_metrics] Registered metrics not available.")
#     return attached
