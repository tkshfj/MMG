# metrics_utils.py
import logging
import torch
from monai.data.meta_tensor import MetaTensor


def get_segmentation_metrics():
    import torch.nn as nn
    return {"loss": nn.BCEWithLogitsLoss()}


def get_classification_metrics(loss_fn=None):
    import torch.nn as nn
    return {
        "loss": nn.CrossEntropyLoss(),
    }


def make_metrics(
    tasks,
    num_classes,
    loss_fn=None,
    cls_output_transform=None,
    auc_output_transform=None,
    seg_confmat_output_transform=None,
    multitask=False,
):
    from ignite.metrics import Accuracy, ConfusionMatrix, Loss, DiceCoefficient, JaccardIndex
    from ignite.metrics import Precision, Recall
    from ignite.metrics.roc_auc import ROC_AUC
    import torch.nn as nn
    # from monai.metrics import DiceMetric, MeanIoU

    metrics = {}

    # Classification metrics
    if "classification" in tasks or multitask:
        metrics.update({
            "val_acc": Accuracy(output_transform=cls_output_transform),
            "val_prec": Precision(output_transform=cls_output_transform, average=False),
            "val_recall": Recall(output_transform=cls_output_transform, average=False),
            "val_auc": ROC_AUC(output_transform=auc_output_transform),
            "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform),
        })

    # Segmentation metrics (via ConfusionMatrix)
    if "segmentation" in tasks or multitask:
        seg_cm = ConfusionMatrix(num_classes=num_classes, output_transform=seg_confmat_output_transform)
        metrics.update({
            "val_dice": DiceCoefficient(cm=seg_cm),
            "val_iou": JaccardIndex(cm=seg_cm),
            "val_seg_confmat": seg_cm,
            # If Dice expects discrete labels, use seg_discrete_output_transform instead.
            # "dice": DiceMetric(
            #     include_background=False,
            #     reduction="mean",
            #     get_not_nans=False,
            #     output_transform=lambda o: (torch.sigmoid(seg_output_transform(o)[0]), seg_output_transform(o)[1].long())),
            # or discrete version:
            # "dice": DiceMetric(include_background=False, reduction="mean",
            #                    output_transform=lambda o: seg_discrete_output_transform(o, "sigmoid", 0.5)),
            # "miou": MeanIoU(include_background=False, output_transform=lambda o: seg_discrete_output_transform(o, "softmax", 0.5))
        })

        # Loss metric wiring
        if multitask:
            # multitask_loss expects (pred_dict, target_dict)
            metrics["val_loss"] = Loss(loss_fn=loss_fn, output_transform=loss_output_transform)
        else:
            # single-task loss: pick the right transform automatically
            if "classification" in tasks and "segmentation" not in tasks:
                metrics["val_loss"] = Loss(
                    loss_fn=loss_fn or nn.CrossEntropyLoss(),
                    output_transform=cls_output_transform
                )
            elif "segmentation" in tasks and "classification" not in tasks:
                metrics["val_loss"] = Loss(
                    loss_fn=loss_fn or nn.BCEWithLogitsLoss(),
                    output_transform=seg_output_transform
                )
            else:
                # If both tasks are present but NOT multitask loss, you can expose per-task losses instead:
                metrics["val_cls_loss"] = Loss(
                    loss_fn=nn.CrossEntropyLoss(), output_transform=cls_output_transform
                )
                metrics["val_seg_loss"] = Loss(
                    loss_fn=nn.BCEWithLogitsLoss(), output_transform=seg_output_transform
                )

    return metrics


def _to_tensor(x):
    return x.as_tensor() if isinstance(x, MetaTensor) else x


# Classification
def cls_output_transform(output):
    # works for engine.state.output being a dict (this case) or a (y_pred, y) tuple
    if isinstance(output, tuple) and len(output) == 2:
        y_pred, y = output
    else:
        y_pred = output["pred"]["class_logits"]
        y = output["label"]["label"]
    return _to_tensor(y_pred).float(), _to_tensor(y).long()


# For AUROC specifically, return positive-class score
def auc_output_transform(output, positive_index=1):
    logits, y = cls_output_transform(output)
    if logits.ndim == 2 and logits.shape[-1] == 2:
        scores = torch.softmax(logits, dim=-1)[..., positive_index]
    elif logits.ndim == 1:
        scores = torch.sigmoid(logits)
    elif logits.ndim == 2 and logits.shape[-1] == 1:
        scores = torch.sigmoid(logits.squeeze(-1))
    else:
        raise ValueError(f"[AUC] Unexpected logits shape: {logits.shape}")
    return scores, y


# Segmentation
def seg_output_transform(output):
    if isinstance(output, tuple) and len(output) == 2:
        y_pred, y = output
    else:
        y_pred = output["pred"]["seg_out"]
        y = output["label"]["mask"]
    return _to_tensor(y_pred).float(), _to_tensor(y)


# If a metric expects discrete masks, add activation+thresholding here
def seg_discrete_output_transform(output, activation="sigmoid", threshold=0.5):
    logits, y = seg_output_transform(output)
    # probs
    if activation == "sigmoid":          # binary / multi-label: (B,1,...)
        probs = torch.sigmoid(logits)
        y_pred = (probs > threshold).long()
    elif activation == "softmax":        # multi-class: (B,C,...)
        probs = torch.softmax(logits, dim=1)
        y_pred = probs.argmax(dim=1)     # (B, H, W[, D])
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return y_pred, y.long()


# Optional: for computing multitask loss as a metric
# Returns (pred_dict, target_dict) so can pass `multitask_loss`
def loss_output_transform(output):
    if isinstance(output, dict) and ("pred" in output) and ("label" in output):
        return output["pred"], output["label"]
    if isinstance(output, tuple) and len(output) == 2:
        return output  # (y_pred, y)
    raise ValueError(f"[loss_output_transform] Unexpected output type: {type(output)}")


def cls_confmat_output_transform(output):
    """
    Output transform for classification confusion matrix.
    Expects batched output. Returns (predicted_class, true_class) as 1D tensors.
    """
    import torch

    y_pred, y_true = cls_output_transform(output)

    # Assert that both are tensors and have matching batch size
    if not isinstance(y_pred, torch.Tensor):
        raise TypeError(f"y_pred must be a tensor, got {type(y_pred)}")
    if not isinstance(y_true, torch.Tensor):
        raise TypeError(f"y_true must be a tensor, got {type(y_true)}")
    assert y_pred.shape[0] == y_true.shape[0], f"Batch size mismatch: {y_pred.shape}, {y_true.shape}"

    # Convert y_pred (logits) to class indices if necessary
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(dim=1)
    y_pred = y_pred.view(-1).long()
    y_true = y_true.view(-1).long()

    print(f"[DEBUG] Confmat Transform - y_pred.shape: {y_pred.shape}, y_true.shape: {y_true.shape}")

    return y_pred, y_true


def seg_confmat_output_transform(output):
    """
    Returns:
      probs [B, C, H, W]  (float)
      mask  [B, H, W]     (long)
    For binary (C=1) logits, expands to 2 channels: [bg, fg].
    """
    import torch

    if not isinstance(output, (list, tuple)):
        # when engine uses batch dict (no decollation)
        output = [output]

    probs_list, masks_list = [], []
    for sample in output:
        pred = sample["pred"]["seg_out"] if isinstance(sample.get("pred"), dict) else sample["pred"][1]
        mask = sample["label"]["mask"]

        # ensure tensors
        pred = pred.float()
        if isinstance(mask, torch.Tensor) and mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        mask = mask.long()

        # shape handling
        # pred: [B?, C, H, W] or [C, H, W]
        if pred.ndim == 3:  # [C, H, W] -> [1, C, H, W]
            pred = pred.unsqueeze(0)

        C = pred.shape[1]
        if C == 1:
            # binary: make 2-channel probs
            prob_fg = torch.sigmoid(pred)               # [B,1,H,W]
            prob_bg = 1.0 - prob_fg
            probs = torch.cat([prob_bg, prob_fg], dim=1)  # [B,2,H,W]
        else:
            probs = torch.softmax(pred, dim=1)          # [B,C,H,W]

        probs_list.append(probs)
        masks_list.append(mask)

    probs_batch = torch.cat(probs_list, dim=0) if len(probs_list) > 1 else probs_list[0]
    masks_batch = torch.cat(masks_list, dim=0) if len(masks_list) > 1 else masks_list[0]

    return probs_batch, masks_batch


def seg_confmat_output_transform_default(output):
    """
    Ignite ConfusionMatrix transform for segmentation.
    Returns:
        y_pred: [B, C, H, W] probabilities/logits (ConfusionMatrix will argmax if 4D)
        y:      [B, H, W] integer mask
    Uses pred["seg_out"] and label["mask"] explicitly.
    Supports a single dict per batch or a list of sample dicts (decollated).
    """
    def _as_batch(x):
        # Ensure [B, ...]
        return x if x.dim() >= 4 else x.unsqueeze(0)

    def _extract(sample):
        seg_logits = _to_tensor(sample["pred"]["seg_out"]).float()  # [B?, C, H, W] or [C, H, W]
        mask = _to_tensor(sample["label"]["mask"])                   # [B?, H, W] or [1, H, W]
        if seg_logits.dim() == 3:   # [C, H, W] -> [1, C, H, W]
            seg_logits = seg_logits.unsqueeze(0)
        if mask.dim() == 3 and mask.shape[0] == 1:  # [1, H, W] -> [H, W]
            mask = mask.squeeze(0)
        if mask.dim() == 2:         # [H, W] -> [1, H, W]
            mask = mask.unsqueeze(0)
        return seg_logits, mask.long()

    # Handle a single dict (batched tensors) OR a decollated list of dicts
    if isinstance(output, dict):
        y_pred, y = _extract(output)
        return y_pred, y

    if isinstance(output, list) and output and isinstance(output[0], dict):
        preds, masks = zip(*[_extract(s) for s in output])
        # Concatenate along batch
        y_pred = torch.cat(preds, dim=0)
        y = torch.cat(masks, dim=0)
        return y_pred, y

    raise ValueError("seg_confmat_output_transform_default expects dict or list[dict]")


# Attach Metrics
def attach_metrics(evaluator, model, config=None, val_loader=None):
    """Attach metrics to evaluator using model registry's get_metrics() method."""
    # Get metrics dictionary from the model registry
    metrics = model.get_metrics()
    # Attach each metric by key
    attached = []
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
        attached.append(name)
    # Debug/Info logs
    logging.info(f"Attached metrics: {', '.join(attached)}")
    if hasattr(evaluator, "metrics"):
        print("[attach_metrics] Registered metrics:", list(evaluator.metrics.keys()))
    elif hasattr(evaluator, "_metrics"):
        print("[attach_metrics] Registered metrics:", list(evaluator._metrics.keys()))
    else:
        print("[attach_metrics] Registered metrics not available.")
    return attached
