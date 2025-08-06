# metrics_utils.py
import torch
import logging
import os


def _debug_enabled():
    return os.environ.get("DEBUG_OUTPUT_TRANSFORMS", "0") in ("1", "true", "True", "yes", "YES")


# Metric/Handler/Config Factories
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
    from ignite.metrics.roc_auc import ROC_AUC
    metrics = {}

    if "classification" in tasks or multitask:
        # For multitask, we often want separate "val_acc", "val_auc" etc
        metrics.update({
            "val_acc": Accuracy(output_transform=cls_output_transform),
            "val_auc": ROC_AUC(output_transform=auc_output_transform),
            "val_loss": Loss(loss_fn=loss_fn, output_transform=cls_output_transform),
            "val_cls_confmat": ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform),
        })

    if "segmentation" in tasks or multitask:
        seg_cm = ConfusionMatrix(num_classes=num_classes, output_transform=seg_confmat_output_transform)
        metrics.update({
            "val_dice": DiceCoefficient(cm=seg_cm),
            "val_iou": JaccardIndex(cm=seg_cm),
            "val_seg_confmat": seg_cm,
        })
    return metrics


# Output Transforms for Accuracy, ConfusionMatrix
def cls_output_transform(output):
    # DEBUG: print the output structure to help debugging!
    # print("DEBUG output to cls_output_transform:", type(output), output)
    def unwrap_dict_to_tensor(obj, keys=("label", "classification", "class", "value")):
        # Recursively unwrap nested dicts
        while isinstance(obj, dict):
            found = False
            for k in keys:
                if k in obj:
                    obj = obj[k]
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not extract tensor from nested dict: {obj}")
        return obj

    # Unpack output
    if isinstance(output, (tuple, list)):
        if len(output) == 2:
            y_pred, y = output
        elif len(output) > 2:
            y_pred, y = output[0], output[1]
        else:
            raise ValueError(f"Output tuple/list too short: {output}")
    elif isinstance(output, dict):
        y_pred = output.get("y_pred") or output.get("pred") or output.get("logits")
        y = output.get("y") or output.get("label") or output.get("classification")
        if y_pred is None or y is None:
            raise ValueError(f"Cannot extract y_pred and y from dict output: keys={output.keys()}")
    else:
        raise ValueError(f"Unexpected output type in metric output_transform: {type(output)}")
    # Handle nested dicts (common for y_pred, y)
    if isinstance(y_pred, dict):
        for k in ("logits", "pred", "output"):
            if k in y_pred:
                y_pred = y_pred[k]
                break
        else:
            raise ValueError(f"Could not extract tensor from y_pred dict: {y_pred}")
    # Recursively unwrap if still dict
    y_pred = unwrap_dict_to_tensor(y_pred, keys=("logits", "pred", "output"))
    y = unwrap_dict_to_tensor(y, keys=("label", "classification", "class", "value"))

    # If y_pred or y is a list, filter out non-classification tensors (e.g., ignore 3D/4D tensors)
    def is_classification_tensor(t):
        return isinstance(t, torch.Tensor) and (t.ndim <= 2)

    if isinstance(y_pred, list):
        filtered = [yy for yy in y_pred if is_classification_tensor(yy)]
        if not filtered:
            raise ValueError(f"No valid classification logits in y_pred list: {[getattr(yy,'shape',None) for yy in y_pred]}")
        y_pred = torch.stack(filtered)
    if isinstance(y, list):
        filtered = [yy for yy in y if is_classification_tensor(yy)]
        if not filtered:
            raise ValueError(f"No valid class labels in y list: {[getattr(yy,'shape',None) for yy in y]}")
        y = torch.stack(filtered)
    # Convert to tensor if not already
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.as_tensor(y_pred)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    # Convert one-hot y to indices if necessary
    if y.ndim == 2 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
    if y.ndim == 0:
        y = y.unsqueeze(0)
    if y.ndim > 1:
        y = y.view(-1)

    # print(f"[DEBUG] y_pred.shape after extraction: {y_pred.shape}, y.shape: {y.shape}")
    print(f"[DEBUG Confmat Transform] y_pred.shape: {y_pred.shape}, y.shape: {y.shape}, y_pred: {y_pred}, y: {y}")

    return y_pred, y


def cls_confmat_output_transform(output):
    """
    Output transform for classification confusion matrix.
    Returns (predicted_class, true_class) 1D tensors.
    """
    y_pred, y = cls_output_transform(output)
    # y_pred should be [B, 2], y should be [B]
    assert y_pred.shape[0] == y.shape[0], f"Shape mismatch: {y_pred.shape}, {y.shape}"
    # Convert logits/probs to class indices if needed
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(dim=1)
    # Make sure both are 1D and long dtype
    y_pred = y_pred.long().view(-1)
    y = y.long().view(-1)
    print(f"[DEBUG] Confmat Transform - y_pred.shape: {y_pred.shape}, y.shape: {y.shape}")
    return y_pred, y


# for ROC AUC
def auc_output_transform(output):
    """
    Extracts (prob_class1, class_idx) from model output for ROC_AUC metric.
    Handles tuple/list, dict, and nested dict structures robustly.
    Hardened: prints input and fails clearly if label is missing or None.
    """
    def unwrap(obj, keys=("y_pred", "pred", "logits", "output")):
        while isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    obj = obj[k]
                    break
            else:
                # If dict but none of the keys match, fail
                raise ValueError(f"Cannot extract tensor from dict: {obj}")
        return obj

    # DEBUG: print the incoming output for analysis
    if _debug_enabled():
        print(f"[DEBUG] auc_output_transform raw output: {type(output)}, {output}")

    if isinstance(output, (tuple, list)):
        # If list of dicts, old collate format
        if all(isinstance(x, dict) for x in output):
            logits_list = []
            labels_list = []
            for idx, x in enumerate(output):
                # Extract logits
                try:
                    logits_val = unwrap(x.get("pred", x.get("logits")), ("logits", "pred"))
                except Exception:
                    if _debug_enabled():
                        print(f"[ERROR] Failed to extract logits at entry {idx}: {x}")
                    raise
                logits_list.append(logits_val)

                # Extract label
                lbl = x.get("label")
                if isinstance(lbl, dict):
                    for key in ("label", "classification", "class"):
                        if key in lbl:
                            lbl = lbl[key]
                            break
                if lbl is None:
                    if _debug_enabled():
                        print(f"[ERROR] Label missing or None at entry {idx}: {x}")
                    raise ValueError(f"Label not found in output entry {idx}: {x}")
                labels_list.append(lbl)

            # Check for None in labels_list and print if present
            if any(label is None for label in labels_list):
                print("[ERROR] NoneType found in labels_list:", labels_list)
                raise ValueError("One or more labels are None in auc_output_transform.")

            # Before stacking, inspect and convert if needed
            for i, logit in enumerate(logits_list):
                if _debug_enabled():
                    print(f"[DEBUG] Before tensor conversion: logits_list[{i}] type={type(logit)}, value={logit}")
                # Now, more robust: flatten or select just the classification output if tuple/list
                if isinstance(logit, (list, tuple)):
                    # If tuple/list of tensors, pick the first tensor (usually the classifier head)
                    # Optionally: add checks for what each element is
                    if _debug_enabled():
                        print(f"[DEBUG] logits_list[{i}] is list/tuple, length={len(logit)}")
                    # Let's check all elements:
                    for j, elem in enumerate(logit):
                        if _debug_enabled():
                            print(f"    [DEBUG] logits_list[{i}][{j}] type={type(elem)}, shape={getattr(elem, 'shape', None)}")
                    # Try to select the "most likely" tensor: pick the first tensor with 1D/2D shape
                    found = False
                    for elem in logit:
                        if isinstance(elem, torch.Tensor) and elem.ndim in (1, 2):
                            logits_list[i] = elem
                            found = True
                            break
                    if not found:
                        raise ValueError(f"[ERROR] Could not find a 1D/2D tensor in logits_list[{i}]: {logit}")
                    continue
                elif isinstance(logit, torch.Tensor):
                    logits_list[i] = logit
                elif isinstance(logit, (int, float)):
                    logits_list[i] = torch.tensor([logit])
                else:
                    # Try as_tensor for numpy
                    try:
                        import numpy as np
                        if isinstance(logit, np.ndarray):
                            logits_list[i] = torch.as_tensor(logit)
                            continue
                    except ImportError:
                        pass
                    raise ValueError(f"[ERROR] Unrecognized logit type at index {i}: {type(logit)}: {logit}")

            shapes = [logit.shape for logit in logits_list]
            if _debug_enabled():
                print("[DEBUG] Final logits_list shapes before stack:", shapes)
            if len(set(shapes)) != 1:
                if _debug_enabled():
                    print("[ERROR] Inconsistent shapes in logits_list:", shapes)
                raise ValueError(f"Inconsistent shapes in logits_list: {shapes}")

            if _debug_enabled():
                print("DEBUG logits_list shapes:", [getattr(logit, 'shape', None) for logit in logits_list])

            logits = torch.stack(logits_list)
            # logits = torch.stack([torch.as_tensor(logit) for logit in logits_list])
            labels = torch.tensor(labels_list, dtype=torch.long, device=logits.device)

        else:
            # Modern collate: output = (y_pred, y)
            if len(output) < 2:
                if _debug_enabled():
                    print("[ERROR] Output tuple/list too short for auc_output_transform:", output)
                raise ValueError(f"Output tuple/list too short for auc_output_transform: {output}")
            logits, labels = output[0], output[1]
    elif isinstance(output, dict):
        logits = unwrap(output.get("y_pred") or output.get("pred") or output.get("logits"), ("logits", "pred", "output"))
        labels = unwrap(output.get("y") or output.get("label") or output.get("classification"), ("label", "classification", "class", "value"))
    else:
        if _debug_enabled():
            print("[ERROR] Unexpected output type in auc_output_transform:", type(output))
        raise ValueError(f"Unexpected output type in auc_output_transform: {type(output)}")

    # DEBUG: print the processed logits and labels
    if _debug_enabled():
        print("[DEBUG] auc_output_transform: logits shape", getattr(logits, 'shape', None), "labels", labels)

    # At this point, logits should be [B, C] or [C], labels [B] or []
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)

    # AUC expects: (prob_class1, labels) for binary, (B, C), (B,)
    if logits.ndim == 1:  # [C]
        logits = logits.unsqueeze(0)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = torch.argmax(labels, dim=1)

    # Use class 1 probability for binary
    if logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)
        prob_class1 = probs[:, 1]  # Probability of class 1; positive_class_probs
    elif logits.shape[1] == 1:
        # Single logit: use sigmoid, prob = torch.sigmoid(logits)
        prob_class1 = torch.sigmoid(logits.squeeze(1))
    else:
        # Multi-class: just use softmax as probs
        prob_class1 = torch.softmax(logits, dim=1)

    if _debug_enabled():
        print("[DEBUG] auc_output_transform: final prob_class1", prob_class1, "labels", labels)
    # For ignite.ROC_AUC, expected: (probs, labels)
    return prob_class1, labels


def seg_output_transform(output):
    """
    Extracts predicted and true segmentation masks for segmentation metrics.
    Returns:
        - pred_masks: [B, 1, H, W] or [B, H, W]
        - true_masks: [B, 1, H, W] or [B, H, W]
    """
    print("[DEBUG] seg_output_transform CALLED. type(output):", type(output))
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        pred_masks, true_masks = [], []
        for i, x in enumerate(output):
            if not isinstance(x.get("pred", None), (list, tuple)) or len(x["pred"]) < 2:
                raise ValueError(f"seg_output_transform: Missing or malformed 'pred' at index {i}: {x.get('pred')}")
            if "label" not in x or "mask" not in x["label"]:
                raise ValueError(f"seg_output_transform: Missing 'label' or 'label.mask' at index {i}: {x.get('label')}")
            pred = x["pred"][1]  # segmentation logits or mask
            mask = x["label"]["mask"]
            if pred.ndim == 4 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            pred_masks.append(pred)
            true_masks.append(mask)
        pred_masks = torch.stack(pred_masks)
        true_masks = torch.stack(true_masks)
        return pred_masks, true_masks
    raise ValueError("seg_output_transform expects a list of dicts with 'pred' (as tuple/list) and nested 'label.mask' key.")


def seg_confmat_output_transform(output):
    """
    Transform output for Ignite ConfusionMatrix (segmentation).
    Converts logits to probabilities with 2 channels for binary, or softmax for multiclass.
    Expects output: list of dicts, each with keys 'pred' and 'label'.
      - sample['pred'][1]: logits, shape [C, H, W] (C=1 or 2)
      - sample['label']['mask']: mask, shape [1, H, W] or [H, W]
    Returns:
        probs: [B, 2, H, W] (float, probabilities)
        mask:  [B, H, W]    (long, ground truth)
    """
    print("[DEBUG] seg_output_transform CALLED. type(output):", type(output))
    if not (isinstance(output, list) and all(isinstance(x, dict) for x in output)):
        raise ValueError("seg_confmat_output_transform expected list of dicts")

    probs_list = []
    masks_list = []
    for sample in output:
        seg_logits = sample['pred'][1]  # [C, H, W]
        # Handle binary case
        if seg_logits.shape[0] == 1:
            prob_fg = torch.sigmoid(seg_logits)              # [1, H, W]
            prob_bg = 1.0 - prob_fg                         # [1, H, W]
            probs = torch.cat([prob_bg, prob_fg], dim=0)     # [2, H, W]
        else:  # Multiclass
            probs = torch.softmax(seg_logits, dim=0)         # [C, H, W]
            assert probs.shape[0] >= 2, "Expected at least 2 channels for multiclass"
        probs_list.append(probs.unsqueeze(0))                # [1, C, H, W]

        mask = sample['label']['mask']                       # [1, H, W] or [H, W]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        masks_list.append(mask.unsqueeze(0))                 # [1, H, W]

    probs_batch = torch.cat(probs_list, dim=0)               # [B, C, H, W]
    masks_batch = torch.cat(masks_list, dim=0).long()        # [B, H, W]

    print("[DEBUG] seg_confmat_output_transform: probs_batch shape:", probs_batch.shape, "masks_batch shape:", masks_batch.shape)
    print("[DEBUG] seg_confmat_output_transform: probs_batch dtype:", probs_batch.dtype, "masks_batch dtype:", masks_batch.dtype)

    return probs_batch, masks_batch


# def seg_confmat_output_transform(output):
#     """
#     Output transform for segmentation confusion matrix metrics (Ignite expects):
#       - probs: [B, num_classes, H, W] (float)
#       - true_mask: [B, H, W] (long)
#     """
#     # Use the canonical extraction
#     seg_logits, true_mask = seg_output_transform(output)  # shapes: [B, C, H, W] or [B, H, W]

#     # Ensure seg_logits has 4 dims [B, C, H, W]
#     if seg_logits.ndim == 3:
#         seg_logits = seg_logits.unsqueeze(1)  # [B, 1, H, W]
#     # For binary: [B, 1, H, W] -> [B, 2, H, W] (background/foreground)
#     if seg_logits.shape[1] == 1:
#         probs_bg = 1 - torch.sigmoid(seg_logits)
#         probs_fg = torch.sigmoid(seg_logits)
#         probs = torch.cat([probs_bg, probs_fg], dim=1)  # [B, 2, H, W]
#     else:
#         probs = torch.softmax(seg_logits, dim=1)  # [B, C, H, W]

#     # Remove channel dim from mask if needed
#     if true_mask.ndim == 4 and true_mask.shape[1] == 1:
#         true_mask = true_mask.squeeze(1)
#     print("[DEBUG] seg_confmat_output_transform: probs.shape:", probs.shape, "true_mask.shape:", true_mask.shape)
#     return probs, true_mask.long()


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


# Multitask Loss
# def multitask_loss(y_pred, y_true):
#     """Handles both dict (training) and tensor (metric) cases."""
#     # If y_true is a dict (from training step)
#     if isinstance(y_true, dict):
#         class_logits, seg_out = None, None
#         if isinstance(y_pred, tuple):
#             if len(y_pred) == 2:
#                 class_logits, seg_out = y_pred
#             elif len(y_pred) == 1:
#                 class_logits, seg_out = y_pred[0], None
#         elif isinstance(y_pred, dict):
#             class_logits = y_pred.get("label", None)
#             seg_out = y_pred.get("mask", None)
#         else:
#             class_logits, seg_out = y_pred, None

#         loss = 0.0
#         if "label" in y_true and class_logits is not None:
#             loss += get_classification_metrics()["loss"](class_logits, y_true["label"])
#         if "mask" in y_true and seg_out is not None:
#             loss += get_segmentation_metrics()["loss"](seg_out, y_true["mask"])
#         if loss == 0.0:
#             raise ValueError(f"No valid targets found in y_true: keys={list(y_true.keys())}")
#         return loss
#     # If y_true is a tensor (from Ignite metric)
#     elif torch.is_tensor(y_true):
#         # Assume classification-only, match y_pred shape
#         return get_classification_metrics()["loss"](y_pred, y_true)
#     else:
#         raise TypeError(f"Unsupported y_true type for multitask_loss: {type(y_true)}")
