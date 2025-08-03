# metrics_utils.py
import torch
import logging
from eval_utils import get_classification_metrics, get_segmentation_metrics
import os


def _debug_enabled():
    return os.environ.get("DEBUG_OUTPUT_TRANSFORMS", "0") in ("1", "true", "True", "yes", "YES")


# Output Transforms for Accuracy, ConfusionMatrix
def cls_output_transform(output):
    import torch
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

    return y_pred, y


# for ROC AUC
def auc_output_transform(output):
    """
    Extracts (prob_class1, class_idx) from model output for ROC_AUC metric.
    Handles tuple/list, dict, and nested dict structures robustly.
    Hardened: prints input and fails clearly if label is missing or None.
    """
    import torch

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

            #     if isinstance(logit, torch.Tensor):
            #         # Already a tensor; nothing to do
            #         logits_list[i] = logit
            #     elif isinstance(logit, (list, tuple)):
            #         # List or tuple of tensors or numbers
            #         if all(isinstance(xx, torch.Tensor) for xx in logit):
            #             # Stack if all are tensors
            #             if len(logit) == 1:
            #                 logits_list[i] = logit[0]
            #             else:
            #                 try:
            #                     logits_list[i] = torch.stack(logit)
            #                 except Exception:
            #                     print(f"[ERROR] Could not stack logit at idx {i}: {logit}")
            #                     raise
            #         else:
            #             logits_list[i] = torch.as_tensor(logit)
            #     else:
            #         logits_list[i] = torch.as_tensor(logit)

            #     # If still not 1D or 2D, raise error
            #     if logit.ndim not in (1, 2):
            #         print(f"[ERROR] Unexpected logit shape at idx {i}: {logit.shape}")
            #         raise ValueError(f"Logit at index {i} is not 1D/2D: shape {logit.shape}")

            #     print(f"[DEBUG] logit[{i}] type: {type(logit)}, value: {logit}, shape: {getattr(logit, 'shape', None)}")

            shapes = [logit.shape for logit in logits_list]
            if _debug_enabled():
                print("[DEBUG] Final logits_list shapes before stack:", shapes)
            if len(set(shapes)) != 1:
                if _debug_enabled():
                    print("[ERROR] Inconsistent shapes in logits_list:", shapes)
                raise ValueError(f"Inconsistent shapes in logits_list: {shapes}")

            # DEBUG:
            # for i, logit in enumerate(logits_list):
            #     print(f"[DEBUG] logits_list[{i}]: type={type(logit)}, value={logit}, shape={getattr(logit, 'shape', None)}")
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


# def auc_output_transform(output):
#     """
#     Extracts (prob_class1, class_idx) from model output for ROC_AUC metric.
#     Handles tuple/list, dict, multitask (tuple/list) and nested dict structures robustly.
#     """
#     import torch

#     def extract_logits(obj):
#         # For multitask (tuple/list): pick first (classification) head
#         if isinstance(obj, (tuple, list)):
#             return extract_logits(obj[0])
#         # For dict: unwrap by known keys recursively
#         while isinstance(obj, dict):
#             for k in ("logits", "pred", "output", "y_pred"):
#                 if k in obj:
#                     obj = obj[k]
#                     break
#             else:
#                 raise ValueError(f"Cannot extract tensor from dict: {obj}")
#         return obj

#     def extract_label(obj):
#         # For dict: unwrap by known keys recursively
#         while isinstance(obj, dict):
#             for k in ("label", "classification", "class", "value"):
#                 if k in obj:
#                     obj = obj[k]
#                     break
#             else:
#                 raise ValueError(f"Cannot extract label tensor from dict: {obj}")
#         return obj

#     # Handle list of dicts (DataLoader collate style)
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         logits_list = []
#         labels_list = []
#         for x in output:
#             logits = extract_logits(x.get("pred") or x.get("logits"))
#             labels = extract_label(x.get("label") or x.get("y") or x.get("classification"))
#             logits_list.append(logits)
#             labels_list.append(labels)
#         # DEBUG:
#         print("DEBUG: labels_list", labels_list)

#         # Ensure all logits are tensors
#         logits = torch.stack([logit if isinstance(logit, torch.Tensor) else torch.as_tensor(logit) for logit in logits_list])
#         labels = torch.tensor(labels_list, dtype=torch.long, device=logits.device)
#     # Handle tuple/list (modern, batched): output = (y_pred, y)
#     elif isinstance(output, (tuple, list)) and len(output) >= 2:
#         logits = extract_logits(output[0])
#         labels = extract_label(output[1])
#     # Handle dict (single sample)
#     elif isinstance(output, dict):
#         logits = extract_logits(output.get("y_pred") or output.get("pred") or output.get("logits"))
#         labels = extract_label(output.get("y") or output.get("label") or output.get("classification"))
#     else:
#         raise ValueError(f"Unexpected output type in auc_output_transform: {type(output)}")

#     # DEBUG:
#     # print(f"DEBUG AUC: logits shape {logits.shape}, labels shape {labels.shape}")
#     # print(f"DEBUG AUC: logits {logits}, labels {labels}")

#     # Make sure both logits and labels are tensors
#     if not isinstance(logits, torch.Tensor):
#         logits = torch.as_tensor(logits)
#     if not isinstance(labels, torch.Tensor):
#         labels = torch.as_tensor(labels)

#     # Ensure correct shapes
#     if logits.ndim == 1:
#         logits = logits.unsqueeze(0)
#     if labels.ndim == 0:
#         labels = labels.unsqueeze(0)
#     if labels.ndim == 2 and labels.shape[1] > 1:
#         labels = torch.argmax(labels, dim=1)

#     # For ROC AUC (binary): use probability of class 1
#     if logits.shape[1] == 2:
#         probs = torch.softmax(logits, dim=1)
#         prob_class1 = probs[:, 1]
#     elif logits.shape[1] == 1:
#         prob_class1 = torch.sigmoid(logits.squeeze(1))
#     else:
#         prob_class1 = torch.softmax(logits, dim=1)  # multiclass

#     return prob_class1, labels


def seg_output_transform(output):
    """
    Extracts predicted and true segmentation masks for segmentation metrics.
    Returns:
        - pred_masks: [B, 1, H, W] or [B, H, W]
        - true_masks: [B, 1, H, W] or [B, H, W]
    """
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        pred_masks, true_masks = [], []
        for i, x in enumerate(output):
            # Check nested structure for pred[1] and label["mask"]
            if not isinstance(x.get("pred", None), (list, tuple)) or len(x["pred"]) < 2:
                raise ValueError(f"seg_output_transform: Missing or malformed 'pred' at index {i}: {x.get('pred')}")
            if "label" not in x or "mask" not in x["label"]:
                raise ValueError(f"seg_output_transform: Missing 'label' or 'label.mask' at index {i}: {x.get('label')}")
            pred = x["pred"][1]  # segmentation logits or mask
            mask = x["label"]["mask"]
            # Optionally squeeze singleton channels, if present
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


def seg_output_transform_for_confmat(output):
    # output: list of dicts per sample
    seg_logits = torch.stack([x["pred"][1] for x in output])      # [B, 1, H, W] or [B, num_classes, H, W]
    true_mask = torch.stack([x["label"]["mask"] for x in output])  # [B, 1, H, W] or [B, H, W]

    # For binary: [B, 1, H, W] → [B, 2, H, W] (foreground/background)
    if seg_logits.shape[1] == 1:
        # Use sigmoid for binary, create background channel
        probs_bg = 1 - torch.sigmoid(seg_logits)      # background probability
        probs_fg = torch.sigmoid(seg_logits)           # foreground probability
        probs = torch.cat([probs_bg, probs_fg], dim=1)  # [B, 2, H, W]
    else:
        # Use softmax for multiclass
        probs = torch.softmax(seg_logits, dim=1)

    # Remove channel dim from mask if needed
    if true_mask.ndim == 4 and true_mask.shape[1] == 1:
        true_mask = true_mask.squeeze(1)
    return probs, true_mask.long()


# Key Presence Utility
def _validate_key_in_dataset(loader, key, num_batches=3):
    """Return True if dotted key is present in any of the first num_batches of loader."""
    def get_nested(d, keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d

    keys = key.split('.')
    try:
        for i, batch in enumerate(loader):
            items = batch if isinstance(batch, (list, tuple)) else [batch]
            if any(get_nested(item, keys) is not None for item in items if isinstance(item, dict)):
                return True
            if i + 1 >= num_batches:
                break
    except Exception as e:
        logging.warning(f"Could not inspect loader for key '{key}': {e}")
    return False


def debug_output_transform(orig_transform):
    def wrapped(output):
        y_pred, y = orig_transform(output)
        print("DEBUG:", "y_pred.shape", y_pred.shape, "y_pred.dtype", y_pred.dtype)
        print("DEBUG:", "y.shape", y.shape, "y.dtype", y.dtype)
        return y_pred, y
    return wrapped


# Attach Metrics
def attach_metrics(
    evaluator,
    config=None,
    seg_output_transform_for_metrics=None,
    cls_output_transform=None,
    auc_output_transform=None,
    num_classes=2,
    val_loader=None
):
    """
    Attach classification and/or segmentation metrics to evaluator based on task,
    available output transforms, and actual presence of label/mask in validation loader.
    Ensures all metrics use consistent Wandb keys.
    """
    from ignite.metrics import Accuracy, Loss, ConfusionMatrix, DiceCoefficient, JaccardIndex
    from ignite.metrics.roc_auc import ROC_AUC
    from eval_utils import get_classification_metrics  # get_segmentation_metrics

    task = config.get("task", "multitask") if config else "multitask"
    has_label = (val_loader is None) or _validate_key_in_dataset(val_loader, "label.label")
    has_mask = (val_loader is None) or _validate_key_in_dataset(val_loader, "label.mask")

    # Classification Metrics
    if task in ("classification", "multitask"):
        if cls_output_transform is not None and has_label:
            Accuracy(output_transform=cls_output_transform).attach(evaluator, "val_acc")
            ROC_AUC(output_transform=auc_output_transform or cls_output_transform).attach(evaluator, "val_auc")
            Loss(
                loss_fn=get_classification_metrics()["loss"],
                output_transform=cls_output_transform
            ).attach(evaluator, "val_loss")
            ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform).attach(evaluator, "val_cls_confmat")
            logging.info("Attached classification metrics: val_acc, val_auc, val_loss, val_cls_confmat")
        else:
            logging.warning("No label data or output_transform found. Classification metrics not attached.")

    # Segmentation Metrics
    if task in ("segmentation", "multitask"):
        if seg_output_transform_for_metrics is not None and has_mask:
            cm = ConfusionMatrix(num_classes=num_classes, output_transform=seg_output_transform_for_metrics)
            DiceCoefficient(cm).attach(evaluator, "val_dice")
            JaccardIndex(cm).attach(evaluator, "val_iou")
            logging.info("Attached segmentation metrics: val_dice, val_iou")
        else:
            logging.warning("No mask data or output_transform found. Segmentation metrics not attached.")

    # Optional: Print attached metrics for debugging
    print("[attach_metrics] Final attached metrics:", evaluator.state.metrics.keys() if hasattr(evaluator, 'state') else "not available")


# Multitask Loss
def multitask_loss(y_pred, y_true):
    """Handles both dict (training) and tensor (metric) cases."""
    # If y_true is a dict (from training step)
    if isinstance(y_true, dict):
        class_logits, seg_out = None, None
        if isinstance(y_pred, tuple):
            if len(y_pred) == 2:
                class_logits, seg_out = y_pred
            elif len(y_pred) == 1:
                class_logits, seg_out = y_pred[0], None
        elif isinstance(y_pred, dict):
            class_logits = y_pred.get("label", None)
            seg_out = y_pred.get("mask", None)
        else:
            class_logits, seg_out = y_pred, None

        loss = 0.0
        if "label" in y_true and class_logits is not None:
            loss += get_classification_metrics()["loss"](class_logits, y_true["label"])
        if "mask" in y_true and seg_out is not None:
            loss += get_segmentation_metrics()["loss"](seg_out, y_true["mask"])
        if loss == 0.0:
            raise ValueError(f"No valid targets found in y_true: keys={list(y_true.keys())}")
        return loss
    # If y_true is a tensor (from Ignite metric)
    elif torch.is_tensor(y_true):
        # Assume classification-only, match y_pred shape
        return get_classification_metrics()["loss"](y_pred, y_true)
    else:
        raise TypeError(f"Unsupported y_true type for multitask_loss: {type(y_true)}")
