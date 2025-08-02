# metrics_utils.py
import torch
import logging
from eval_utils import get_classification_metrics, get_segmentation_metrics


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

    # if isinstance(y, dict):
    #     for k in ("label", "classification", "class"):
    #         if k in y:
    #             y = y[k]
    #             break
    #     else:
    #         raise ValueError(f"Could not extract tensor from y dict: {y}")

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
    """For binary ROC AUC: returns (prob_class1, true_class)"""
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        logits = torch.stack([x['pred'][0] for x in output])   # [B, 2]
        labels = torch.tensor(
            [x['label'].get('label') for x in output if 'label' in x['label']],
            dtype=torch.long, device=logits.device
        )
        # Convert logits to probability for class 1
        probs = torch.softmax(logits, dim=1)   # [B, 2]
        prob_class1 = probs[:, 1]              # [B]
        labels = labels.view(-1)
        return prob_class1, labels
    raise ValueError("auc_output_transform expects a list of dicts with 'pred' and 'label' keys.")


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
