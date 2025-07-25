# metrics_utils.py
import torch
import logging
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, ROC_AUC, DiceCoefficient, JaccardIndex
from eval_utils import get_classification_metrics, get_segmentation_metrics


# Output Transforms
def cls_output_transform(output):
    """
    Extract classification logits and labels (both as tensors).
    Returns: (logits, labels) for metrics; labels must be torch.long
    """
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        logits = torch.stack([s['pred'][0] for s in output])
        labels = torch.tensor(
            [s['label'].get('label') for s in output if 'label' in s['label']],
            dtype=torch.long, device=logits.device
        )
        if labels.shape[0] == 0:
            raise ValueError("cls_output_transform: No labels found in batch!")
        return logits, labels
    raise ValueError("cls_output_transform expects a list of dicts with 'pred' and 'label' keys.")


def auc_output_transform(output):
    """
    Extract probabilities and labels for AUROC computation.
    Returns: (probs, labels) where both are tensors.
    Skips samples without label.
    """
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        device = torch.as_tensor(output[0]['pred'][0]).device
        probs, labels = [], []
        for s in output:
            label_dict = s.get('label', {})
            if 'label' not in label_dict:
                continue
            logits = torch.as_tensor(s['pred'][0]).to(device)
            # [batch, num_classes] or [num_classes]
            if logits.ndim == 1:
                prob = torch.softmax(logits, dim=0)[1]
            elif logits.ndim == 2:
                prob = torch.softmax(logits, dim=1)[0, 1]
            else:
                raise ValueError(f"auc_output_transform: Invalid logits shape: {logits.shape}")
            probs.append(prob.unsqueeze(0))
            labels.append(torch.tensor([label_dict["label"]], dtype=torch.long, device=device))
        if not labels:
            logging.error(f"[auc_output_transform] No labels found in batch! Output: {output}")
            raise ValueError("auc_output_transform found no labels in batch for AUROC metric.")
        return torch.cat(probs, dim=0), torch.cat(labels, dim=0)
    raise ValueError("auc_output_transform expects a list of dicts")


def seg_output_transform(output):
    """
    Extract predicted and true segmentation masks for segmentation loss.
    Returns: (pred_labels, true_masks) -- both long tensors, shape [B, H, W]
    """
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        preds, masks = [], []
        for s in output:
            seg_logits = s['pred'][1]
            # Binary vs multiclass
            if seg_logits.shape[0] == 1:
                # Binary, output is [1, H, W]; sigmoid, threshold at 0.5
                pred_labels = (torch.sigmoid(seg_logits) > 0.5).long().squeeze(0)
            else:
                # Multiclass: [C, H, W]; argmax over C
                pred_labels = torch.argmax(seg_logits, dim=0).long()
            preds.append(pred_labels.unsqueeze(0))  # [1, H, W]
            mask = s['label']['mask']
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            masks.append(mask.long().unsqueeze(0))
        pred_tensor = torch.cat(preds, dim=0)   # [B, H, W]
        true_tensor = torch.cat(masks, dim=0)   # [B, H, W]
        return pred_tensor, true_tensor
    raise ValueError("seg_output_transform expects a list of dicts")


def seg_output_transform_for_confmat(output):
    """ Returns raw logits/probabilities and ground truth mask for ConfusionMatrix. """
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        preds, masks = [], []
        for s in output:
            seg_logits = s['pred'][1] if isinstance(s.get('pred'), (tuple, list)) and len(s['pred']) > 1 else None
            mask = s.get('label', {}).get('mask', None)
            if seg_logits is not None and mask is not None:
                # For binary, expand to 2 channels if needed
                if seg_logits.shape[0] == 1:
                    prob = torch.cat([1 - torch.sigmoid(seg_logits), torch.sigmoid(seg_logits)], dim=0)
                else:
                    prob = torch.softmax(seg_logits, dim=0)
                preds.append(prob.unsqueeze(0))  # [1, 2, H, W]
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                masks.append(mask.long().unsqueeze(0))
        if not preds or not masks:
            logging.error(f"[seg_output_transform_for_confmat] No masks found in batch! Output: {output}")
            raise ValueError("No masks found in batch for segmentation metric.")
        pred_tensor = torch.cat(preds, dim=0)   # [B, 2, H, W]
        mask_tensor = torch.cat(masks, dim=0)   # [B, H, W], long
        return pred_tensor, mask_tensor
    raise ValueError("seg_output_transform_for_confmat expects a list of dicts")


# def seg_output_transform_for_confmat(output):
#     """
#     Returns predicted *class* labels and ground truth mask for ConfusionMatrix.
#     Both must be long tensors, shape [B, H, W].
#     """
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         preds, masks = [], []
#         for s in output:
#             seg_logits = s['pred'][1] if isinstance(s.get('pred'), (tuple, list)) and len(s['pred']) > 1 else None
#             mask = s.get('label', {}).get('mask', None)
#             if seg_logits is not None and mask is not None:
#                 # If seg_logits shape [C, H, W]
#                 if seg_logits.shape[0] == 1:
#                     # Binary: shape [1, H, W]
#                     pred_labels = (torch.sigmoid(seg_logits) > 0.5).long().squeeze(0)  # [H, W]
#                 else:
#                     # Multiclass: shape [C, H, W]
#                     pred_labels = torch.argmax(seg_logits, dim=0).long()  # [H, W]
#                 preds.append(pred_labels.unsqueeze(0))  # [1, H, W]
#                 if mask.ndim == 3 and mask.shape[0] == 1:
#                     mask = mask.squeeze(0)
#                 masks.append(mask.long().unsqueeze(0))  # [1, H, W]
#         if not preds or not masks:
#             logging.error(f"[seg_output_transform_for_confmat] No masks found in batch! Output: {output}")
#             raise ValueError("No masks found in batch for segmentation metric.")
#         pred_tensor = torch.cat(preds, dim=0)   # [B, H, W], long
#         mask_tensor = torch.cat(masks, dim=0)   # [B, H, W], long
#         return pred_tensor, mask_tensor
#     raise ValueError("seg_output_transform_for_confmat expects a list of dicts")


# Key Presence Utility
def _validate_key_in_dataset(loader, key, num_batches=3):
    """Return True if key is present in any of the first num_batches of loader."""
    try:
        checked = 0
        for batch in loader:
            # batch: dict or tuple/list of dicts
            if isinstance(batch, dict) and key in batch:
                return True
            if isinstance(batch, (tuple, list)):
                for entry in batch:
                    if isinstance(entry, dict) and key in entry:
                        return True
            checked += 1
            if checked >= num_batches:
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
    """
    task = config.get("task", "multitask") if config else "multitask"

    has_label = (val_loader is None) or _validate_key_in_dataset(val_loader, "label")
    has_mask = (val_loader is None) or _validate_key_in_dataset(val_loader, "mask")

    # Attach classification metrics
    if task in ("classification", "multitask") and cls_output_transform is not None and has_label:
        Accuracy(output_transform=cls_output_transform).attach(evaluator, "val_acc")
        ROC_AUC(output_transform=auc_output_transform or cls_output_transform).attach(evaluator, "val_auc")
        Loss(
            loss_fn=get_classification_metrics()["loss"],
            output_transform=cls_output_transform
        ).attach(evaluator, "val_loss")
        ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform).attach(evaluator, "val_cls_confmat")
        logging.info("Attached classification metrics: val_acc, val_auc, val_loss, val_cls_confmat")
    elif task in ("classification", "multitask"):
        logging.warning("No label data or output_transform found. Classification metrics not attached.")

    # Attach segmentation metrics
    if task in ("segmentation", "multitask") and seg_output_transform_for_metrics is not None and has_mask:
        cm = ConfusionMatrix(num_classes=num_classes, output_transform=seg_output_transform_for_metrics)
        # debug_transform = debug_output_transform(seg_output_transform_for_metrics)
        # cm = ConfusionMatrix(num_classes=num_classes, output_transform=debug_transform)
        DiceCoefficient(cm).attach(evaluator, "val_dice")
        JaccardIndex(cm).attach(evaluator, "val_iou")
        logging.info("Attached segmentation metrics: val_dice, val_iou")
    elif task in ("segmentation", "multitask"):
        logging.warning("No mask data or output_transform found. Segmentation metrics not attached.")


# Multitask Loss
def multitask_loss(y_pred, y_true):
    """
    Combined segmentation + classification loss for multitask, segmentation-only, or classification-only tasks.
    Accepts:
        - y_pred: tuple or dict, model outputs (e.g., (class_logits, seg_out) or just logits/mask)
        - y_true: dict, must include 'label' and/or 'mask' as appropriate
    """
    if isinstance(y_pred, tuple):
        if len(y_pred) == 2:
            class_logits, seg_out = y_pred
        elif len(y_pred) == 1:
            class_logits, seg_out = y_pred[0], None
        else:
            class_logits, seg_out = y_pred, None
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


# Fixes
# For segmentation output transforms (especially for ConfusionMatrix):
# - Always return tensors of type torch.long, representing integer class indices.
# - Do not return probability maps (float) to Ignite's confusion-based metrics.
# Add .long() everywhere masks/class predictions are computed.
# For binary segmentation:
# - Use sigmoid + threshold + .long().
# For multiclass segmentation:
# - Use softmax + argmax + .long().

# import torch
# import logging
# from ignite.metrics import Accuracy, Loss, ConfusionMatrix, ROC_AUC, DiceCoefficient, JaccardIndex
# from eval_utils import get_classification_metrics, get_segmentation_metrics


# # Output Transforms
# def cls_output_transform(output):
#     """Extract classification logits and labels, defensively."""
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         logits = torch.stack([s['pred'][0] for s in output])
#         labels = torch.tensor(
#             [s['label'].get('label') for s in output if 'label' in s['label']],
#             dtype=torch.long, device=logits.device
#         )
#         if labels.shape[0] == 0:
#             raise ValueError("cls_output_transform: No labels found in batch!")
#         return logits, labels
#     raise ValueError("cls_output_transform expects a list of dicts with 'pred' and 'label' keys.")


# def auc_output_transform(output):
#     """Extract probabilities and labels for AUROC computation. Skips samples without label."""
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         device = torch.as_tensor(output[0]['pred'][0]).device
#         probs, labels = [], []
#         for s in output:
#             # Only proceed if label is present
#             label_dict = s.get('label', {})
#             if 'label' not in label_dict:
#                 continue
#             logits = torch.as_tensor(s['pred'][0]).to(device)
#             if logits.ndim == 1:
#                 prob = torch.softmax(logits, dim=0)[1]
#             elif logits.ndim == 2:
#                 prob = torch.softmax(logits, dim=1)[0, 1]
#             else:
#                 raise ValueError(f"auc_output_transform: Invalid logits shape: {logits.shape}")
#             probs.append(prob.unsqueeze(0))
#             labels.append(torch.tensor([label_dict["label"]], dtype=torch.long, device=device))
#         if not labels:
#             logging.error(f"[auc_output_transform] No labels found in batch! Output: {output}")
#             raise ValueError("auc_output_transform found no labels in batch for AUROC metric.")
#         return torch.cat(probs, dim=0), torch.cat(labels, dim=0)
#     raise ValueError("auc_output_transform expects a list of dicts")


# def seg_output_transform(output):
#     """Extract predicted and true segmentation masks. Assumes masks always present."""
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         preds, masks = [], []
#         for s in output:
#             seg_logits = s['pred'][1]
#             # Binary vs multiclass
#             if seg_logits.shape[0] == 1:
#                 prob = torch.cat([1 - torch.sigmoid(seg_logits), torch.sigmoid(seg_logits)], dim=0)
#             else:
#                 prob = torch.softmax(seg_logits, dim=0)
#             preds.append(prob.unsqueeze(0))  # [1, C, H, W]
#             mask = s['label']['mask'].long()
#             if mask.ndim == 3 and mask.shape[0] == 1:
#                 mask = mask.squeeze(0)
#             masks.append(mask.unsqueeze(0))
#         pred_logits = torch.cat(preds, dim=0)   # [B, C, H, W]
#         pred_labels = torch.argmax(pred_logits, dim=1)  # [B, H, W]
#         true_masks = torch.cat(masks, dim=0)    # [B, H, W]
#         return pred_labels, true_masks
#     raise ValueError("seg_output_transform expects a list of dicts")


# def seg_output_transform_for_confmat(output):
#     """Return raw probability maps and masks for Ignite ConfusionMatrix."""
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         preds, masks = [], []
#         for s in output:
#             seg_logits = s['pred'][1] if isinstance(s.get('pred'), (tuple, list)) and len(s['pred']) > 1 else None
#             mask = s.get('label', {}).get('mask', None)
#             if seg_logits is not None and mask is not None:
#                 if seg_logits.shape[0] == 1:
#                     prob = torch.cat([1 - torch.sigmoid(seg_logits), torch.sigmoid(seg_logits)], dim=0)
#                 else:
#                     prob = torch.softmax(seg_logits, dim=0)
#                 preds.append(prob.unsqueeze(0))
#                 if mask.ndim == 3 and mask.shape[0] == 1:
#                     mask = mask.squeeze(0)
#                 masks.append(mask.unsqueeze(0))
#         if not preds or not masks:
#             logging.error(f"[seg_output_transform_for_confmat] No masks found in batch! Output: {output}")
#             raise ValueError("No masks found in batch for segmentation metric.")
#         pred_tensor = torch.cat(preds, dim=0)
#         mask_tensor = torch.cat(masks, dim=0)
#         return pred_tensor, mask_tensor
#     raise ValueError("seg_output_transform_for_confmat expects a list of dicts")


# # Key Presence Utility
# def _validate_key_in_dataset(loader, key, num_batches=3):
#     """Return True if key is present in any of the first num_batches of loader."""
#     try:
#         checked = 0
#         for batch in loader:
#             # batch: dict or tuple/list of dicts
#             if isinstance(batch, dict) and key in batch:
#                 return True
#             if isinstance(batch, (tuple, list)):
#                 for entry in batch:
#                     if isinstance(entry, dict) and key in entry:
#                         return True
#             checked += 1
#             if checked >= num_batches:
#                 break
#     except Exception as e:
#         logging.warning(f"Could not inspect loader for key '{key}': {e}")
#     return False


# # Attach Metrics
# def attach_metrics(
#     evaluator,
#     config=None,
#     seg_output_transform_for_metrics=None,
#     cls_output_transform=None,
#     auc_output_transform=None,
#     num_classes=2,
#     val_loader=None
# ):
#     """
#     Attach classification and/or segmentation metrics to evaluator based on task,
#     available output transforms, and actual presence of label/mask in validation loader.
#     """
#     task = config.get("task", "multitask") if config else "multitask"

#     has_label = (val_loader is None) or _validate_key_in_dataset(val_loader, "label")
#     has_mask = (val_loader is None) or _validate_key_in_dataset(val_loader, "mask")

#     # Attach classification metrics
#     if task in ("classification", "multitask") and cls_output_transform is not None and has_label:
#         Accuracy(output_transform=cls_output_transform).attach(evaluator, "val_acc")
#         ROC_AUC(output_transform=auc_output_transform or cls_output_transform).attach(evaluator, "val_auc")
#         Loss(
#             loss_fn=get_classification_metrics()["loss"],
#             output_transform=cls_output_transform
#         ).attach(evaluator, "val_loss")
#         ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform).attach(evaluator, "val_cls_confmat")
#         logging.info("Attached classification metrics: val_acc, val_auc, val_loss, val_cls_confmat")
#     elif task in ("classification", "multitask"):
#         logging.warning("No label data or output_transform found. Classification metrics not attached.")

#     # Attach segmentation metrics
#     if task in ("segmentation", "multitask") and seg_output_transform_for_metrics is not None and has_mask:
#         cm = ConfusionMatrix(num_classes=num_classes, output_transform=seg_output_transform_for_metrics)
#         DiceCoefficient(cm).attach(evaluator, "val_dice")
#         JaccardIndex(cm).attach(evaluator, "val_iou")
#         logging.info("Attached segmentation metrics: val_dice, val_iou")
#     elif task in ("segmentation", "multitask"):
#         logging.warning("No mask data or output_transform found. Segmentation metrics not attached.")


# # Multitask Loss
# def multitask_loss(y_pred, y_true):
#     """
#     Combined segmentation + classification loss for multitask, segmentation-only, or classification-only tasks.
#     Accepts:
#         - y_pred: tuple or dict, model outputs (e.g., (class_logits, seg_out) or just logits/mask)
#         - y_true: dict, must include 'label' and/or 'mask' as appropriate
#     """
#     # Support both tuple and dict outputs
#     if isinstance(y_pred, tuple):
#         if len(y_pred) == 2:
#             class_logits, seg_out = y_pred
#         elif len(y_pred) == 1:
#             class_logits, seg_out = y_pred[0], None
#         else:
#             class_logits, seg_out = y_pred, None
#     elif isinstance(y_pred, dict):
#         class_logits = y_pred.get("label", None)
#         seg_out = y_pred.get("mask", None)
#     else:
#         class_logits, seg_out = y_pred, None

#     loss = 0.0
#     if "label" in y_true and class_logits is not None:
#         loss += get_classification_metrics()["loss"](class_logits, y_true["label"])
#     if "mask" in y_true and seg_out is not None:
#         loss += get_segmentation_metrics()["loss"](seg_out, y_true["mask"])
#     if loss == 0.0:
#         raise ValueError(f"No valid targets found in y_true: keys={list(y_true.keys())}")
#     return loss


# import torch
# import logging
# from ignite.metrics import Accuracy, Loss, ConfusionMatrix
# from ignite.metrics import ROC_AUC
# from ignite.metrics import DiceCoefficient, JaccardIndex
# from eval_utils import get_classification_metrics, get_segmentation_metrics


# # Output Transforms
# def cls_output_transform(output):
#     """Extract classification logits and labels."""
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         logits = torch.stack([s['pred'][0] for s in output])  # [B, C]
#         labels = torch.tensor([s['label']['label'] for s in output], dtype=torch.long, device=logits.device)
#         return logits, labels
#     raise ValueError("cls_output_transform expects a list of dicts with 'pred' and 'label' keys")


# def auc_output_transform(output):
#     """
#     Extract probabilities and labels for AUROC computation.
#     Returns (probs, labels) only if all labels are present, else raises ValueError.
#     """
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         device = torch.as_tensor(output[0]['pred'][0]).device
#         probs, labels = [], []
#         for s in output:
#             # Defensive: skip if label is missing
#             if "label" not in s or "label" not in s["label"]:
#                 continue
#             logits = torch.as_tensor(s['pred'][0]).to(device)
#             if logits.ndim == 1:
#                 prob = torch.softmax(logits, dim=0)[1]
#             elif logits.ndim == 2:
#                 prob = torch.softmax(logits, dim=1)[0, 1]
#             else:
#                 raise ValueError(f"Invalid logits shape: {logits.shape}")
#             probs.append(prob.unsqueeze(0))
#             labels.append(torch.tensor([s['label']['label']], dtype=torch.long, device=device))
#         if not labels:
#             raise ValueError("auc_output_transform found no labels in batch for AUROC metric.")
#         return torch.cat(probs, dim=0), torch.cat(labels, dim=0)
#     raise ValueError("auc_output_transform expects a list of dicts")


# def seg_output_transform(output):
#     """Extract predicted and true segmentation masks."""
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         preds, masks = [], []
#         for s in output:
#             seg_logits = s['pred'][1]
#             if seg_logits.shape[0] == 1:
#                 prob = torch.cat([1 - torch.sigmoid(seg_logits), torch.sigmoid(seg_logits)], dim=0)
#             else:
#                 prob = torch.softmax(seg_logits, dim=0)
#             preds.append(prob.unsqueeze(0))  # [1, 2, H, W]
#             mask = s['label']['mask'].long()
#             if mask.ndim == 3 and mask.shape[0] == 1:
#                 mask = mask.squeeze(0)
#             masks.append(mask.unsqueeze(0))
#         pred_logits = torch.cat(preds, dim=0)   # [B, 2, H, W]
#         pred_labels = torch.argmax(pred_logits, dim=1)  # [B, H, W]
#         true_masks = torch.cat(masks, dim=0)    # [B, H, W]
#         return pred_labels, true_masks
#     raise ValueError("seg_output_transform expects a list of dicts")


# def seg_output_transform_for_confmat(output):
#     """
#     Return raw probability maps and masks for Ignite ConfusionMatrix.
#     Handles missing masks gracefully and supports both binary and multiclass.
#     """
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         preds, masks = [], []
#         for s in output:
#             # Safely extract segmentation logits
#             seg_logits = None
#             if isinstance(s.get('pred', None), (tuple, list)) and len(s['pred']) > 1:
#                 seg_logits = s['pred'][1]
#             # Safely extract mask
#             label_dict = s.get('label', {})
#             mask = label_dict.get('mask', None)

#             if seg_logits is not None and mask is not None:
#                 # Binary or multiclass softmax
#                 if seg_logits.shape[0] == 1:
#                     prob = torch.cat([1 - torch.sigmoid(seg_logits), torch.sigmoid(seg_logits)], dim=0)
#                 else:
#                     prob = torch.softmax(seg_logits, dim=0)
#                 preds.append(prob.unsqueeze(0))  # [1, C, H, W]
#                 if mask.ndim == 3 and mask.shape[0] == 1:
#                     mask = mask.squeeze(0)
#                 masks.append(mask.unsqueeze(0))
#         if not preds or not masks:
#             raise ValueError("No masks found in batch for segmentation metric.")
#         pred_tensor = torch.cat(preds, dim=0)   # [B, C, H, W]
#         mask_tensor = torch.cat(masks, dim=0)   # [B, H, W]
#         return pred_tensor, mask_tensor

#     raise ValueError("seg_output_transform_for_confmat expects a list of dicts")


# # Multitask Loss
# def multitask_loss(y_pred, y_true):
#     """
#     Combined segmentation + classification loss for multitask, segmentation-only, or classification-only tasks.
#     Accepts:
#         - y_pred: tuple or dict, model outputs (e.g., (class_logits, seg_out) or just logits/mask)
#         - y_true: dict, must include 'label' and/or 'mask' as appropriate
#     """
#     # Support both tuple and dict outputs
#     if isinstance(y_pred, tuple):
#         # Assume (class_logits, seg_out) for multitask; may be single output for single task
#         if len(y_pred) == 2:
#             class_logits, seg_out = y_pred
#         elif len(y_pred) == 1:
#             class_logits, seg_out = y_pred[0], None
#         else:
#             class_logits, seg_out = y_pred, None
#     elif isinstance(y_pred, dict):
#         class_logits = y_pred.get("label", None)
#         seg_out = y_pred.get("mask", None)
#     else:
#         class_logits, seg_out = y_pred, None

#     loss = 0.0
#     # Classification loss
#     if "label" in y_true and class_logits is not None:
#         loss_cls = get_classification_metrics()["loss"](class_logits, y_true["label"])
#         loss += loss_cls
#     # Segmentation loss
#     if "mask" in y_true and seg_out is not None:
#         loss_seg = get_segmentation_metrics()["loss"](seg_out, y_true["mask"])
#         loss += loss_seg

#     if loss == 0.0:
#         raise ValueError(f"No valid targets found in y_true: keys={list(y_true.keys())}")
#     return loss


# def _validate_key_in_dataset(loader, key, num_batches=3):
#     """Check first few batches for presence of key."""
#     try:
#         checked = 0
#         for batch in loader:
#             # Handle dict
#             if isinstance(batch, dict) and key in batch:
#                 return True
#             # Handle tuple/list (e.g., (inputs, targets))
#             if isinstance(batch, (tuple, list)):
#                 for entry in batch:
#                     if isinstance(entry, dict) and key in entry:
#                         return True
#             checked += 1
#             if checked >= num_batches:
#                 break
#     except Exception as e:
#         logging.warning(f"Could not inspect loader for key '{key}': {e}")
#     return False


# def attach_metrics(
#     evaluator,
#     config=None,
#     seg_output_transform_for_metrics=None,
#     cls_output_transform=None,
#     auc_output_transform=None,
#     num_classes=2,
#     val_loader=None
# ):
#     """
#     Attach classification and/or segmentation metrics to evaluator based on task,
#     available output transforms, and actual presence of label/mask in validation loader.
#     """
#     task = config["task"] if config and "task" in config else "multitask"

#     # Check for label key in validation data
#     has_label = (val_loader is None) or _validate_key_in_dataset(val_loader, "label")
#     # Check for mask key in validation data
#     has_mask = (val_loader is None) or _validate_key_in_dataset(val_loader, "mask")

#     # Classification metrics (require label)
#     attach_class_metrics = (
#         task in ("classification", "multitask") and cls_output_transform is not None and has_label
#     )
#     if attach_class_metrics:
#         Accuracy(output_transform=cls_output_transform).attach(evaluator, "val_acc")
#         ROC_AUC(output_transform=auc_output_transform or cls_output_transform).attach(evaluator, "val_auc")
#         Loss(
#             loss_fn=get_classification_metrics()["loss"],
#             output_transform=cls_output_transform
#         ).attach(evaluator, "val_loss")
#         ConfusionMatrix(num_classes=num_classes, output_transform=cls_output_transform).attach(evaluator, "val_cls_confmat")
#         logging.info("Attached classification metrics: val_acc, val_auc, val_loss, val_cls_confmat")
#     elif task in ("classification", "multitask"):
#         logging.warning("No label data or output_transform found. Classification metrics not attached.")

#     # Segmentation metrics (require mask)
#     attach_seg_metrics = (
#         task in ("segmentation", "multitask") and seg_output_transform_for_metrics is not None and has_mask
#     )
#     if attach_seg_metrics:
#         cm = ConfusionMatrix(num_classes=num_classes, output_transform=seg_output_transform_for_metrics)
#         DiceCoefficient(cm).attach(evaluator, "val_dice")
#         JaccardIndex(cm).attach(evaluator, "val_iou")
#         logging.info("Attached segmentation metrics: val_dice, val_iou")
#     elif task in ("segmentation", "multitask"):
#         logging.warning("No mask data or output_transform found. Segmentation metrics not attached.")
