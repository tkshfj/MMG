# metrics_utils.py
import torch
from ignite.metrics import Accuracy, Loss, ROC_AUC, ConfusionMatrix, DiceCoefficient, JaccardIndex
from eval_utils import get_classification_metrics, get_segmentation_metrics


# Output Transforms
def cls_output_transform(output):
    """Extract classification logits and labels."""
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        logits = torch.stack([s['pred'][0] for s in output])  # [B, C]
        labels = torch.tensor([s['label']['label'] for s in output], dtype=torch.long, device=logits.device)
        return logits, labels
    raise ValueError("cls_output_transform expects a list of dicts with 'pred' and 'label' keys")


def auc_output_transform(output):
    """Extract probabilities and labels for AUROC computation."""
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        device = torch.as_tensor(output[0]['pred'][0]).device
        probs, labels = [], []
        for s in output:
            logits = torch.as_tensor(s['pred'][0]).to(device)
            if logits.ndim == 1:
                prob = torch.softmax(logits, dim=0)[1]
            elif logits.ndim == 2:
                prob = torch.softmax(logits, dim=1)[0, 1]
            else:
                raise ValueError(f"Invalid logits shape: {logits.shape}")
            probs.append(prob.unsqueeze(0))
            labels.append(torch.tensor([s['label']['label']], dtype=torch.long, device=device))
        return torch.cat(probs, dim=0), torch.cat(labels, dim=0)
    raise ValueError("auc_output_transform expects a list of dicts")


def seg_output_transform(output):
    """Extract predicted and true segmentation masks."""
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        preds, masks = [], []
        for s in output:
            seg_logits = s['pred'][1]
            if seg_logits.shape[0] == 1:
                prob = torch.cat([1 - torch.sigmoid(seg_logits), torch.sigmoid(seg_logits)], dim=0)
            else:
                prob = torch.softmax(seg_logits, dim=0)
            preds.append(prob.unsqueeze(0))  # [1, 2, H, W]
            mask = s['label']['mask'].long()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            masks.append(mask.unsqueeze(0))
        pred_logits = torch.cat(preds, dim=0)   # [B, 2, H, W]
        pred_labels = torch.argmax(pred_logits, dim=1)  # [B, H, W]
        true_masks = torch.cat(masks, dim=0)    # [B, H, W]
        return pred_labels, true_masks
    raise ValueError("seg_output_transform expects a list of dicts")


def seg_output_transform_for_confmat(output):
    """Return raw probability maps for Ignite ConfusionMatrix."""
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        preds, masks = [], []
        for s in output:
            seg_logits = s['pred'][1]
            if seg_logits.shape[0] == 1:
                prob = torch.cat([1 - torch.sigmoid(seg_logits), torch.sigmoid(seg_logits)], dim=0)
            else:
                prob = torch.softmax(seg_logits, dim=0)
            preds.append(prob.unsqueeze(0))  # [1, 2, H, W]
            mask = s['label']['mask'].long()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            masks.append(mask.unsqueeze(0))
        pred_tensor = torch.cat(preds, dim=0)   # [B, 2, H, W]
        mask_tensor = torch.cat(masks, dim=0)   # [B, H, W]
        return pred_tensor, mask_tensor
    raise ValueError("seg_output_transform_for_confmat expects a list of dicts")


# Multitask Loss
def multitask_loss(y_pred, y_true):
    """Combined segmentation + classification loss."""
    class_logits, seg_out = y_pred
    labels = y_true['label']
    masks = y_true['mask']
    loss_cls = get_classification_metrics()["loss"](class_logits, labels)
    loss_seg = get_segmentation_metrics()["loss"](seg_out, masks)
    return loss_cls + loss_seg


# Metric Attachment
def attach_metrics(evaluator, config=None, seg_output_transform_for_metrics=None):
    """Attach classification and segmentation metrics to evaluator."""
    # Classification
    Accuracy(output_transform=cls_output_transform).attach(evaluator, "val_acc")
    ROC_AUC(output_transform=auc_output_transform).attach(evaluator, "val_auc")
    Loss(
        loss_fn=get_classification_metrics()["loss"],
        output_transform=cls_output_transform
    ).attach(evaluator, "val_loss")
    ConfusionMatrix(num_classes=2, output_transform=cls_output_transform).attach(evaluator, "val_cls_confmat")

    # Segmentation
    if seg_output_transform_for_metrics is not None:
        cm = ConfusionMatrix(num_classes=2, output_transform=seg_output_transform_for_metrics)
        DiceCoefficient(cm).attach(evaluator, "val_dice")
        JaccardIndex(cm).attach(evaluator, "val_iou")
