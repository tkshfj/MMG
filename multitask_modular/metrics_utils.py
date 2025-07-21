# metrics_utils.py
import torch
from ignite.metrics import Accuracy, Loss, ROC_AUC, ConfusionMatrix, DiceCoefficient, JaccardIndex
from eval_utils import get_classification_metrics, get_segmentation_metrics


# Output transform for classification metrics
def cls_output_transform(output):
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        pred_logits = torch.stack([s['pred'][0] for s in output])
        device = pred_logits.device
        true_labels = torch.tensor(
            [s['label']['label'] for s in output], dtype=torch.long, device=device
        )
        return pred_logits, true_labels
    raise ValueError("cls_output_transform expects a list of dicts with 'pred' and 'label' keys")


# Output transform for AUROC (returns probabilities and labels)
def auc_output_transform(output):
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        probs_list, true_labels = [], []
        device = torch.as_tensor(output[0]['pred'][0]).device

        for sample in output:
            class_logits = torch.as_tensor(sample['pred'][0]).to(device)
            if class_logits.ndim == 1:
                prob = torch.softmax(class_logits, dim=0)[1]
            elif class_logits.ndim == 2:
                prob = torch.softmax(class_logits, dim=1)[0, 1]
            else:
                raise ValueError(f"Unexpected logits shape: {class_logits.shape}")
            probs_list.append(prob.unsqueeze(0))
            true_labels.append(torch.tensor([sample['label']['label']], dtype=torch.long, device=device))

        return torch.cat(probs_list, dim=0), torch.cat(true_labels, dim=0)
    raise ValueError("auc_output_transform expects list of dicts")


# Output transform for segmentation metrics
def seg_output_transform(output):
    if isinstance(output, list) and all(isinstance(x, dict) for x in output):
        pred_logits = []
        true_masks = []
        for sample in output:
            seg_logits = sample['pred'][1]  # [2, H, W]
            # Assume logits already [2, H, W]. If [1, H, W], convert to [2, H, W] as before.
            if seg_logits.shape[0] == 1:
                prob = torch.sigmoid(seg_logits)
                prob = torch.cat([1 - prob, prob], dim=0)  # [2, H, W]
                # prob = torch.cat([1 - torch.sigmoid(seg_logits), torch.sigmoid(seg_logits)], dim=0)
            else:
                prob = torch.softmax(seg_logits, dim=0)
                # prob = torch.softmax(seg_logits, dim=0)
            pred_logits.append(prob.unsqueeze(0))  # [1, 2, H, W]
            mask = sample['label']['mask'].long()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            true_masks.append(mask.unsqueeze(0))  # [1, H, W]
        pred_logits = torch.cat(pred_logits, dim=0)  # [B, 2, H, W]
        pred_labels = torch.argmax(pred_logits, dim=1)  # [B, H, W]
        true_masks = torch.cat(true_masks, dim=0)    # [B, H, W]
        return pred_logits, true_masks
        # return pred_labels, true_masks
    else:
        raise ValueError("seg_output_transform expected list of dicts")


# Combined loss for multitask learning
def multitask_loss(y_pred, y_true):
    class_logits, seg_out = y_pred
    labels = y_true['label']
    masks = y_true['mask']
    loss_cls = get_classification_metrics()["loss"](class_logits, labels)
    loss_seg = get_segmentation_metrics()["loss"](seg_out, masks)
    return loss_cls + loss_seg


# Attach classification + segmentation metrics to evaluator
def attach_metrics(evaluator, config, seg_output_transform=None):
    # Classification metrics
    Accuracy(output_transform=cls_output_transform).attach(evaluator, "val_acc")
    ROC_AUC(output_transform=auc_output_transform).attach(evaluator, "val_auc")
    Loss(
        loss_fn=get_classification_metrics()["loss"],
        output_transform=cls_output_transform
    ).attach(evaluator, "val_loss")

    # Segmentation metrics
    if seg_output_transform is not None:
        cm = ConfusionMatrix(num_classes=2, output_transform=seg_output_transform)
        DiceCoefficient(cm).attach(evaluator, "val_dice")
        JaccardIndex(cm).attach(evaluator, "val_iou")


# def seg_output_transform(output):
#     # For segmentation (Dice/IoU): returns (pred_mask, true_mask), both [batch, H, W], integer labels
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         pred_logits = []
#         true_masks = []
#         for sample in output:
#             # seg_logits: [1, H, W] or [2, H, W]
#             seg_logits = sample['pred'][1]
#             # If output is [1, H, W], convert to [2, H, W] by stacking
#             if seg_logits.shape[0] == 1:
#                 # Binary segmentation, get class probabilities for [bg, fg]
#                 prob = torch.sigmoid(seg_logits)
#                 prob = torch.cat([1 - prob, prob], dim=0)  # [2, H, W]
#             else:
#                 prob = torch.softmax(seg_logits, dim=0)  # already [2, H, W]
#             pred_logits.append(prob.unsqueeze(0))  # [1, 2, H, W]
#             mask = sample['label']['mask'].long()  # [H, W] or [1, H, W]
#             if mask.ndim == 3 and mask.shape[0] == 1:
#                 mask = mask.squeeze(0)
#             true_masks.append(mask.unsqueeze(0))  # [1, H, W]
#         pred_logits = torch.cat(pred_logits, dim=0)  # [batch, 2, H, W]
#         pred_labels = torch.argmax(pred_logits, dim=1)  # [batch, H, W]
#         true_masks = torch.cat(true_masks, dim=0)      # [batch, H, W]
#         return pred_labels, true_masks
#     else:
#         raise ValueError("seg_output_transform expected list of dicts")


# # def seg_output_transform(output):
# #     """
# #     For segmentation (Dice/IoU/Jaccard): returns (pred_onehot, true_onehot), both [B, 2, H, W]
# #     - Accepts output as a list of dicts, each with ['pred'] and ['label']['mask'].
# #     - Applies sigmoid + threshold to get binary mask, then one-hot encodes for Ignite metrics.
# #     """
# #     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
# #         pred_masks, true_masks = [], []
# #         # Get device from first sample's seg_logits
# #         device = output[0]['pred'][1].device
# #         for sample in output:
# #             seg_logits = sample['pred'][1]
# #             true_mask = sample['label']['mask']
# #             # (1, H, W)
# #             pred_mask = (torch.sigmoid(seg_logits).unsqueeze(0) > 0.5).float().to(device)
# #             true_mask = true_mask.unsqueeze(0).float().to(device)
# #             pred_masks.append(pred_mask)
# #             true_masks.append(true_mask)
# #         pred_masks = torch.cat(pred_masks, dim=0)   # [B, 1, H, W]
# #         true_masks = torch.cat(true_masks, dim=0)   # [B, 1, H, W]

# #         # Convert to one-hot for Ignite's ConfusionMatrix (assumes binary: 0=bg, 1=fg)
# #         pred_onehot = torch.cat([(1 - pred_masks), pred_masks], dim=1)     # [B, 2, H, W]
# #         true_onehot = torch.cat([(1 - true_masks), true_masks], dim=1)     # [B, 2, H, W]
# #         return pred_onehot, true_onehot
# #     else:
# #         raise ValueError(f"seg_output_transform expected list of dicts, got: {output}")


# # def seg_output_transform(output):
# #     # For segmentation (Dice/IoU/Jaccard): returns (pred_mask, true_mask), both batched
# #     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
# #         pred_masks, true_masks = [], []
# #         # Get device from first sample's seg_logits
# #         device = output[0]['pred'][1].device
# #         for sample in output:
# #             seg_logits = sample['pred'][1]
# #             true_mask = sample['label']['mask']
# #             # Ensure (1, H, W)
# #             pred_mask = (torch.sigmoid(seg_logits).unsqueeze(0) > 0.5).float().to(device)
# #             true_mask = true_mask.unsqueeze(0).float().to(device)
# #             pred_masks.append(pred_mask)
# #             true_masks.append(true_mask)
# #         pred_masks = torch.cat(pred_masks, dim=0)
# #         true_masks = torch.cat(true_masks, dim=0)
# #         return pred_masks, true_masks
# #     else:
# #         raise ValueError(f"seg_output_transform expected list of dicts, got: {output}")


# def cls_output_transform(output):
#     # For classification (Accuracy, AUROC): returns (pred_label, true_label), both batched
#     # output is list of dicts per sample: [{'pred': (class_logits, seg_out), 'label': {...}}, ...]
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         # Should return [batch, num_classes], NOT [batch]
#         pred_logits = torch.stack([s['pred'][0] for s in output])  # s['pred'][0] should be [num_classes]
#         device = pred_logits.device
#         true_labels = torch.tensor([s['label']['label'] for s in output], dtype=torch.long, device=device)
#         assert pred_logits.shape[0] == true_labels.shape[0], "Batch size mismatch in classification output"
#         print("DEBUG: pred_logits.shape:", pred_logits.shape)
#         print("CLS METRIC TRANS:", pred_logits.shape, true_labels.shape)
#         return pred_logits, true_labels
#     else:
#         raise ValueError("cls_output_transform expects a list of dicts")


# def auc_output_transform(output):
#     """
#     For AUROC: returns (probs, true_label), batched.
#     Handles both single-sample ([num_classes]) and batched ([batch, num_classes]) logits.
#     """
#     if isinstance(output, list) and all(isinstance(x, dict) for x in output):
#         probs_list, true_labels = [], []
#         # Get device from first sample's logits
#         device = torch.as_tensor(output[0]['pred'][0]).device
#         for sample in output:
#             class_logits = torch.as_tensor(sample['pred'][0]).to(device)
#             label = sample['label']['label']
#             # If class_logits is shape [num_classes] (standard case), softmax on dim=0
#             if class_logits.ndim == 1:
#                 prob = torch.softmax(class_logits, dim=0)[1]
#             # If shape is [1, num_classes], squeeze then softmax on dim=0
#             elif class_logits.ndim == 2 and class_logits.shape[0] == 1:
#                 prob = torch.softmax(class_logits.squeeze(0), dim=0)[1]
#             # If shape is [batch, num_classes], use dim=1, pick first sample
#             elif class_logits.ndim == 2:
#                 prob = torch.softmax(class_logits, dim=1)[0, 1]
#             else:
#                 raise ValueError(f"class_logits shape unexpected: {class_logits.shape}")
#             probs_list.append(prob.unsqueeze(0))
#             # true_labels.append(torch.tensor([label], dtype=torch.long))
#             true_labels.append(torch.tensor([label], dtype=torch.long, device=device))
#         return torch.cat(probs_list, dim=0), torch.cat(true_labels, dim=0)
#     else:
#         raise ValueError(f"auc_output_transform expected list of dicts, got: {output}")


# # Multitask loss function (segmentation + classification)
# def multitask_loss(y_pred, y_true):
#     """Compute combined segmentation + classification loss."""
#     class_logits, seg_out = y_pred            # model outputs
#     labels = y_true['label']                  # ground truth class labels
#     masks = y_true['mask']                    # ground truth segmentation masks
#     # Individual losses
#     loss_seg = get_segmentation_metrics()["loss"](seg_out, masks)
#     loss_cls = get_classification_metrics()["loss"](class_logits, labels)
#     return loss_seg + loss_cls


# # Attach metrics to evaluator so StatsHandler and wandb can access them
# def attach_metrics(evaluator, config, seg_output_transform=None):
#     # Attach classification metrics
#     Accuracy(output_transform=cls_output_transform).attach(evaluator, "val_acc")
#     # Accuracy(
#     #     output_transform=lambda output: (
#     #         torch.stack([s['pred'][0] for s in output]),
#     #         torch.tensor([int(s['label']['label']) for s in output], dtype=torch.long, device=torch.stack([s['pred'][0] for s in output]).device)
#     #     )
#     # ).attach(evaluator, "val_acc")
#     # ROC AUC (binary/multiclass)
#     ROC_AUC(output_transform=auc_output_transform).attach(evaluator, "val_auc")
#     # Classification Loss
#     Loss(
#         loss_fn=get_classification_metrics()["loss"],
#         output_transform=lambda output: (
#             torch.stack([s["pred"][0] for s in output]),
#             torch.tensor([int(s["label"]["label"]) for s in output], dtype=torch.long, device=torch.stack([s['pred'][0] for s in output]).device)
#         )
#     ).attach(evaluator, "val_loss")

#     # Accuracy(output_transform=cls_output_transform).attach(evaluator, "val_acc")
#     # ROC_AUC(output_transform=auc_output_transform).attach(evaluator, "val_auc")
#     # Loss(
#     #     loss_fn=get_classification_metrics()["loss"],
#     #     output_transform=cls_output_transform
#     # ).attach(evaluator, "val_loss")

#     # Attach segmentation metrics if seg_output_transform is provided
#     # if seg_output_transform is not None:
#     # DiceCoefficient(output_transform=seg_output_transform).attach(evaluator, "val_dice")
#     # DiceCoefficient().attach(evaluator, "val_dice", output_transform=seg_output_transform)

#     # JaccardIndex(output_transform=seg_output_transform).attach(evaluator, "val_iou")
#     # JaccardIndex().attach(evaluator, "val_iou", output_transform=seg_output_transform)

#     # Segmentation metrics
#     if seg_output_transform is not None:
#         cm = ConfusionMatrix(num_classes=2, output_transform=seg_output_transform)
#         DiceCoefficient(cm).attach(evaluator, "val_dice")
#         JaccardIndex(cm).attach(evaluator, "val_iou")
