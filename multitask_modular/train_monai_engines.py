import torch
import os
import wandb
from monai.utils import set_determinism
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler
from model_utils import build_model, get_optimizer
from data_utils import build_dataloaders
from eval_utils import (
    get_segmentation_metrics, get_config,
    compute_segmentation_metrics, compute_classification_metrics, log_wandb,
    log_sample_images
)


def create_multitask_loss_function():
    """Create a custom loss function for multitask learning"""
    def multitask_loss(y_pred, y_true):
        """
        y_pred: (class_logits, seg_out) or similar structure
        y_true: {'label': classification_targets, 'mask': segmentation_targets}
        """
        # Handle different input formats robustly
        if isinstance(y_pred, (list, tuple)) and len(y_pred) == 2:
            class_logits, seg_out = y_pred
        elif isinstance(y_pred, dict):
            class_logits = y_pred.get('class_logits', y_pred.get('classification', None))
            seg_out = y_pred.get('seg_out', y_pred.get('segmentation', None))
        elif isinstance(y_pred, (list, tuple)) and len(y_pred) == 1:
            # Sometimes wrapped in an extra list
            inner = y_pred[0]
            if isinstance(inner, (list, tuple)) and len(inner) == 2:
                class_logits, seg_out = inner
            else:
                raise ValueError(f"Unexpected nested structure in loss: {type(inner)}")
        else:
            raise ValueError(f"Unexpected y_pred format in loss: {type(y_pred)}")

        if class_logits is None or seg_out is None:
            raise ValueError(f"Could not extract class_logits and seg_out from y_pred in loss: {type(y_pred)}")

        labels = y_true['label']
        masks = y_true['mask']

        # Get loss functions
        criterion_seg = torch.nn.BCEWithLogitsLoss()
        criterion_cls = torch.nn.CrossEntropyLoss()

        # Ensure proper shapes and types
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()
        labels = labels.long()

        # Calculate losses
        loss_seg = criterion_seg(seg_out, masks)
        loss_cls = criterion_cls(class_logits, labels)

        # Combined loss
        total_loss = loss_seg + loss_cls

        return total_loss

    return multitask_loss


def create_multitask_postprocessing():
    """Create postprocessing function for predictions"""
    def postprocess(y_pred):
        """
        Apply postprocessing to model outputs
        Handle different input formats robustly
        """
        # Handle different input formats
        if isinstance(y_pred, (list, tuple)) and len(y_pred) == 2:
            class_logits, seg_out = y_pred
        elif isinstance(y_pred, dict):
            # Check what keys are available for debugging
            available_keys = list(y_pred.keys())

            # Try multiple possible key names for classification
            class_logits = None
            for key in ['class_logits', 'classification', 'class_out', 'cls_logits', '0']:
                if key in y_pred and isinstance(y_pred[key], torch.Tensor):
                    class_logits = y_pred[key]
                    break

            # Try numeric index if string keys failed
            if class_logits is None and 0 in y_pred and isinstance(y_pred[0], torch.Tensor):
                class_logits = y_pred[0]

            # Try multiple possible key names for segmentation
            seg_out = None
            for key in ['seg_out', 'segmentation', 'seg_logits', 'mask', '1']:
                if key in y_pred and isinstance(y_pred[key], torch.Tensor):
                    seg_out = y_pred[key]
                    break

            # Try numeric index if string keys failed
            if seg_out is None and 1 in y_pred and isinstance(y_pred[1], torch.Tensor):
                seg_out = y_pred[1]

            # If still not found, try to extract from a nested structure
            if class_logits is None or seg_out is None:
                for key, value in y_pred.items():
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        class_logits, seg_out = value
                        break
                    elif isinstance(value, torch.Tensor):
                        # If we find tensors, try to assign them based on likely dimensions
                        if value.ndim == 2 and class_logits is None:  # Likely classification logits
                            class_logits = value
                        elif value.ndim == 4 and seg_out is None:  # Likely segmentation output
                            seg_out = value

            # If still None, provide detailed error
            if class_logits is None or seg_out is None:
                tensor_info = {}
                for k, v in y_pred.items():
                    if isinstance(v, torch.Tensor):
                        tensor_info[k] = f"shape={v.shape}, ndim={v.ndim}"
                    else:
                        tensor_info[k] = f"type={type(v)}"
                error_msg = "Could not extract class_logits and seg_out from y_pred dict. "
                error_msg += f"Available keys: {available_keys}. Tensor info: {tensor_info}"
                raise ValueError(error_msg)

        elif isinstance(y_pred, (list, tuple)) and len(y_pred) == 1:
            # Sometimes wrapped in an extra list
            inner = y_pred[0]
            if isinstance(inner, (list, tuple)) and len(inner) == 2:
                class_logits, seg_out = inner
            else:
                raise ValueError(f"Unexpected nested structure: {type(inner)}, len={len(inner) if hasattr(inner, '__len__') else 'no len'}")
        else:
            raise ValueError(f"Unexpected y_pred format: {type(y_pred)}, len={len(y_pred) if hasattr(y_pred, '__len__') else 'no len'}")

        # Classification: softmax (handle different shapes)
        if class_logits.ndim == 1:
            # If 1D, add batch dimension
            class_logits = class_logits.unsqueeze(0)

        if class_logits.shape[1] == 1:
            # Binary classification with single output
            class_probs = torch.sigmoid(class_logits)
            # Convert to 2-class probabilities
            class_probs = torch.cat([1 - class_probs, class_probs], dim=1)
            class_preds = (class_probs[:, 1] > 0.5).long()
        else:
            # Multi-class classification
            class_probs = torch.softmax(class_logits, dim=1)
            class_preds = torch.argmax(class_probs, dim=1)

        # Segmentation: sigmoid + threshold (handle different shapes)
        if seg_out.ndim == 3:
            # Add channel dimension if missing
            seg_out = seg_out.unsqueeze(1)

        seg_probs = torch.sigmoid(seg_out)
        seg_preds = (seg_probs > 0.5).float()

        return {
            'class_logits': class_logits,
            'class_probs': class_probs,
            'class_preds': class_preds,
            'seg_out': seg_out,
            'seg_probs': seg_probs,
            'seg_preds': seg_preds
        }

    return postprocess


def create_multitask_key_metric():
    """Create a key metric function for model selection"""
    def key_metric_fn(y_pred, y_true):
        """
        Calculate a combined metric for model selection
        """
        # Extract predictions and targets
        pred_dict = y_pred
        class_preds = pred_dict['class_preds']
        seg_preds = pred_dict['seg_preds']

        labels = y_true['label']
        masks = y_true['mask']

        # Ensure proper shapes
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if seg_preds.ndim == 3:
            seg_preds = seg_preds.unsqueeze(1)

        # Classification accuracy
        class_acc = (class_preds == labels).float().mean()

        # Segmentation Dice
        intersection = (seg_preds * masks).sum()
        union = seg_preds.sum() + masks.sum()
        dice = (2.0 * intersection) / (union + 1e-8)

        # Combined metric (you can adjust weights)
        combined_metric = 0.5 * class_acc + 0.5 * dice

        return combined_metric

    return key_metric_fn


class CustomValidationHandler:
    """Custom handler for validation logging"""

    def __init__(self, val_data_loader, device, task="multitask"):
        self.val_data_loader = val_data_loader
        self.device = device
        self.task = task
        self.dice_metric = get_segmentation_metrics()["dice"]
        self.iou_metric = get_segmentation_metrics()["iou"]

    def __call__(self, engine):
        """Called at the end of each epoch"""
        model = engine.network
        model.eval()

        val_loss = 0.0
        manual_dices = []
        cls_preds, cls_probs, cls_targets = [], [], []
        self.dice_metric.reset()
        self.iou_metric.reset()
        images_for_logging, masks_for_logging, preds_for_logging = [], [], []

        with torch.no_grad():
            for batch in self.val_data_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label'].long().to(self.device)

                # Ensure proper shapes
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                # Forward pass
                class_logits, seg_out = model(images)
                pred_mask = (torch.sigmoid(seg_out) > 0.5).float()
                if pred_mask.ndim == 3:
                    pred_mask = pred_mask.unsqueeze(1)

                # Ensure types match
                masks = masks.float().to(self.device)
                pred_mask = pred_mask.float().to(self.device)

                # Calculate losses
                criterion_seg = torch.nn.BCEWithLogitsLoss()
                criterion_cls = torch.nn.CrossEntropyLoss()
                loss_seg = criterion_seg(seg_out, masks)
                loss_cls = criterion_cls(class_logits, labels)
                loss = loss_seg + loss_cls
                val_loss += loss.item()

                # Classification metrics
                probs = torch.softmax(class_logits, dim=1)
                pred_labels = torch.argmax(probs, dim=1)
                cls_preds.extend(pred_labels.cpu().numpy())
                cls_probs.extend(probs[:, 1].cpu().numpy())
                cls_targets.extend(labels.cpu().numpy())

                # MONAI metrics
                self.dice_metric(pred_mask, masks)
                self.iou_metric(pred_mask, masks)

                # Manual Dice
                intersection = (pred_mask * masks).sum(dim=(1, 2, 3))
                union = pred_mask.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                manual_dice = (2. * intersection / (union + 1e-8)).mean().item()
                manual_dices.append(manual_dice)

                # Store images for logging
                if len(images_for_logging) < 2:
                    images_for_logging.append(images.cpu())
                    masks_for_logging.append(masks.cpu())
                    preds_for_logging.append(pred_mask.cpu())

        # Compute metrics
        avg_val_loss = val_loss / len(self.val_data_loader)
        avg_dice, avg_iou, avg_manual_dice = compute_segmentation_metrics(
            self.dice_metric, self.iou_metric, manual_dices, task=self.task
        )
        val_acc, val_rocauc = compute_classification_metrics(cls_targets, cls_preds, cls_probs)

        # Get current epoch
        epoch = engine.state.epoch - 1  # MONAI engines are 1-indexed

        # Log to console
        print(
            f"Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_dice:.4f} | Manual Dice: {avg_manual_dice:.4f} | Val IoU: {avg_iou:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val AUC: {val_rocauc:.4f}"
        )

        # Log to WandB
        # Extract train loss robustly from engine output
        output = engine.state.output
        if isinstance(output, dict) and 'loss' in output:
            train_loss = output['loss']
        elif isinstance(output, list) and len(output) > 0:
            train_loss = output[0]
        elif isinstance(output, dict):
            train_loss = next(iter(output.values())) if output else 0.0
        else:
            train_loss = output if isinstance(output, (int, float)) else 0.0

        log_wandb(epoch, train_loss, avg_val_loss, avg_dice, avg_manual_dice, avg_iou, val_acc, val_rocauc, task=self.task)

        # Log sample images
        if images_for_logging:
            log_sample_images(images_for_logging[0], masks_for_logging[0], preds_for_logging[0], epoch, task=self.task)

        # Store metrics in engine state for handlers to access
        engine.state.metrics = {
            'val_loss': avg_val_loss,
            'val_dice': avg_dice,
            'val_iou': avg_iou,
            'val_acc': val_acc,
            'val_auc': val_rocauc
        }


def train_with_monai_engines(config=None):
    """Training function using MONAI SupervisedTrainer and SupervisedEvaluator"""

    with wandb.init(config=config):
        config = get_config(wandb.config)
        set_determinism(seed=42)
        run_id = wandb.run.id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build data loaders
        train_loader, val_loader, test_loader = build_dataloaders(
            metadata_csv="../data/processed/cbis_ddsm_metadata_full.csv",
            input_shape=(256, 256),
            batch_size=config.batch_size,
            task="multitask",
            split=(0.7, 0.15, 0.15),
            num_workers=12
        )

        # Build model and optimizer
        model = build_model(config).to(device)
        optimizer = get_optimizer(config.optimizer, model.parameters(), config.learning_rate, config.weight_decay)

        # Create loss function
        loss_function = create_multitask_loss_function()

        # Create postprocessing
        post_transform = create_multitask_postprocessing()

        # Create key metric (commented out since key_train_metric is used differently)
        # key_metric = create_multitask_key_metric()

        # Prepare batch function for multitask learning
        def prepare_batch(batchdata, device=None, non_blocking=False, **kwargs):
            """
            Transform batch data for multitask learning
            Args:
                batchdata: The batch from your dataloader
                device: Target device for tensors
                non_blocking: Whether to use non-blocking transfer
                **kwargs: Additional arguments
            Returns:
                tuple: (inputs, targets) where targets is dict format expected by loss function
            """
            # Move tensors to device if specified
            if device is not None:
                inputs = batchdata['image'].to(device, non_blocking=non_blocking)
                mask = batchdata['mask'].to(device, non_blocking=non_blocking)
                label = batchdata['label'].to(device, non_blocking=non_blocking)
            else:
                inputs = batchdata['image']
                mask = batchdata['mask']
                label = batchdata['label']

            targets = {
                'label': label,
                'mask': mask
            }
            return inputs, targets

        # Create trainer
        trainer = SupervisedTrainer(
            device=device,
            max_epochs=config.epochs,
            train_data_loader=train_loader,
            network=model,
            optimizer=optimizer,
            loss_function=loss_function,
            prepare_batch=prepare_batch,  # Custom batch preparation for multitask learning
            inferer=None,  # Use default inferer
            postprocessing=post_transform,
            key_train_metric=None,  # Can add metrics here if needed
            additional_metrics=None,
            metric_cmp_fn=lambda x, y: x > y,  # Higher is better for our combined metric
            train_handlers=None,
            amp=False  # Set to True if you want automatic mixed precision
        )

        # Add handlers to trainer - StatsHandler attaches itself automatically
        # Create output transform for logging (only scalars)
        def train_output_transform(x):
            """Extract scalar loss value for training logging"""
            if isinstance(x, dict) and 'loss' in x:
                loss_val = x['loss']
                if hasattr(loss_val, 'item'):
                    return loss_val.item()
                return loss_val
            elif isinstance(x, list) and len(x) > 0:
                loss_val = x[0]
                if hasattr(loss_val, 'item'):
                    return loss_val.item()
                return loss_val
            elif isinstance(x, dict):
                # If dict doesn't have 'loss', try to get first scalar value
                for key, value in x.items():
                    if isinstance(value, (int, float)):
                        return value
                    elif hasattr(value, 'item') and value.numel() == 1:
                        return value.item()
                return 0.0
            else:
                if hasattr(x, 'item'):
                    return x.item()
                return x if isinstance(x, (int, float)) else 0.0

        stats_handler = StatsHandler(
            tag_name="train",
            output_transform=train_output_transform
        )
        stats_handler.attach(trainer)

        # Add custom validation handler
        validation_handler = CustomValidationHandler(val_loader, device, task=config.task)
        trainer.add_event_handler("EPOCH_COMPLETED", validation_handler)

        # Add epoch-level stats handler with scalar-only output
        def epoch_output_transform(x):
            """Extract scalar loss for epoch logging"""
            if isinstance(x, dict) and 'loss' in x:
                loss_val = x['loss']
                if hasattr(loss_val, 'item'):
                    return {'loss': loss_val.item()}
                return {'loss': loss_val}
            elif isinstance(x, list) and len(x) > 0:
                loss_val = x[0]
                if hasattr(loss_val, 'item'):
                    return {'loss': loss_val.item()}
                return {'loss': loss_val}
            elif isinstance(x, dict):
                # If dict doesn't have 'loss', try to get first scalar value
                for key, value in x.items():
                    if isinstance(value, (int, float)):
                        return {'loss': value}
                    elif hasattr(value, 'item') and value.numel() == 1:
                        return {'loss': value.item()}
                return {'loss': 0.0}
            else:
                if hasattr(x, 'item'):
                    return {'loss': x.item()}
                return {'loss': x if isinstance(x, (int, float)) else 0.0}

        epoch_stats_handler = StatsHandler(
            tag_name="epoch",
            output_transform=epoch_output_transform,
            iteration_log=False,
            epoch_log=True
        )
        epoch_stats_handler.attach(trainer)

        # Optional: Add early stopping based on validation metric
        # from monai.handlers import EarlyStopHandler
        # trainer.add_event_handler(
        #     "EPOCH_COMPLETED",
        #     EarlyStopHandler(
        #         patience=10,
        #         min_delta=0.001,
        #         score_function=lambda engine: engine.state.metrics.get('val_auc', 0.0)
        #     )
        # )

        # Run training
        print(f"Starting training for {config.epochs} epochs...")
        trainer.run()

        # Save final model
        print("Training complete.")
        os.makedirs("outputs/models", exist_ok=True)
        save_path = f"outputs/models/multitask_unet_monai_{run_id}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train_with_monai_engines()
