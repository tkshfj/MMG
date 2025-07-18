import torch
import os
import wandb
from monai.utils import set_determinism
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler
from model_utils import build_model, get_optimizer
from data_utils import build_dataloaders
from eval_utils import get_config


def multitask_loss_function(y_pred, y_true):
    """
    Custom loss function for multitask learning
    Args:
        y_pred: tuple of (class_logits, seg_out) from model
        y_true: dict with 'label' and 'mask' keys
    """
    class_logits, seg_out = y_pred
    labels = y_true['label']
    masks = y_true['mask']

    # Loss functions
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

    return loss_seg + loss_cls


def prepare_batch_for_trainer(batch):
    """
    Transform batch data for MONAI trainer
    Args:
        batch: dict with 'image', 'mask', 'label' keys
    Returns:
        tuple: (inputs, targets) where targets is dict format expected by loss function
    """
    inputs = batch['image']
    targets = {
        'label': batch['label'],
        'mask': batch['mask']
    }
    return inputs, targets


class ValidationHandler:
    """Simple validation handler for logging metrics"""

    def __init__(self, val_loader, device, task="multitask"):
        self.val_loader = val_loader
        self.device = device
        self.task = task

    def __call__(self, engine):
        """Called at end of each epoch for validation"""
        model = engine.network
        epoch = engine.state.epoch - 1  # Convert to 0-indexed

        # Simple validation loop
        model.eval()
        val_losses = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 5:  # Just validate on first 5 batches for speed
                    break

                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                class_logits, seg_out = model(images)

                # Calculate loss
                targets = {'label': labels, 'mask': masks}
                predictions = (class_logits, seg_out)
                loss = multitask_loss_function(predictions, targets)
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

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

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Log to WandB (simplified)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": avg_val_loss
        })


def train_with_monai_simple(config=None):
    """Simplified training with MONAI SupervisedTrainer"""

    with wandb.init(config=config):
        config = get_config(wandb.config)
        set_determinism(seed=42)
        run_id = wandb.run.id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {device}")

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

        # Prepare data loaders for MONAI trainer
        # MONAI expects (inputs, targets) format
        def prepare_batch(batchdata, device=None, non_blocking=False, **kwargs):
            """
            Transform batch data for MONAI trainer
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
            loss_function=multitask_loss_function,
            prepare_batch=prepare_batch,  # Transform batch format
            iteration_update=None,
            inferer=None,
            postprocessing=None,
            key_train_metric=None,
            additional_metrics=None,
            metric_cmp_fn=None,
            train_handlers=None,
            amp=False
        )

        # Add logging handlers - StatsHandler attaches itself automatically
        def train_output_transform(x):
            if isinstance(x, dict) and 'loss' in x:
                return x['loss']
            elif isinstance(x, list) and len(x) > 0:
                return x[0]
            elif isinstance(x, dict):
                # If dict doesn't have 'loss', try to get first value or return 0
                return next(iter(x.values())) if x else 0.0
            else:
                return x

        stats_handler = StatsHandler(
            tag_name="train",
            output_transform=train_output_transform
        )
        stats_handler.attach(trainer)

        # Add validation handler
        val_handler = ValidationHandler(val_loader, device, task=config.task)
        trainer.add_event_handler("EPOCH_COMPLETED", val_handler)

        print(f"Starting training for {config.epochs} epochs...")

        # Run training
        trainer.run()

        # Save model
        print("Training complete.")
        os.makedirs("outputs/models", exist_ok=True)
        save_path = f"outputs/models/multitask_unet_monai_simple_{run_id}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train_with_monai_simple()
