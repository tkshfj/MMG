# multitask_modular/train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch
import wandb
from ignite.metrics import ROC_AUC
from monai.utils import set_determinism
from config_utils import load_and_validate_config
from data_utils import build_dataloaders
from model_utils import build_model, get_optimizer
from engine_utils import build_trainer, build_evaluator
from metrics_utils import seg_output_transform, auc_output_transform, multitask_loss, attach_metrics
from handlers import register_handlers, wandb_log_handler, image_log_handler, manual_dice_handler


# Batch preparation function for engines
def prepare_batch(batch, device=None, non_blocking=False):
    """Move batch data to device and prepare inputs/targets for the model."""
    images = batch['image'].to(device, non_blocking=non_blocking)
    masks = batch['mask'].to(device, non_blocking=non_blocking)
    labels = batch['label'].long().to(device, non_blocking=non_blocking)
    return images, {'label': labels, 'mask': masks}


# Main training function using MONAI engines
def main(config=None):
    # Initialize W&B run (config passed from sweep or CLI)
    with wandb.init(config=config, dir="outputs/wandb"):
        config = load_and_validate_config(wandb.config)
        set_determinism(seed=42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config["device"] = device
        run_id = wandb.run.id

        # Prepare data loaders
        train_loader, val_loader, test_loader = build_dataloaders(
            metadata_csv=config["metadata_csv"],
            input_shape=config.get("input_shape", (256, 256)),
            batch_size=config.get("batch_size", 16),
            task=config.get("task", "multitask"),
            split=config.get("split", (0.7, 0.15, 0.15)),
            num_workers=config.get("num_workers", 32)
        )

        # Build model and optimizer
        model = build_model(config).to(device)
        optimizer = get_optimizer(
            config["optimizer"],
            model.parameters(),
            lr=config.get("learning_rate", config.get("base_learning_rate", 2e-4)),
            weight_decay=config.get("weight_decay", config.get("l2_reg", 1e-4)),
        )

        # MONAI Trainer Setup
        trainer = build_trainer(device, config["epochs"], train_loader, model, optimizer, multitask_loss, prepare_batch)

        # MONAI Evaluator Setup
        evaluator = build_evaluator(
            device=device,
            val_data_loader=val_loader,
            network=model,
            prepare_batch=prepare_batch,
            key_val_metric={"val_auc": ROC_AUC(output_transform=auc_output_transform)}
        )

        # Attach metrics to evaluator
        attach_metrics(evaluator, config, seg_output_transform=seg_output_transform)

        register_handlers(
            trainer,
            evaluator,
            model,
            config,
            train_loader=train_loader,
            val_loader=val_loader,
            manual_dice_handler=manual_dice_handler,
            image_log_handler=image_log_handler,
            wandb_log_handler=wandb_log_handler,
            add_segmentation_metrics=True,
            num_classes=2,
            seg_output_transform=seg_output_transform,
            dice_name="val_dice",
            iou_name="val_iou",
            prepare_batch=prepare_batch
        )

        # Training loop
        trainer.run()

        # Save final model
        os.makedirs("outputs/models", exist_ok=True)
        final_path = f"outputs/models/multitask_unet_{run_id}.pth"
        torch.save(model.state_dict(), final_path)
        print("Training complete. Final model saved to:", final_path)


if __name__ == "__main__":
    main()
