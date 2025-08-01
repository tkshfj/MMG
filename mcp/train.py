# train.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch
import wandb
from monai.utils import set_determinism
from config_utils import load_and_validate_config
from data_utils import build_dataloaders
from model_registry import MODEL_REGISTRY
from model_utils import get_optimizer
from engine_utils import build_trainer, build_evaluator
from eval_utils import prepare_batch
from handlers import register_handlers, wandb_log_handler, image_log_handler, manual_dice_handler


def main(config=None):
    # Initialize Weights & Biases run (config passed from sweep or CLI)
    with wandb.init(config=config, dir="outputs"):
        # Load and validate config
        config = load_and_validate_config(wandb.config)
        set_determinism(seed=42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config["device"] = device
        run_id = wandb.run.id

        # Build data loaders
        train_loader, val_loader, test_loader = build_dataloaders(
            metadata_csv=config["metadata_csv"],
            input_shape=config.get("input_shape", (256, 256)),
            batch_size=config.get("batch_size", 16),
            task=config.get("task", "multitask"),
            split=config.get("split", (0.7, 0.15, 0.15)),
            num_workers=config.get("num_workers", 32),
            debug=False
        )

        # Select model class and construct model
        architecture = config.get("architecture", "multitask_unet").lower()
        if architecture not in MODEL_REGISTRY:
            raise ValueError(f"Model '{architecture}' not found in MODEL_REGISTRY.")
        model_class = MODEL_REGISTRY[architecture]
        model = model_class.build_model(config).to(device)

        # Get optimizer, loss, metrics, output_transform, handler kwargs
        optimizer = get_optimizer(
            config["optimizer"],
            model.parameters(),
            lr=config.get("learning_rate", config.get("base_learning_rate", 2e-4)),
            weight_decay=config.get("weight_decay", config.get("l2_reg", 1e-4)),
        )
        loss_fn = model_class.get_loss_fn()
        metrics = model_class.get_metrics()
        output_transform = model_class.get_output_transform()
        handler_kwargs = model_class.get_handler_kwargs()

        # Build trainer and evaluator
        trainer = build_trainer(
            device=device,
            max_epochs=config["epochs"],
            train_data_loader=train_loader,
            network=model,
            optimizer=optimizer,
            loss_function=loss_fn,
            prepare_batch=prepare_batch
        )

        evaluator = build_evaluator(
            device=device,
            val_data_loader=val_loader,
            network=model,
            prepare_batch=prepare_batch,
            metrics=metrics,
            output_transform=output_transform
        )

        # Register handlers (fully model-agnostic)
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
            prepare_batch=prepare_batch,
            **handler_kwargs
        )

        # Main training loop (MONAI SupervisedTrainer controls epochs)
        trainer.run()

        # Save final model checkpoint
        os.makedirs("outputs/best_model", exist_ok=True)
        final_path = f"outputs/best_model/{architecture}_{run_id}.pth"
        torch.save(model.state_dict(), final_path)
        print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
