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
from engine_utils import build_trainer, build_evaluator, make_prepare_batch
from handlers import register_handlers, wandb_log_handler
# image_log_handler, manual_dice_handler
from metrics_utils import attach_metrics


def main(config=None):
    # Initialize Weights & Biases run (config passed from sweep)
    with wandb.init(config=config, dir="outputs"):
        # Load and validate config
        config = load_and_validate_config(wandb.config)
        set_determinism(seed=42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config["device"] = device
        run_id = wandb.run.id
        # alpha = float(wandb.config["alpha"])
        # beta  = float(wandb.config["beta"])

        # Build data loaders
        train_loader, val_loader, test_loader = build_dataloaders(
            metadata_csv=config["metadata_csv"],
            input_shape=config.get("input_shape", (256, 256)),
            batch_size=config.get("batch_size", 16),
            task=config.get("task", "multitask"),
            split=config.get("split", (0.7, 0.15, 0.15)),
            num_workers=config.get("num_workers", 32),
            debug=True
        )

        # Print dataset sizes
        print("[Sample Counts]")
        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")
        print(f"Number of test samples: {len(test_loader.dataset)}")

        # DEBUG: Check batch structure before passing to prepare_batch
        for batch in val_loader:
            print("[DEBUG] Loader batch type:", type(batch))
            print("[DEBUG] Loader batch keys:", batch.keys() if isinstance(batch, dict) else "not dict")
            # Optionally: print sample shapes
            if isinstance(batch, dict):
                for k in batch:
                    print(f"  - {k}: type={type(batch[k])}, shape={getattr(batch[k], 'shape', 'N/A')}")
            break  # Only check the first batch

        # Select model class and construct model
        architecture = config.get("architecture", "multitask_unet").lower()
        if architecture not in MODEL_REGISTRY:
            raise ValueError(f"Model '{architecture}' not found in MODEL_REGISTRY.")
        model_cls = MODEL_REGISTRY[architecture]
        wrapper = model_cls()  # Create model instance
        model = wrapper.build_model(config).to(device)  # Get the nn.Module for training/inference

        # DEBUG
        batch = next(iter(train_loader))
        print(batch["image"].shape)  # torch.Size([B, 1, H, W])
        print(batch["mask"].shape)   # torch.Size([B, 1, H, W]), if segmentation
        print(batch["label"].shape)  # torch.Size([B]), for classification/multitask

        # print(f"[DEBUG batch] type: {type(batch)}, keys: {getattr(batch, 'keys', lambda: None)()}")
        # images = batch["image"]
        # print(f"[DEBUG images] type: {type(images)}")
        # batch = next(iter(val_loader))
        # print(type(batch))
        # print(batch.keys())
        # print(type(batch["image"]))  # Should be torch.Tensor
        # print(type(batch["label"]))  # Should be dict for multitask

        # Get optimizer, loss, metrics, output_transform, handler kwargs
        optimizer = get_optimizer(
            config["optimizer"],
            model.parameters(),
            lr=config.get("learning_rate", config.get("base_learning_rate", 2e-4)),
            weight_decay=config.get("weight_decay", config.get("l2_reg", 1e-4)),
        )
        loss_fn = wrapper.get_loss_fn()
        metrics = wrapper.get_metrics()
        handler_kwargs = wrapper.get_handler_kwargs()

        # Dynamically create a prepare_batch with the correct task argument
        task = config.get("task", None)
        if not task:
            # Try to auto-infer from model if not specified
            task = wrapper.get_supported_tasks()[0]
        assert task in ["classification", "segmentation", "multitask"], f"Unknown task: {task}"

        prepare_batch_fn = make_prepare_batch(task)

        # Build trainer and evaluator
        trainer = build_trainer(
            device=device,
            max_epochs=config["epochs"],
            train_data_loader=train_loader,
            network=model,
            optimizer=optimizer,
            loss_function=loss_fn,
            prepare_batch=prepare_batch_fn
        )

        evaluator = build_evaluator(
            device=device,
            val_data_loader=val_loader,
            network=model,
            prepare_batch=prepare_batch_fn,
            metrics=metrics,
            decollate=False,
            postprocessing=None,
            inferer=None,
        )

        # ADD DEBUG HANDLER
        from ignite.engine import Events

        def debug_model_output(engine):
            output = engine.state.output
            print("[DEBUG] model output type:", type(output))
            print("[DEBUG] model output:", output)
            if isinstance(output, (tuple, list)):
                for i, item in enumerate(output):
                    print(f"[DEBUG] output[{i}] type:", type(item))
                    if isinstance(item, dict):
                        print(f"   keys: {item.keys()}")
            elif isinstance(output, dict):
                for k, v in output.items():
                    print(f"[DEBUG] output['{k}'] type: {type(v)}")

        evaluator.add_event_handler(Events.ITERATION_COMPLETED, debug_model_output)

        # Metric Attachment
        attach_metrics(evaluator, wrapper, config, val_loader)

        # Register handlers (fully model-agnostic)
        register_handlers(
            trainer,
            evaluator,
            model,
            config,
            train_loader=train_loader,
            val_loader=val_loader,
            # image_log_handler=image_log_handler,
            wandb_log_handler=wandb_log_handler,
            prepare_batch=prepare_batch_fn,
            **handler_kwargs
        )

        # DEBUG: Check model output shape
        b = next(iter(train_loader))
        prep_fn = make_prepare_batch("multitask")  # or "classification"/"segmentation"
        pb_out = prep_fn(b, device=device, non_blocking=True)

        if isinstance(pb_out, tuple) and len(pb_out) == 4:
            x, targets, args, kwargs = pb_out
            print("[PREP CHECK] tuple(4) OK")
            print("  x:", tuple(x.shape), x.dtype, x.device)
            if isinstance(targets, dict):
                for k, v in targets.items():
                    try:
                        print(f"  targets[{k}]:", tuple(v.shape), v.dtype)
                    except Exception:
                        print(f"  targets[{k}]:", type(v))
            else:
                try:
                    print("  targets:", tuple(targets.shape), targets.dtype)
                except Exception:
                    print("  targets type:", type(targets))
        elif isinstance(pb_out, dict):
            print("[PREP CHECK] legacy dict path")
            for k, v in pb_out.items():
                try:
                    print(f"  {k}:", tuple(v.shape), v.dtype)
                except Exception:
                    print(f"  {k} type:", type(v))
        else:
            raise TypeError(f"Unexpected prepare_batch output: {type(pb_out)}")

        # Main training loop
        trainer.run()

        # Save final model checkpoint
        os.makedirs("outputs/best_model", exist_ok=True)
        final_path = f"outputs/best_model/{architecture}_{run_id}.pth"
        torch.save(model.state_dict(), final_path)
        print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
