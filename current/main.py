# main.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import wandb
from monai.utils import set_determinism

from config_utils import load_and_validate_config
from data_utils import build_dataloaders
from model_registry import MODEL_REGISTRY
from optim_utils import get_optimizer
from engine_utils import build_trainer, build_evaluator, make_prepare_batch
from handlers import register_handlers, wandb_log_handler


def run(cfg: dict) -> None:
    # setup
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    # data
    train_loader, val_loader, test_loader = build_dataloaders(
        metadata_csv=cfg["metadata_csv"],
        input_shape=cfg.get("input_shape", (256, 256)),
        batch_size=cfg.get("batch_size", 16),
        task=cfg.get("task", "multitask"),
        split=cfg.get("split", (0.7, 0.15, 0.15)),
        num_workers=cfg.get("num_workers", 32),
        debug=bool(cfg.get("debug", False)),
    )

    if cfg.get("print_dataset_sizes", True):
        print(f"[INFO] Train: {len(train_loader.dataset)} | "
              f"Val: {len(val_loader.dataset)} | "
              f"Test: {len(test_loader.dataset)}")

    # model
    arch = cfg.get("architecture", "multitask_unet").lower()
    try:
        wrapper = MODEL_REGISTRY[arch]()   # model wrapper
    except KeyError as e:
        raise ValueError(f"Unknown architecture '{arch}'.") from e

    model = wrapper.build_model(cfg).to(device)

    # optim/metrics/prepare_batch
    optimizer = get_optimizer(
        cfg.get("optimizer", "adamw"),
        model.parameters(),
        lr=cfg.get("learning_rate", cfg.get("base_learning_rate", 2e-4)),
        weight_decay=cfg.get("weight_decay", cfg.get("l2_reg", 1e-4)),
        verbose=bool(cfg.get("debug", False)),
    )

    # task/capabilities
    task = (cfg.get("task") or wrapper.get_supported_tasks()[0]).lower()
    has_cls = task != "segmentation"
    has_seg = task != "classification"

    metrics = wrapper.get_metrics(config=cfg, task=task, has_cls=has_cls, has_seg=has_seg)
    print("[DEBUG] evaluator metrics (to attach):", list(metrics.keys()))

    loss_fn = (wrapper.get_loss_fn(task, cfg) if hasattr(wrapper, "get_loss_fn") else wrapper.get_loss_fn())

    prepare_batch = make_prepare_batch(
        task=task,  # "classification" | "segmentation" | "multitask"
        debug=cfg.get("debug"),
        binary_label_check=True,
        num_classes=int(cfg.get("num_classes", 2)),
        # seg_target="indices"  # default; returns mask as [B,H,W] Long
    )

    # engines
    trainer = build_trainer(
        device=device,
        max_epochs=cfg["epochs"],
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_fn,
        prepare_batch=prepare_batch,
    )

    evaluator = build_evaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        prepare_batch=prepare_batch,
        metrics=metrics,
        decollate=False,
        postprocessing=None,
        inferer=None,
    )

    # Handlers + single-batch debug
    if cfg.get("debug_batch_once", bool(cfg.get("debug", False))):
        from debug_utils import debug_batch
        debug_batch(next(iter(train_loader)), model)

    handler_kwargs = dict(wrapper.get_handler_kwargs())
    # Ensure handler-side gating aligns with task
    handler_kwargs.update({
        "add_classification_metrics": has_cls,
        "add_segmentation_metrics": has_seg,
    })

    register_handlers(
        trainer, evaluator, model, cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_log_handler=wandb_log_handler,
        metric_names=list(metrics.keys()),
        **wrapper.get_handler_kwargs(),
    )

    # train & save
    trainer.run()
    os.makedirs("outputs/best_model", exist_ok=True)
    save_path = f"outputs/best_model/{arch}_{wandb.run.id}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Training complete. Saved: {save_path}")


def main(config: dict | None = None) -> None:
    with wandb.init(config=config or {}, dir="outputs") as wb_run:
        cfg = load_and_validate_config(dict(wandb.config))
        # make run info available
        cfg.setdefault("run_id", wb_run.id)
        cfg.setdefault("run_name", wb_run.name)
        # define epoch-based step semantics
        try:
            wandb.define_metric("epoch")
            wandb.define_metric("val_*", step_metric="epoch")
            wandb.define_metric("train/*", step_metric="epoch")
        except Exception:
            pass
        # call training pipeline
        run(cfg)


if __name__ == "__main__":
    main()
