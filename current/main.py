# main_refactored.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from typing import Tuple
from collections.abc import Callable

import torch
import wandb
from monai.utils import set_determinism

from config_utils import load_and_validate_config
from data_utils import build_dataloaders
from model_registry import MODEL_REGISTRY
from engine_utils import build_trainer, build_evaluator, make_prepare_batch, attach_scheduler
from handlers import register_handlers, wandb_log_handler, CHECKPOINT_DIR
from optim_factory import get_optimizer
from resume_utils import restore_training_state


def auto_device() -> torch.device:
    """Select CUDA, then MPS (macOS), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def normalize_arch(name: str | None) -> str:
    """Normalize architecture name to a canonical registry key."""
    return str(name or "model").lower().replace("/", "_").replace(" ", "_")


def get_task_flags(task: str | None) -> Tuple[bool, bool]:
    """Return (has_cls, has_seg) from task label."""
    t = (task or "multitask").lower()
    return t != "segmentation", t != "classification"


def dict_safe(loss: torch.nn.Module | Callable):
    """Wrap a torch loss to accept dict(pred/target) or raw tensors."""
    if not isinstance(loss, torch.nn.Module):
        return loss

    bce = torch.nn.BCEWithLogitsLoss

    def _fn(pred, tgt):
        # logits
        pred_t = pred.get("cls_out", pred.get("class_logits", pred)) if isinstance(pred, dict) else pred
        # target
        tgt_t = tgt.get("label", tgt.get("y", tgt)) if isinstance(tgt, dict) else tgt
        if isinstance(loss, bce):
            return loss(pred_t, torch.as_tensor(tgt_t).float().view_as(pred_t))
        return loss(pred_t, torch.as_tensor(tgt_t).long().view(-1))

    return _fn


def configure_wandb_step_semantics() -> None:
    """Unify step semantics: both train/* and val/* step on global_step."""
    try:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("val_*", step_metric="epoch")
    except Exception:
        pass


def run(cfg: dict) -> None:
    # setup
    set_determinism(seed=42)
    device = auto_device()
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
        print(
            f"[INFO] Train: {len(train_loader.dataset)} | "
            f"Val: {len(val_loader.dataset)} | "
            f"Test: {len(test_loader.dataset)}"
        )

    # model
    arch = normalize_arch(cfg.get("architecture"))
    try:
        wrapper = MODEL_REGISTRY[arch]()  # model wrapper
    except KeyError as e:
        raise ValueError(f"Unknown architecture '{arch}'.") from e

    model = wrapper.build_model(cfg).to(device)

    # optim
    optimizer = get_optimizer(cfg, model)

    # task + metrics
    task = (cfg.get("task") or wrapper.get_supported_tasks()[0]).lower()
    has_cls, has_seg = get_task_flags(task)
    metrics = wrapper.get_metrics(config=cfg, task=task, has_cls=has_cls, has_seg=has_seg)

    # loss (dict-safe)
    loss_fn = dict_safe(wrapper.get_loss_fn(task, cfg))

    # batch prep
    prepare_batch = make_prepare_batch(
        task=task,
        debug=cfg.get("debug"),
        binary_label_check=True,
        num_classes=int(cfg.get("num_classes", 2)),
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

    # scheduler
    scheduler = attach_scheduler(
        cfg=cfg,
        trainer=trainer,
        evaluator=evaluator,
        optimizer=optimizer,
        train_loader=train_loader,
        use_monai_handler=False,
    )

    # optional one-shot debug
    if cfg.get("debug_batch_once", bool(cfg.get("debug", False))):
        from debug_utils import debug_batch
        debug_batch(next(iter(train_loader)), model)

    # handlers
    handler_kwargs = {
        **wrapper.get_handler_kwargs(),
        "add_classification_metrics": has_cls,
        "add_segmentation_metrics": has_seg,
    }

    # checkpoints
    run_id = str(cfg.get("run_id") or (wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")))
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_id)

    # resume
    last_epoch = restore_training_state(
        dirname=ckpt_dir,
        prefix=arch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        map_location=device.type if device.type != "cuda" else "cuda",
    )
    if last_epoch is not None:
        print(f"[resume] restored training state from epoch {last_epoch}")

    register_handlers(
        cfg=cfg,
        trainer=trainer,
        evaluator=evaluator,
        model=model,
        save_dir=ckpt_dir,
        prefix=arch,
        wandb_log_handler=wandb_log_handler,
        metric_names=list(metrics.keys()),
        **handler_kwargs,
    )

    print("[DEBUG] max_epochs (pre-run) =", trainer.state.max_epochs)

    # train
    trainer.run()

    # final export
    if cfg.get("save_final_state", True):
        os.makedirs("outputs/best_model", exist_ok=True)
        save_path = f"outputs/best_model/{arch}_{wandb.run.id}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Training complete. Saved: {save_path}")


def main(config: dict | None = None) -> None:
    with wandb.init(config=config or {}, dir="outputs") as wb_run:
        cfg = load_and_validate_config(dict(wandb.config))
        cfg.setdefault("run_id", wb_run.id)
        cfg.setdefault("run_name", wb_run.name)
        configure_wandb_step_semantics()
        run(cfg)


if __name__ == "__main__":
    main()
