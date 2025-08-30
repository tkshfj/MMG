# main.py
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # cuBLAS deterministic
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)

import time
from typing import Tuple
from collections.abc import Callable

import torch
from monai.utils import set_determinism
from ignite.engine import Events

from config_utils import load_and_validate_config
from data_utils import build_dataloaders
from model_registry import MODEL_REGISTRY
from engine_utils import build_trainer, make_prepare_batch, attach_lr_scheduling, read_current_lr, get_label_indices_from_batch
from evaluator_two_pass import attach_two_pass_validation
from optim_factory import get_optimizer, make_scheduler
from handlers import register_handlers, configure_wandb_step_semantics
from constants import CHECKPOINT_DIR
from resume_utils import restore_training_state
from calibrator import Calibrator, CalConfig

# determinism (CPU side + algorithm choices), safe pre-fork
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_determinism(seed=seed)


def enforce_fp32_policy():
    # Call this AFTER DataLoaders are built (to avoid early CUDA init)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def auto_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def normalize_arch(name: str | None) -> str:
    return str(name or "model").lower().replace("/", "_").replace(" ", "_")


def get_task_flags(task: str | None) -> Tuple[bool, bool]:
    t = (task or "multitask").lower()
    return t != "segmentation", t != "classification"


def dict_safe(loss: torch.nn.Module | Callable):
    if not isinstance(loss, torch.nn.Module):
        return loss
    bce = torch.nn.BCEWithLogitsLoss

    def _fn(pred, tgt):
        pred_t = pred.get("cls_out", pred.get("class_logits", pred)) if isinstance(pred, dict) else pred
        tgt_t = tgt.get("label", tgt.get("y", tgt)) if isinstance(tgt, dict) else tgt
        if isinstance(loss, bce):
            val = loss(pred_t, torch.as_tensor(tgt_t).float().view_as(pred_t))
        else:
            val = loss(pred_t, torch.as_tensor(tgt_t).long().view(-1))
        # normalize to dict for engine_utils.coerce_loss_dict
        return {"loss": val}
    return _fn


@torch.no_grad()
def _compute_loss(model, val_loader, loss_fn, prepare_batch, device) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    pin = bool(getattr(val_loader, "pin_memory", False))
    for batch in val_loader:
        inputs, targets = prepare_batch(batch, device, pin)
        outputs = model(inputs)
        # accept tensor OR dict; normalize via coerce_loss_dict
        loss_out = loss_fn(outputs, targets)
        try:
            from engine_utils import coerce_loss_dict as _coerce  # local import ok
            loss_map = _coerce(loss_out)
            loss_t = loss_map["loss"]
        except Exception:
            loss_t = loss_out
        loss = float(loss_t.detach().cpu().item()) if torch.is_tensor(loss_t) else float(loss_t)
        total += loss
        count += 1
    if was_training:
        model.train()
    return total / max(1, count)


def run(
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
) -> None:
    # model
    arch = normalize_arch(cfg.get("architecture"))
    try:
        wrapper = MODEL_REGISTRY[arch]()
    except KeyError as e:
        raise ValueError(f"Unknown architecture '{arch}'.") from e

    # resolve class_counts once (prefer cfg, else dataset attribute)
    counts = cfg.get("class_counts", None)
    if counts is None:
        ds = getattr(train_loader, "dataset", None)
        # common attribute names used by custom datasets
        counts = (
            getattr(ds, "class_counts", None) or getattr(ds, "label_counts", None) or getattr(ds, "counts", None)
        )
        if counts is not None:
            cfg["class_counts"] = counts  # persist for downstream consumers
    # expose to wrapper so build_model can init bias from priors
    setattr(wrapper, "class_counts", cfg.get("class_counts"))

    # build concrete torch.nn.Module
    model = wrapper.build_model(cfg).to(device=device, dtype=torch.float32)

    # optim
    optimizer = get_optimizer(cfg, model)

    # flags
    task = (cfg.get("task") or wrapper.get_supported_tasks()[0]).lower()
    has_cls, has_seg = get_task_flags(task)

    # two prepare_batch functions
    prepare_batch_std = make_prepare_batch(
        task=task,
        debug=cfg.get("debug"),
        binary_label_check=True,
        num_classes=int(cfg.get("num_classes", 2)),
    )

    # loss with explicit class_counts threading
    loss_fn_core = None
    try:
        loss_fn_core = wrapper.get_loss_fn(task=task, cfg=cfg, class_counts=cfg.get("class_counts"))
    except TypeError:
        # backward-compat with older signatures
        try:
            loss_fn_core = wrapper.get_loss_fn(task, cfg)  # type: ignore
        except TypeError:
            loss_fn_core = wrapper.get_loss_fn()           # type: ignore
    # If a torch.nn.Module is returned, wrap it so training loop can consume dict or tensor
    loss_fn = dict_safe(loss_fn_core)

    # engines
    trainer = build_trainer(
        device=device,
        max_epochs=cfg["epochs"],
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_fn,
        prepare_batch=prepare_batch_std,   # training remains the std path
    )

    # Set cal_init_threshold; fall back to cls_threshold, then threshold
    init_thr = float(cfg.get("cal_init_threshold", cfg.get("cls_threshold", cfg.get("threshold", 0.5))))
    # Two-pass evaluator wiring
    cal = Calibrator(CalConfig(
        method=str(cfg.get("calibration_method", "youden")),
        q_bounds=tuple(cfg.get("cal_q_bounds", (0.10, 0.90))),
        rate_tol=float(cfg.get("cal_rate_tolerance", 0.15)),
        warmup_epochs=int(cfg.get("cal_warmup_epochs", 2)),
        auc_floor=float(cfg.get("cal_auc_floor", 0.52)),
        fallback=str(cfg.get("cal_fallback", "rate_match")),
        init_threshold=init_thr,
        ema_beta=float(cfg.get("cal_ema_beta", 0.2)),
        max_delta=float(cfg.get("cal_max_delta", 0.10)),
        min_tp=int(cfg.get("cal_min_tp", 10)),
        bootstraps=int(cfg.get("cal_bootstraps", 0)),
    ))

    # Two-pass validation: populate trainer.state.metrics each epoch
    attach_two_pass_validation(
        trainer=trainer,
        model=model,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        calibrator=cal,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def _store_loss(engine):
        loss = _compute_loss(model, val_loader, loss_fn, prepare_batch_std, device)
        engine.state.metrics = engine.state.metrics or {}
        engine.state.metrics["loss"] = float(loss)

    # Scheduler: construct + attach (trainer-only)
    scheduler = make_scheduler(cfg=cfg, optimizer=optimizer, train_loader=train_loader)

    # Attach stepping to the trainer only (never evaluator).
    attach_lr_scheduling(
        trainer=trainer,
        evaluator=None,  # two-pass writes into trainer.state.metrics
        optimizer=optimizer,
        scheduler=scheduler,
        plateau_metric=cfg.get("plateau_metric", "val_loss"),
        plateau_mode=cfg.get("plateau_mode", "min"),
    )

    # log LR right AFTER scheduler.step() has run (per epoch).
    @trainer.on(Events.EPOCH_COMPLETED)
    def _log_lr(engine):
        try:
            import wandb
            sched = getattr(engine.state, "_scheduler", None)
            lr = read_current_lr(optimizer, sched)
            if lr is not None:
                wandb.log({"trainer/epoch": int(engine.state.epoch), "opt/lr": float(lr)})
        except Exception:
            pass

    # optional one-shot debug
    if cfg.get("debug_batch_once", bool(cfg.get("debug", False))):
        from debug_utils import debug_batch
        debug_batch(next(iter(train_loader)), model)

    run_id = str(cfg.get("run_id") or (wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")))
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_id)

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
        trainer=trainer,
        evaluator=None,
        model=model,
        optimizer=optimizer,
        out_dir=getattr(cfg, "out_dir", getattr(cfg, "output_dir", "./outputs")),
        watch_metric=getattr(getattr(cfg, "early_stop", {}), "metric", "val_auc"),
        watch_mode=getattr(getattr(cfg, "early_stop", {}), "mode", "max"),
        save_last_n=getattr(cfg, "save_last_n", 2),
        wandb_project=getattr(cfg, "wandb_project", None),
        wandb_run_name=getattr(cfg, "run_name", None),
        log_images=getattr(cfg, "log_images", False),
        image_every_n_epochs=getattr(cfg, "image_every_n_epochs", 1),
        image_max_items=getattr(cfg, "image_max_items", 8),
    )

    print("[DEBUG] max_epochs (pre-run) =", trainer.state.max_epochs)
    trainer.run()

    if cfg.get("save_final_state", True):
        os.makedirs("outputs/best_model", exist_ok=True)
        save_path = f"outputs/best_model/{arch}_{wandb.run.id}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Training complete. Saved: {save_path}")


if __name__ == "__main__":
    import sys
    import multiprocessing as mp
    import wandb

    method = "fork" if sys.platform.startswith("linux") else "spawn"
    # method = "spawn"  # use spawn on all OSes
    try:
        mp.set_start_method(method, force=True)
    except RuntimeError:
        pass
    ctx = mp.get_context(method)

    with wandb.init(config={}, dir="outputs") as wb_run:
        cfg = load_and_validate_config(dict(wandb.config))
        cfg.setdefault("run_id", wb_run.id)
        cfg.setdefault("run_name", wb_run.name)
        configure_wandb_step_semantics()

        # Build DataLoaders FIRST (CPU path), pass context & pin_memory as desired
        train_loader, val_loader, test_loader = build_dataloaders(
            metadata_csv=cfg["metadata_csv"],
            input_shape=cfg.get("input_shape", (256, 256)),
            batch_size=cfg.get("batch_size", 16),
            task=cfg.get("task", "multitask"),
            split=cfg.get("split", (0.7, 0.15, 0.15)),
            num_workers=cfg.get("num_workers", 32),
            debug=bool(cfg.get("debug", False)),
            pin_memory=bool(cfg.get("pin_memory", False)),
            multiprocessing_context=ctx,
            seg_target="indices",
            num_classes=int(cfg.get("num_classes", 2)),
            sampling="weighted",
            seed=42,
        )

        if cfg.get("print_dataset_sizes", True):
            print(f"[INFO] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

        # Only now touch CUDA/TF32 settings and select device
        enforce_fp32_policy()
        device = auto_device()
        cfg["device"] = device

        # derive class_counts from TRAIN set (not val)
        C = int(cfg.get("num_classes", 2))
        train_ds = getattr(train_loader, "dataset", None)
        counts = (
            cfg.get("class_counts") or getattr(train_ds, "class_counts", None) or getattr(train_ds, "label_counts", None) or getattr(train_ds, "counts", None)
        )
        # sanity: length and sum must match TRAIN set
        if counts is not None:
            counts = list(map(int, counts))
            if len(counts) != C or sum(counts) != len(train_ds):
                print(f"[WARN] ignoring provided class_counts={counts} "
                      f"(expected len={C}, sum={len(train_ds)}); recomputing from TRAIN")
                counts = None
        # recompute from TRAIN if needed (single pass)
        if counts is None:
            import torch
            agg = torch.zeros(C, dtype=torch.long)
            for batch in train_loader:
                y = get_label_indices_from_batch(batch, num_classes=C)
                agg += torch.bincount(y, minlength=C)
            counts = agg.tolist()
            print(f"[INFO] computed train class_counts={counts}")
        cfg["class_counts"] = counts

        # Run training with explicit loaders & device
        run(cfg, train_loader, val_loader, test_loader, device)
