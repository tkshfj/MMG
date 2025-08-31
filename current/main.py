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

import torch
from monai.utils import set_determinism
from ignite.engine import Events

from config_utils import load_and_validate_config
from data_utils import build_dataloaders
from model_registry import build_model, make_default_loss
from engine_utils import (build_trainer, make_prepare_batch, attach_lr_scheduling, read_current_lr, get_label_indices_from_batch,
                          build_evaluator, attach_val_threshold_search, attach_early_stopping, attach_best_checkpoint,)
from evaluator_two_pass import attach_two_pass_validation
from metrics_utils import make_cls_val_metrics
from optim_factory import get_optimizer, make_scheduler
from handlers import register_handlers, configure_wandb_step_semantics
from constants import CHECKPOINT_DIR
from resume_utils import restore_training_state

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
    # build torch.nn.Module via the registry adapter
    model = build_model(cfg).to(device=device, dtype=torch.float32)

    # optim
    optimizer = get_optimizer(cfg, model)

    # flags
    task = str(cfg.get("task", "multitask")).lower()
    has_cls, has_seg = get_task_flags(task)

    # two prepare_batch functions
    prepare_batch_std = make_prepare_batch(
        task=task,
        debug=cfg.get("debug"),
        binary_label_check=True,
        num_classes=int(cfg.get("num_classes", 2)),
    )

    # loss (generic, uses class_counts for weights/pos_weight)
    loss_fn = make_default_loss(cfg, class_counts=cfg.get("class_counts"))

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

    # Standard evaluator used by the two-pass hook (no direct logging here)
    std_evaluator = build_evaluator(
        device=device,
        data_loader=val_loader,
        network=model,
        prepare_batch=prepare_batch_std,
        metrics=None,            # (optional) attach metrics if we have a factory
        include_seg=has_seg,
        trainer_for_logging=trainer,
    )
    # Optionally compute per-epoch threshold & confusion stats for classification
    if has_cls:
        attach_val_threshold_search(evaluator=std_evaluator, mode="acc")
        val_metrics = make_cls_val_metrics(
            num_classes=int(cfg.get("num_classes", 2)),
            decision=str(cfg.get("cls_decision", "threshold")),
            threshold=float(cfg.get("cls_threshold", 0.5)),
            positive_index=int(cfg.get("positive_index", 1)),
        )
        for k, m in val_metrics.items():
            m.attach(std_evaluator, k)

    two_pass_enabled = bool(cfg.get("two_pass_val", False))
    log_calibrated = bool(cfg.get("log_calibrated", True))
    default_watch = "auc" if has_cls else "dice"
    watch_metric = cfg.get("watch_metric", default_watch)
    cal_warmup = int(cfg.get("cal_warmup_epochs", 1))   # for 2-pass gate
    es_warmup = int(cfg.get("cal_warmup_epochs", 1))   # for early stop (or a separate cfg key)

    def _cast_loggable(v):
        try:
            import numpy as _np
            import torch as _torch
            if isinstance(v, (int, float, bool, str)):
                return v
            if _torch.is_tensor(v):
                return float(v.item()) if v.numel() == 1 else v.detach().cpu().tolist()
            if isinstance(v, _np.ndarray):
                return float(v.item()) if v.size == 1 else v.tolist()
            return v
        except Exception:
            return v

    if not two_pass_enabled:
        @trainer.on(Events.EPOCH_COMPLETED)
        def _run_std_val(engine):
            # 1) run standard evaluator
            std_evaluator.run(val_loader)

            # 2) build a flat W&B payload with explicit "val/" prefix
            ep = int(engine.state.epoch)
            payload = {"trainer/epoch": ep}

            # evaluator metrics (bare names) → "val/<name>"
            for k, v in (std_evaluator.state.metrics or {}).items():
                payload[f"val/{k}"] = _cast_loggable(v)

            # (optional) include val loss for dashboards
            try:
                vloss = _compute_loss(model, val_loader, loss_fn, prepare_batch_std, device)
                payload["val/loss"] = float(vloss)
            except Exception:
                pass

            if len(payload) > 1:
                wandb.log(payload)
    else:
        # Preferred path: use a TwoPass evaluator with .validate(...) if available
        two_pass = None
        _mk = None
        try:
            from evaluator_two_pass import make_two_pass_evaluator as _mk  # prefer a factory if present
        except Exception:
            _mk = None

        if _mk is not None:
            try:
                two_pass = _mk(
                    calibrator=None,  # pass Calibrator instance if we enable it
                    task=cfg.get("task", "multitask"),
                    trainer=trainer,
                    positive_index=int(cfg.get("positive_index", 1)),
                    cls_decision=str(cfg.get("cls_decision", "threshold")),
                    cls_threshold=float(cfg.get("cls_threshold", 0.5)),
                    num_classes=int(cfg.get("num_classes", 2)),
                    multitask=str(cfg.get("task", "multitask")).lower() == "multitask",
                )
            except Exception:
                two_pass = None

        if two_pass is not None:
            @trainer.on(Events.EPOCH_COMPLETED)
            def _run_two_pass(engine):
                ep = int(engine.state.epoch)
                if ep < int(cfg.cal_warmup_epochs):
                    return
                # keep std_evaluator fresh for early-stop/ckpt
                std_evaluator.run(val_loader)

                # two-pass returns: threshold, cls_metrics (dict), seg_metrics (dict)
                t, cls_m, seg_m = two_pass.validate(
                    epoch=ep, model=model, val_loader=val_loader, base_rate=None
                )

                payload = {"trainer/epoch": ep}
                if log_calibrated and t is not None:
                    try:
                        payload["val/cal_thr"] = float(t)
                    except Exception:
                        pass

                # merge cls + seg metrics, prefixing with "val/"
                for k, v in (cls_m or {}).items():
                    payload[f"val/{k}"] = _cast_loggable(v)
                for k, v in (seg_m or {}).items():
                    payload.setdefault(f"val/{k}", _cast_loggable(v))

                # (optional) include val loss
                try:
                    vloss = _compute_loss(model, val_loader, loss_fn, prepare_batch_std, device)
                    payload["val/loss"] = float(vloss)
                except Exception:
                    pass

                if len(payload) > 1:
                    wandb.log(payload)

        else:
            # Fallback path: keep existing attach_two_pass_validation(..)
            # It runs std_evaluator under the hood and merges metrics into trainer.state.metrics
            def _two_pass_collect(engine):
                std_evaluator.run(val_loader)
                em = engine.state.metrics = (engine.state.metrics or {})
                for k, v in (std_evaluator.state.metrics or {}).items():
                    # scalarize safely
                    if hasattr(v, "item") and callable(v.item):
                        v = float(v.item())
                    em[k] = v

            attach_two_pass_validation(
                trainer=trainer,
                two_pass_fn=_two_pass_collect,
                cfg=cfg,
            )

            @trainer.on(Events.EPOCH_COMPLETED)
            def _log_two_pass(engine):
                ep = int(engine.state.epoch)
                if ep < cal_warmup:
                    return

                payload = {"trainer/epoch": ep}
                # trainer.state.metrics now contains the latest validation metrics
                for k, v in (engine.state.metrics or {}).items():
                    payload[f"val/{k}"] = _cast_loggable(v)

                # (optional) include val loss
                try:
                    vloss = _compute_loss(model, val_loader, loss_fn, prepare_batch_std, device)
                    payload["val/loss"] = float(vloss)
                except Exception:
                    pass

                if len(payload) > 1:
                    wandb.log(payload)

    # watch_metric = cfg.get("watch_metric", "bal_acc")
    watch_metric = cfg.get("watch_metric", "auc")
    watch_mode = cfg.get("watch_mode", "max")
    patience = int(cfg.get("early_stop_patience", 5))

    # Early stopping on evaluator completion (warmup-safe)
    attach_early_stopping(
        trainer=trainer,
        evaluator=std_evaluator,
        metric_key=watch_metric,
        mode=watch_mode,
        patience=patience,
        warmup_epochs=es_warmup,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def _store_loss(engine):
        loss = _compute_loss(model, val_loader, loss_fn, prepare_batch_std, device)
        engine.state.metrics = engine.state.metrics or {}
        # engine.state.metrics["loss"] = float(loss)
        engine.state.metrics["val_loss"] = float(loss)

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

    # Best checkpoint (model only, extend 'objects' as needed)
    attach_best_checkpoint(
        trainer, std_evaluator, model, optimizer,
        save_dir=CHECKPOINT_DIR,
        filename_prefix="best",
        watch_metric=watch_metric,
        watch_mode=watch_mode,
        n_saved=1,
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

    run_id = str(cfg.get("run_id", time.strftime("%Y%m%d-%H%M%S")))
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
        evaluator=std_evaluator,
        model=model,
        optimizer=optimizer,
        out_dir=getattr(cfg, "out_dir", getattr(cfg, "output_dir", "./outputs")),
        watch_metric=watch_metric,
        watch_mode=watch_mode,
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
        save_path = f"outputs/best_model/{arch}_{run_id}.pth"
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
