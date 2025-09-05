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
from engine_utils import (
    build_trainer, make_prepare_batch, attach_lr_scheduling, read_current_lr,
    get_label_indices_from_batch, build_evaluator, attach_best_checkpoint,
    attach_val_threshold_search, attach_pos_score_debug_once,
)
from evaluator_two_pass import attach_two_pass_validation, make_two_pass_evaluator
from calibration import build_calibrator
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


def sanitize_threshold_keys(cfg: dict, strict: bool = True) -> dict:
    """
    - For classification: only 'cls_threshold' is allowed.
    - For segmentation:   'seg_threshold' is allowed.
    - For LR plateau:     use 'plateau_threshold' (or 'lr_plateau_threshold').
    Any bare 'threshold' key is rejected (or migrated with a warning if strict=False).
    """
    import warnings

    # migrate legacy 'threshold' if present (best-effort)
    if "threshold" in cfg:
        val = cfg.get("threshold")
        # If the LR scheduler is plateau and there's no explicit plateau_threshold,
        # assume the old key belonged to the LR scheduler (most common collision).
        if str(cfg.get("lr_scheduler", cfg.get("lr_strategy", ""))).lower() in ("plateau", "reduce", "reducelronplateau"):
            cfg.setdefault("plateau_threshold", float(val))
        else:
            # Otherwise, assume it was a (badly named) classification decision threshold.
            cfg.setdefault("cls_threshold", float(val))

        # Drop the ambiguous key and warn/error
        cfg.pop("threshold", None)
        msg = "Found legacy key 'threshold'; migrated to "
        msg += "'plateau_threshold' (LR) or 'cls_threshold' (classification). Please remove 'threshold' from configs/sweeps."
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)

    # defaults
    cfg.setdefault("cls_threshold", 0.5)

    return cfg


def _to_wandb_value(v):
    if torch.is_tensor(v):
        v = v.detach().cpu()
        return float(v.item()) if v.numel() == 1 else v.tolist()
    if isinstance(v, np.ndarray):
        return float(v.item()) if v.size == 1 else v.tolist()
    return v


def _flatten_metrics_for_wandb(metrics_dict: dict, prefix: str = "val/") -> dict:
    out = {}
    for k, v in (metrics_dict or {}).items():
        key = f"{prefix}{k}"
        if "confmat" in k:  # handles cls_confmat and seg_confmat
            # store the whole matrix as list AND per-cell scalars for easy charting
            if torch.is_tensor(v):
                cm = v.detach().cpu().to(torch.int64).numpy()
            elif isinstance(v, np.ndarray):
                cm = v.astype(np.int64)
            else:
                cm = np.asarray(v, dtype=np.int64)
            out[key] = cm.tolist()
            C = int(cm.shape[0])
            for i in range(C):
                for j in range(C):
                    out[f"{key}_{i}{j}"] = int(cm[i, j])
        else:
            out[key] = _to_wandb_value(v)
    return out


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

    use_split = str(cfg.get("param_groups", "single")).lower() == "split"

    if use_split and all(hasattr(model, m) for m in ("backbone_parameters", "head_parameters")):
        base_lr = float(cfg.get("base_lr", cfg.get("lr", 1e-3)))
        head_mult = float(cfg.get("head_multiplier", 10.0))
        wd = float(cfg.get("weight_decay", 1e-4))

        bb = list(model.backbone_parameters())
        hd = list(model.head_parameters())

        # If something is wrong with grouping, fall back gracefully
        if len(bb) == 0 or len(hd) == 0:
            print("[WARN] split requested but backbone/head groups are empty; using single group.")
            parameters_for_opt = model.parameters()
        else:
            parameters_for_opt = [
                {"params": bb, "lr": base_lr, "weight_decay": wd},  # backbone
                {"params": hd, "lr": base_lr * head_mult, "weight_decay": wd},  # head
            ]
    else:
        parameters_for_opt = model.parameters()

    optimizer = get_optimizer(cfg, parameters_for_opt, verbose=True)

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

    # live threshold store
    _thr = {"v": float(cfg.get("cls_threshold", 0.5))}

    def get_thr():
        return _thr["v"]

    # Standard evaluator used by the two-pass hook (no direct logging here)
    std_evaluator = build_evaluator(
        device=device,
        data_loader=val_loader,
        network=model,
        prepare_batch=prepare_batch_std,
        metrics=None,
        include_seg=has_seg,
        trainer_for_logging=trainer,
    )
    # Always collect a 1D positive score and per-epoch threshold from validation
    attach_val_threshold_search(
        evaluator=std_evaluator,
        mode=str(cfg.get("calibration_method", "bal_acc")),   # or "f1"
        positive_index=int(cfg.get("positive_index", 1)),
    )
    # Optional: one-shot score histogram / sanity-print on the first validation run
    if bool(cfg.get("debug_pos_once", False)) or bool(cfg.get("debug", False)):
        attach_pos_score_debug_once(
            evaluator=std_evaluator,
            positive_index=int(cfg.get("positive_index", 1)),
            bins=int(cfg.get("pos_hist_bins", 20)),
            tag=str(cfg.get("pos_debug_tag", "debug_pos")),
        )
    # # After building std_evaluator
    # if str(cfg.get("architecture", "")).lower() == "vit" and bool(cfg.get("debug_score_hist", True)):
    #     attach_pos_score_debug_once(
    #         evaluator=std_evaluator,
    #         positive_index=int(cfg.get("positive_index", 1)),
    #         bins=int(cfg.get("debug_score_bins", 20)),
    #         print_fn=print,  # or logger.info
    #     )
    # Optionally compute per-epoch threshold & confusion stats for classification
    if has_cls:
        # attach_val_threshold_search(evaluator=std_evaluator, mode="acc")
        val_metrics = make_cls_val_metrics(
            num_classes=int(cfg.get("num_classes", 2)),
            decision=str(cfg.get("cls_decision", "threshold")),
            threshold=get_thr,
            positive_index=int(cfg.get("positive_index", 1)),
        )
        for k, m in val_metrics.items():
            m.attach(std_evaluator, k)

    two_pass_enabled = bool(cfg.get("two_pass_val", False))
    log_calibrated = bool(cfg.get("log_calibrated", True))  # noqa unused
    default_watch = "auc" if has_cls else "dice"
    watch_metric = cfg.get("watch_metric", default_watch)
    cal_warmup = int(cfg.get("cal_warmup_epochs", 1))   # for 2-pass gate
    es_warmup = int(cfg.get("cal_warmup_epochs", 1))     # noqa unused for early stop (or a separate cfg key)

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
            std_evaluator.run(val_loader)
            ep = int(engine.state.epoch)

            payload = {"trainer/epoch": ep}
            payload.update(_flatten_metrics_for_wandb(std_evaluator.state.metrics))
            # Preserve hook-derived threshold separately
            thr_search = std_evaluator.state.metrics.get("threshold", None)
            if thr_search is not None:
                payload["val/search_threshold"] = float(thr_search)
            # Log the decision threshold actually used by metrics
            payload["val/threshold"] = float(get_thr())
            try:
                payload["val/loss"] = float(_compute_loss(model, val_loader, loss_fn, prepare_batch_std, device))
            except Exception:
                pass
            wandb.log(payload)
    else:
        # use a TwoPass evaluator with .validate(...)
        calibrator = None
        try:
            calibrator = build_calibrator(cfg)
        except Exception:
            calibrator = None

        two_pass = make_two_pass_evaluator(
            calibrator=calibrator,
            task=cfg.get("task", "multitask"),
            trainer=trainer,
            positive_index=int(cfg.get("positive_index", 1)),
            cls_decision=str(cfg.get("cls_decision", "threshold")),
            cls_threshold=float(cfg.get("cls_threshold", 0.5)),
            num_classes=int(cfg.get("num_classes", 2)),
            multitask=str(cfg.get("task", "multitask")).lower() == "multitask",
        )

        def _runner(epoch: int):
            # returns (t, cls_metrics, seg_metrics)
            return two_pass.validate(epoch=epoch, model=model, val_loader=val_loader, base_rate=None)

        attach_two_pass_validation(
            trainer=trainer,
            run_two_pass=_runner,
            cal_warmup_epochs=int(cfg.get("cal_warmup_epochs", 1)),
            disable_std_logging=True,
            wandb_prefix="val/",
            log_fn=wandb.log,
        )

        if two_pass is not None:
            @trainer.on(Events.EPOCH_COMPLETED)
            def _run_two_pass_then_eval(engine):
                ep = int(engine.state.epoch)
                if ep < int(cfg.get("cal_warmup_epochs", 1)):
                    # no calibration yet; still run evaluator with the initial threshold
                    std_evaluator.run(val_loader)
                    # inline logging (or call helper if you have one)
                    payload = {"trainer/epoch": ep}
                    payload.update(_flatten_metrics_for_wandb(std_evaluator.state.metrics))
                    thr_search = std_evaluator.state.metrics.get("threshold", None)
                    if thr_search is not None:
                        payload["val/search_threshold"] = float(thr_search)
                    payload["val/threshold"] = float(get_thr())
                    try:
                        payload["val/loss"] = float(_compute_loss(model, val_loader, loss_fn, prepare_batch_std, device))
                    except Exception:
                        pass
                    wandb.log(payload)
                    return
                # Calibrate to get t
                t, cls_m, seg_m = two_pass.validate(epoch=ep, model=model, val_loader=val_loader, base_rate=None)
                if t is not None:
                    _thr["v"] = float(t)   # update live threshold
                # Evaluate with the updated threshold
                std_evaluator.run(val_loader)
                # Log evaluator metrics (now consistent with cal_thr)
                payload = {"trainer/epoch": ep, "val/cal_thr": _thr["v"]}
                payload.update(_flatten_metrics_for_wandb(std_evaluator.state.metrics))
                thr_search = std_evaluator.state.metrics.get("threshold", None)
                if thr_search is not None:
                    payload["val/search_threshold"] = float(thr_search)
                payload["val/threshold"] = float(get_thr())
                try:
                    payload["val/loss"] = float(_compute_loss(model, val_loader, loss_fn, prepare_batch_std, device))
                except Exception:
                    pass
                wandb.log(payload)
        else:
            # Fallback path: keep existing attach_two_pass_validation(..)
            # It runs std_evaluator under the hood and merges metrics into trainer.state.metrics
            def _two_pass_collect(engine):
                # std_evaluator.run(val_loader)
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

    watch_metric = cfg.get("watch_metric", "auc")
    watch_mode = cfg.get("watch_mode", "max")
    patience = int(cfg.get("early_stop_patience", 5))  # noqa unused

    # Early stopping on evaluator completion (warmup-safe)
    # attach_early_stopping(
    #     trainer=trainer,
    #     evaluator=std_evaluator,
    #     metric_key=watch_metric,
    #     mode=watch_mode,
    #     patience=patience,
    #     warmup_epochs=es_warmup,
    # )

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
        plateau_metric=str(cfg.get("plateau_metric", "val/loss")),
        plateau_source="trainer",
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
        attach_validation=False,
        validation_interval=1,
        val_loader=val_loader,
    )

    # print("[DEBUG] max_epochs (pre-run) =", trainer.state.max_epochs)
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
        cfg = sanitize_threshold_keys(cfg, strict=True)
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
