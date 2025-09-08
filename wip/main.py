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
from evaluator_two_pass import make_two_pass_evaluator  # attach_two_pass_validation
from calibration import build_calibrator
from metrics_utils import make_cls_val_metrics
from optim_factory import get_optimizer, make_scheduler
from handlers import register_handlers, configure_wandb_step_semantics
from constants import CHECKPOINT_DIR
from resume_utils import restore_training_state
from posprob import PosProbCfg

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
    import numpy as _np
    out = {}
    for k, v in (metrics_dict or {}).items():
        key = f"{prefix}{k}"
        if "confmat" in k:
            try:
                if torch.is_tensor(v):
                    arr = v.detach().cpu().to(torch.int64).numpy()
                elif isinstance(v, _np.ndarray):
                    arr = v.astype(_np.int64)
                else:
                    arr = _np.asarray(v, dtype=_np.int64)
                if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 1:
                    out[key] = arr.tolist()
                    for i in range(arr.shape[0]):
                        for j in range(arr.shape[1]):
                            out[f"{key}_{i}{j}"] = int(arr[i, j])
                else:
                    # Fallback: store as-is without per-cell scalars
                    out[key] = arr.tolist() if arr.ndim > 0 else int(arr)
            except Exception:
                out[key] = _to_wandb_value(v)
        else:
            out[key] = _to_wandb_value(v)
    return out


def _console_print_metrics(epoch: int, metrics: dict):
    def _as_scalar(x):
        if torch.is_tensor(x) and x.numel() == 1:
            return float(x.item())
        return x

    # format a few scalars nicely
    parts = []
    # for k in ("acc", "prec", "recall", "auc", "pos_rate", "gt_pos_rate"):
    for k in ("acc", "precision_pos", "recall_pos", "auc", "pos_rate", "gt_pos_rate"):
        v = metrics.get(k)
        if v is not None:
            v = _as_scalar(v)
            parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # confusion matrix (handle both names)
    for cm_key in ("cls_confmat", "val/confmat", "seg_confmat"):
        cm = metrics.get(cm_key)
        if cm is not None:
            # MetaTensor → Tensor
            if hasattr(cm, "as_tensor"):
                cm = cm.as_tensor()
            # Tensor/ndarray → python list of ints
            if torch.is_tensor(cm):
                cm_list = cm.detach().cpu().to(torch.int64).tolist()
            else:
                cm_list = np.asarray(cm, dtype=np.int64).tolist()

            parts.append(f"{cm_key}: {cm_list}")
            break

    print(f"[INFO] Epoch[{epoch}] Metrics -- " + " ".join(parts))


def is_wandb_sweep() -> bool:
    # WANDB_SWEEP_ID is set for agent processes
    return bool(os.environ.get("WANDB_SWEEP_ID") or os.environ.get("WANDB_AGENT"))


def decision_health_active_from_cfg(cfg: dict) -> bool:
    # disable during sweeps unless explicitly forced
    flag = bool(cfg.get("enable_decision_health", False))
    if is_wandb_sweep() and not bool(cfg.get("force_decision_health_in_sweep", False)):
        return False
    return flag


def _seed_binary_head_bias_from_prior(model: torch.nn.Module, pos_prior: float) -> bool:
    """Set bias=logit(pos_prior) for any 1-logit binary head. Returns True if applied."""
    import math
    p = float(min(max(pos_prior, 1e-6), 1 - 1e-6))
    b = math.log(p / (1.0 - p))
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and getattr(m, "out_features", None) == 1:
            if m.bias is not None:
                with torch.no_grad():
                    m.bias.fill_(b)
            print(f"[INFO] Initialized binary head bias to {b:.4f} (pos_prior={p:.4f}).")
            return True
    return False


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

    # seed head bias so scores aren’t collapsed at start
    pos_idx = int(cfg.get("positive_index", 1))
    cc = cfg.get("class_counts", [1, 1])
    pos_prior = float(cc[pos_idx] / max(1, sum(cc)))
    if bool(cfg.get("binary_single_logit", True)) and bool(cfg.get("init_head_bias_from_prior", True)):
        _seed_binary_head_bias_from_prior(model, pos_prior)

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
    health_on = decision_health_active_from_cfg(cfg)

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
        metrics=None,
        include_seg=has_seg,
    )
    # Seed the evaluator’s live threshold once
    std_evaluator.state.threshold = float(cfg.get("cls_threshold", 0.5))

    # start from base rate instead of 0.5
    if bool(cfg.get("init_threshold_from_prior", False)):
        std_evaluator.state.threshold = float(pos_prior)

    # Single source of truth for scores
    ppcfg = PosProbCfg(
        binary_single_logit=bool(cfg.get("binary_single_logit", True)),
        positive_index=int(cfg.get("positive_index", 1)),
    )
    # Make the score provider discoverable by helpers
    std_evaluator.state.score_fn = ppcfg

    # guardrail knobs
    thr_min = float(cfg.get("thr_min", 0.10))
    thr_max = float(cfg.get("thr_max", 0.90))
    pos_lo = float(cfg.get("thr_posrate_min", 0.05))
    pos_hi = float(cfg.get("thr_posrate_max", 0.95))
    auc_floor = float(cfg.get("cal_auc_floor", 0.55))
    min_tn0 = int(cfg.get("thr_min_tn", 5))
    min_tp_tgt = int(cfg.get("cal_min_tp", 20))
    thr_warmup = int(cfg.get("thr_warmup_epochs", max(3, int(cfg.get("cal_warmup_epochs", 1)))))

    def _min_tp_for_epoch(ep: int) -> int:
        # ramp: 2 → 5 → target
        return 2 if ep < 3 else (5 if ep < 5 else min_tp_tgt)

    def _ok_confmat(cm) -> tuple[bool, int, int]:
        # Accept torch/np/list forms; expect 2x2
        import numpy as _np
        if hasattr(cm, "as_tensor"):
            cm = cm.as_tensor()
        if torch.is_tensor(cm):
            cm = cm.detach().cpu().to(torch.int64).numpy()
        else:
            cm = _np.asarray(cm)
        if cm.ndim != 2 or cm.shape != (2, 2):
            return False, 0, 0
        tn, fp = int(cm[0, 0]), int(cm[0, 1])  # noqa unused
        fn, tp = int(cm[1, 0]), int(cm[1, 1])  # noqa unused
        return True, tn, tp

    # Helper: current live decision threshold (single SoT)
    def get_live_thr() -> float:
        return float(getattr(std_evaluator.state, "threshold", 0.5))

    score_kwargs = {
        "binary_single_logit": ppcfg.binary_single_logit,
        "positive_index": ppcfg.positive_index,
        # keep the legacy knob if search uses it in logits picking
        "binary_bce_from_two_logits": bool(cfg.get("binary_bce_from_two_logits", False)),
    }

    attach_val_threshold_search(
        evaluator=std_evaluator,
        mode=str(cfg.get("calibration_method", "bal_acc")),  # or "f1"
        positive_index=ppcfg.positive_index,
        score_kwargs=score_kwargs,
    )

    if bool(cfg.get("debug_pos_once", False)) or bool(cfg.get("debug", False)):
        attach_pos_score_debug_once(
            evaluator=std_evaluator,
            bins=int(cfg.get("pos_hist_bins", 20)),
            tag=str(cfg.get("pos_debug_tag", "debug_pos")),
            positive_index=ppcfg.positive_index,
            score_kwargs=score_kwargs,
        )

    # Optionally compute per-epoch threshold & confusion stats for classification
    if has_cls:
        val_metrics = make_cls_val_metrics(
            num_classes=int(cfg.get("num_classes", 2)),
            decision=str(cfg.get("cls_decision", "threshold")),
            threshold=get_live_thr,
            positive_index=int(cfg.get("positive_index", 1)),
            score_provider=ppcfg,
        )
        for k, m in val_metrics.items():
            m.attach(std_evaluator, k)

    # When we have segmentation but two-pass is disabled, attach seg metrics here
    if has_seg and not bool(cfg.get("two_pass_val", False)):
        try:
            # Prefer project-local helper if it exists
            from seg_eval import make_seg_val_metrics  # util, if present
            seg_metrics = make_seg_val_metrics(
                num_classes=int(cfg.get("num_classes", 2)),
                threshold=float(cfg.get("seg_threshold", 0.5)),
            )
            # Keep a "seg/" namespace for clarity
            for k, m in seg_metrics.items():
                m.attach(std_evaluator, f"seg/{k}")
        except Exception:
            # Fallback: do nothing rather than crash; two-pass path will still merge seg metrics
            pass

    # scheduler/watch setup
    two_pass_enabled = bool(cfg.get("two_pass_val", False))
    # default_watch = "auc" if has_cls else "dice"
    default_watch = "multi" if has_seg else "auc"
    watch_metric = cfg.get("watch_metric", default_watch)

    # single source for warmup/log toggles
    cal_warmup = int(cfg.get("cal_warmup_epochs", 1))
    console_log = bool(cfg.get("console_epoch_log", True))

    # optional calibrator + two-pass evaluator (build once)
    two_pass = None
    if two_pass_enabled:
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
            enable_decision_health=health_on,
            score_provider=ppcfg,
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def _validate_and_log(engine):
        ep = int(engine.state.epoch)

        # (1) Optional calibration / two-pass
        seg_metrics_from_tp = {}
        cls_metrics_from_tp = {}
        t_proposed = None
        if two_pass is not None:
            t, cls_metrics_from_tp, seg_metrics_from_tp = two_pass.validate(
                epoch=ep, model=model, val_loader=val_loader, base_rate=None
            )
            # if t is not None and ep >= cal_warmup:
            #     std_evaluator.state.threshold = float(t)

            # uarded adoption of calibrator threshold
            if t_proposed is not None and ep >= cal_warmup and ep >= thr_warmup:
                cand = float(min(max(t_proposed, thr_min), thr_max))
                pr = cls_metrics_from_tp.get("pos_rate", None)
                auc = cls_metrics_from_tp.get("auc", None)
                cm = cls_metrics_from_tp.get("cls_confmat", cls_metrics_from_tp.get("confmat", None))

                pos_ok = isinstance(pr, (int, float)) and (pos_lo <= float(pr) <= pos_hi)
                auc_ok = isinstance(auc, (int, float)) and (float(auc) >= auc_floor)

                tn_tp_ok = True
                if cm is not None:
                    ok, tn, tp = _ok_confmat(cm)
                    if ok:
                        tn_tp_ok = (tn >= min_tn0) and (tp >= _min_tp_for_epoch(ep))

                if pos_ok and auc_ok and tn_tp_ok:
                    std_evaluator.state.threshold = cand
                    # (optional) expose what we actually accepted:
                    cls_metrics_from_tp["threshold"] = cand
                else:
                    # keep previous threshold (do nothing)
                    pass

        # (2) Single standard evaluation (classification metrics + any seg metrics attached to std_evaluator)
        std_evaluator.run(val_loader)

        # (3) Build payload from the standard evaluator first
        metrics = std_evaluator.state.metrics or {}
        payload = {"trainer/epoch": ep}
        payload.update(_flatten_metrics_for_wandb(metrics))  # -> val/...

        # (3b) Merge segmentation metrics returned by two-pass (if any)
        #      We namespace them under val/seg/* to avoid key collisions.
        if has_seg and seg_metrics_from_tp:
            # payload.update(_flatten_metrics_for_wandb(seg_metrics_from_tp, prefix="val/seg/"))
            payload.update(_flatten_metrics_for_wandb(seg_metrics_from_tp, prefix="val/"))

            # Mirror a couple of useful seg values into engine.state.metrics for schedulers/checkpoints
            # Prefer 'dice' if present; otherwise try common alternatives.
            seg_score = None
            for k in ("dice", "mean_dice", "iou", "jaccard"):
                if k in seg_metrics_from_tp:
                    v = seg_metrics_from_tp[k]
                    if torch.is_tensor(v) and v.numel() == 1:
                        v = float(v.item())
                    seg_score = float(v)
                    # Also expose namespaced copy (optional)
                    # engine.state.metrics[f"val/seg/{k}"] = seg_score
                    engine.state.metrics[f"val/{k}"] = seg_score
                    break

            # Construct a combined multi-objective score if configured to do so
            if seg_score is not None:
                # Use AUC if available, else fall back to accuracy (or 0.0)
                cls_score = metrics.get("auc", metrics.get("acc", 0.0))
                if torch.is_tensor(cls_score) and cls_score.numel() == 1:
                    cls_score = float(cls_score.item())
                cls_score = float(cls_score)

                w = float(cfg.get("multi_weight", 0.5))
                multi = w * seg_score + (1.0 - w) * cls_score
                payload["val/multi"] = multi
                engine.state.metrics["val/multi"] = multi
                # make multi and seg metrics visible to checkpoint watch that reads evaluator metrics
                for k, v in seg_metrics_from_tp.items():
                    std_evaluator.state.metrics[f"seg/{k}"] = v
                    std_evaluator.state.metrics["multi"] = multi

        # (3c) If two-pass is OFF but seg metrics were attached to std_evaluator (step 1), also provide val/multi
        elif has_seg:
            # Try to synthesize multi from metrics that were attached to std_evaluator
            seg_score = None
            for k in ("seg/dice", "seg/mean_dice", "seg/iou", "seg/jaccard"):
                if k in metrics:
                    v = metrics[k]
                    if torch.is_tensor(v) and v.numel() == 1:
                        v = float(v.item())
                    seg_score = float(v)
                    break
            if seg_score is not None:
                cls_score = metrics.get("auc", metrics.get("acc", 0.0))
                if torch.is_tensor(cls_score) and cls_score.numel() == 1:
                    cls_score = float(cls_score.item())
                cls_score = float(cls_score)
                w = float(cfg.get("multi_weight", 0.5))
                multi = w * seg_score + (1.0 - w) * cls_score
                payload["val/multi"] = multi
                engine.state.metrics["val/multi"] = multi
                std_evaluator.state.metrics["multi"] = multi

        # (4) Threshold bookkeeping
        thr_search = metrics.get("threshold", None)
        if thr_search is not None:
            payload["val/search_threshold"] = float(thr_search)
        payload["val/threshold"] = float(getattr(std_evaluator.state, "threshold", 0.5))
        if two_pass is not None and ep >= cal_warmup:
            payload["val/cal_thr"] = payload["val/threshold"]

        # (5) Validation loss + mirror into engine state for plateau
        try:
            vloss = float(_compute_loss(model, val_loader, loss_fn, prepare_batch_std, device))
            payload["val/loss"] = vloss
            engine.state.metrics = engine.state.metrics or {}
            engine.state.metrics["val/loss"] = vloss
            # Mirror AUC into engine state if present (optional convenience)
            if "auc" in metrics:
                val_auc = metrics["auc"]
                if torch.is_tensor(val_auc) and val_auc.numel() == 1:
                    val_auc = float(val_auc.item())
                engine.state.metrics["val/auc"] = val_auc
        except Exception:
            pass

        if console_log:
            _console_print_metrics(ep, metrics)

        import wandb
        wandb.log(payload)

    watch_mode = cfg.get("watch_mode", "max")

    # Early stopping on evaluator completion (warmup-safe)
    # attach_early_stopping(
    #     trainer=trainer,
    #     evaluator=std_evaluator,
    #     metric_key=watch_metric,
    #     mode=watch_mode,
    #     patience=patience,
    #     warmup_epochs=es_warmup,
    # )

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
        enable_decision_health=health_on,
    )

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
