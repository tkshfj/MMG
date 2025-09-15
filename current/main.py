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
import torch
import torch.nn as nn
from monai.utils import set_determinism
from ignite.engine import Events

from config_utils import load_and_validate_config
from data_utils import build_dataloaders
from models.model_registry import build_model
from engine_utils import (
    build_trainer, make_prepare_batch, attach_lr_scheduling, read_current_lr,
    get_label_indices_from_batch, build_evaluator, attach_best_checkpoint, attach_val_stack,
)
from evaluator_two_pass import make_two_pass_evaluator
from calibration import build_calibrator
from optim_factory import get_optimizer, make_scheduler
from handlers import register_handlers, configure_wandb_step_semantics
from constants import CHECKPOINT_DIR
from resume_utils import restore_training_state
from utils.posprob import PosProbCfg

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
    import torch

    def _as_1d_array(v):
        # Returns a 1D numpy array if v looks like a vector; else None
        try:
            if isinstance(v, torch.Tensor):
                t = v.detach().cpu()
                if t.ndim == 1:
                    return t.numpy()
                if t.ndim > 1 and 1 in t.shape and t.numel() == max(t.shape):
                    # squeeze (e.g., [1,C] or [C,1])
                    return t.view(-1).numpy()
            elif isinstance(v, _np.ndarray):
                if v.ndim == 1:
                    return v
                if v.ndim > 1 and 1 in v.shape and v.size == max(v.shape):
                    return v.reshape(-1)
            elif isinstance(v, (list, tuple)):
                # only treat short 1D lists as vectors
                return _np.asarray(v).reshape(-1)
        except Exception:
            pass
        return None

    def _log_classwise(key_base: str, vec, out: dict, *, exclude_bg: bool = False):
        arr = _np.asarray(vec, dtype=_np.float64).reshape(-1)
        # mean across classes; toggle exclude_bg to ignore index 0
        start = 1 if (exclude_bg and arr.size > 1) else 0
        mean_val = float(arr[start:].mean()) if arr[start:].size > 0 else float("nan")
        out[f"{prefix}{key_base}"] = mean_val
        # per-class
        for i, v in enumerate(arr):
            out[f"{prefix}{key_base}_{i}"] = float(v)

    out: dict[str, float | list] = {}

    for raw_k, v in (metrics_dict or {}).items():
        # Strip optional seg/ namespace on log keys
        k = raw_k[4:] if raw_k.startswith("seg/") else raw_k
        key = f"{prefix}{k}"

        # Special-case: confusion matrices -> store matrix (list) + per-cell scalars
        if "confmat" in k:
            try:
                if torch.is_tensor(v):
                    arr = v.detach().cpu().to(torch.int64).numpy()
                elif isinstance(v, _np.ndarray):
                    arr = v.astype(_np.int64)
                else:
                    arr = _np.asarray(v, dtype=_np.int64)
                out[key] = arr.tolist()
                if arr.ndim == 2:
                    for i in range(arr.shape[0]):
                        for j in range(arr.shape[1]):
                            out[f"{key}_{i}{j}"] = int(arr[i, j])
                continue
            except Exception:
                # fall through to generic handling
                pass

        # Classwise handling for dice/iou vectors
        if k in ("dice", "iou"):
            vec = _as_1d_array(v)
            if vec is not None and vec.ndim == 1 and 1 <= vec.size <= 64:
                # Set exclude_bg=True if we want the mean over classes 1..C-1
                _log_classwise(k, vec, out, exclude_bg=False)
                continue  # done with this key

        # Generic scalars
        try:
            out[key] = float(v)  # python/numpy scalars
            continue
        except Exception:
            pass

        # Torch tensors
        if isinstance(v, torch.Tensor):
            t = v.detach()
            if t.ndim == 0 or t.numel() == 1:
                out[key] = float(t.item())
            else:
                # For non-dice/iou multi-element tensors, log mean/min/max
                t = t.float()
                out[f"{key}/mean"] = float(t.mean().item())
                out[f"{key}/min"] = float(t.min().item())
                out[f"{key}/max"] = float(t.max().item())
            continue

        # Numpy arrays (non-dice/iou)
        if isinstance(v, _np.ndarray):
            if v.ndim == 0 or v.size == 1:
                out[key] = float(v.reshape(-1)[0])
            else:
                out[f"{key}/mean"] = float(v.mean())
                out[f"{key}/min"] = float(v.min())
                out[f"{key}/max"] = float(v.max())
            continue

        # Lists/tuples (non-dice/iou): summarize
        if isinstance(v, (list, tuple)):
            try:
                arr = _np.asarray(v, dtype=_np.float64)
                if arr.ndim == 0 or arr.size == 1:
                    out[key] = float(arr.reshape(-1)[0])
                else:
                    out[f"{key}/mean"] = float(arr.mean())
                    out[f"{key}/min"] = float(arr.min())
                    out[f"{key}/max"] = float(arr.max())
                continue
            except Exception:
                pass

        # Fallback: skip non-numeric types
        # out[key] = str(v)  # only if we really want the raw repr

    return out


def _console_print_metrics(epoch: int, metrics: dict):
    import numpy as np
    import torch

    def _as_scalar(x):
        if torch.is_tensor(x) and x.numel() == 1:
            return float(x.item())
        return x

    # local coercion to a 1D numpy vector (no new global helpers)
    def _to_1d_array(v):
        try:
            if isinstance(v, torch.Tensor):
                t = v.detach().cpu()
                if t.ndim == 0:
                    return np.asarray([float(t.item())], dtype=np.float64)
                return np.asarray(t.view(-1), dtype=np.float64)
            arr = np.asarray(v)
            if arr.ndim == 0:
                return np.asarray([float(arr.reshape(-1)[0])], dtype=np.float64)
            return arr.reshape(-1).astype(np.float64, copy=False)
        except Exception:
            return None

    parts = []

    # common scalar metrics (classification-friendly)
    for k in ("acc", "precision_pos", "recall_pos", "auc", "pos_rate", "gt_pos_rate"):
        v = metrics.get(k)
        if v is not None:
            v = _as_scalar(v)
            parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # segmentation: scalar & per-class Dice/IoU (supports plain or seg/* keys)
    dice_key = "dice" if "dice" in metrics else ("seg/dice" if "seg/dice" in metrics else None)
    iou_key = "iou" if "iou" in metrics else ("seg/iou" if "seg/iou" in metrics else None)

    MAX_PER_CLASS = 16  # keep console readable

    if dice_key is not None:
        vec = _to_1d_array(metrics.get(dice_key))
        if vec is not None and vec.size > 0:
            mean_dice = float(vec.mean())
            per = ", ".join(f"{i}:{vec[i]:.4f}" for i in range(min(vec.size, MAX_PER_CLASS)))
            parts.append(f"dice: {mean_dice:.4f} ({per})")

    if iou_key is not None:
        vec = _to_1d_array(metrics.get(iou_key))
        if vec is not None and vec.size > 0:
            mean_iou = float(vec.mean())
            per = ", ".join(f"{i}:{vec[i]:.4f}" for i in range(min(vec.size, MAX_PER_CLASS)))
            parts.append(f"iou: {mean_iou:.4f} ({per})")

    # Confusion matrices: always print segmentation if present; also print classification if present.
    # Map multiple possible keys to normalized labels and print each label once.
    cm_candidates = (
        ("seg/confmat", "seg_confmat"),
        ("seg_confmat", "seg_confmat"),
        ("cls_confmat", "cls_confmat"),
        ("confmat", "cls_confmat"),
        ("val/confmat", "cls_confmat"),
    )
    printed = set()
    for cm_key, norm_label in cm_candidates:
        if norm_label in printed:
            continue
        cm = metrics.get(cm_key)
        if cm is None:
            continue
        if hasattr(cm, "as_tensor"):  # MetaTensor → Tensor
            cm = cm.as_tensor()
        try:
            if torch.is_tensor(cm):
                cm_list = cm.detach().cpu().to(torch.int64).tolist()
            else:
                cm_list = np.asarray(cm, dtype=np.int64).tolist()
            parts.append(f"{norm_label}: {cm_list}")
            printed.add(norm_label)
        except Exception:
            # Fall back to repr if conversion fails
            parts.append(f"{norm_label}: {cm}")
            printed.add(norm_label)

    # Fallback: if nothing recognized, dump raw dict
    if not parts:
        parts.append(", ".join(f"{k}: {v}" for k, v in (metrics or {}).items()))

    print(f"[INFO] Epoch[{epoch}] Metrics -- " + " ".join(parts))


# def _console_print_metrics(epoch: int, metrics: dict):
#     def _as_scalar(x):
#         if torch.is_tensor(x) and x.numel() == 1:
#             return float(x.item())
#         return x

#     # format a few scalars nicely
#     parts = []
#     # for k in ("acc", "prec", "recall", "auc", "pos_rate", "gt_pos_rate"):
#     for k in ("acc", "precision_pos", "recall_pos", "auc", "pos_rate", "gt_pos_rate"):
#         v = metrics.get(k)
#         if v is not None:
#             v = _as_scalar(v)
#             parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

#     # confusion matrix (handle both names)
#     for cm_key in ("cls_confmat", "val/confmat", "seg_confmat"):
#         cm = metrics.get(cm_key)
#         if cm is not None:
#             # MetaTensor → Tensor
#             if hasattr(cm, "as_tensor"):
#                 cm = cm.as_tensor()
#             # Tensor/ndarray → python list of ints
#             if torch.is_tensor(cm):
#                 cm_list = cm.detach().cpu().to(torch.int64).tolist()
#             else:
#                 cm_list = np.asarray(cm, dtype=np.int64).tolist()

#             parts.append(f"{cm_key}: {cm_list}")
#             break

#     print(f"[INFO] Epoch[{epoch}] Metrics -- " + " ".join(parts))


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


def _canon_cls_loss(name: str | None) -> str:
    n = str(name or "bce").lower().replace("-", "").replace("_", "")
    if n in ("bce", "bcewithlogits", "bcelogits"):
        return "bce"
    if n in ("ce", "crossentropy", "crossentropyloss", "crossentropyloss"):
        return "ce"
    return n  # allow custom losses elsewhere to handle themselves


def _infer_head_logits(cfg: dict) -> int:
    """Prefer explicit head_logits; else fall back to legacy binary_single_logit heuristic."""
    if "head_logits" in cfg:
        return int(cfg["head_logits"])
    num_classes = int(cfg.get("num_classes", 2))
    single = bool(cfg.get("binary_single_logit", True)) and (num_classes == 2)
    return 1 if single else num_classes


def run(
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
) -> None:
    # model
    arch = normalize_arch(cfg.get("architecture"))

    # loss x head preflight
    head_logits = _infer_head_logits(cfg)
    cls_loss = _canon_cls_loss(cfg.get("cls_loss", "bce"))
    # Enforce valid pairings for standard binary classification heads:
    if head_logits == 1 and cls_loss != "bce":
        raise ValueError("Config mismatch: head_logits=1 requires a BCE-with-logits style loss (set cls_loss=bce).")
    if head_logits == 2 and cls_loss != "ce":
        raise ValueError("Config mismatch: head_logits=2 requires CrossEntropy (set cls_loss=ce).")
    # Make the evaluator/metrics infer the correct probability mapping
    cfg["head_logits"] = head_logits               # persist normalized value
    cfg["cls_loss"] = cls_loss                     # persist normalized value

    # build torch.nn.Module via the registry adapter
    model = build_model(cfg).to(device=device, dtype=torch.float32)

    def apply_dropout_rate(module: nn.Module, p: float):
        if not (0.0 <= p <= 1.0):
            return
        for m in module.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                m.p = float(p)

    dropout_override = cfg.get("dropout_rate", None)
    if dropout_override is not None:
        apply_dropout_rate(model, float(dropout_override))

    # seed head bias so scores aren’t collapsed at start
    pos_idx = int(cfg.get("positive_index", 1))
    cc = cfg.get("class_counts", [1, 1])
    pos_prior = float(cc[pos_idx] / max(1, sum(cc)))
    if head_logits == 1 and bool(cfg.get("init_head_bias_from_prior", True)):
        _seed_binary_head_bias_from_prior(model, pos_prior)

    if "head_multiplier" in cfg and "head_lr_scale" not in cfg:
        cfg["head_lr_scale"] = float(cfg.pop("head_multiplier"))
    if "base_lr" in cfg and "lr" not in cfg:
        cfg["lr"] = float(cfg.pop("base_lr"))

    # Grouping defaults so the factory will auto-split head/backbone & decay/no-decay
    cfg.setdefault("param_groups", "auto")
    cfg.setdefault("decay_split", True)

    optimizer = get_optimizer(cfg, model, verbose=True)

    # Decide task shape
    task = str(cfg.get("task", "multitask")).lower()
    has_cls = (str(cfg.get("task", "classification")).lower() == "classification") or bool(cfg.get("has_cls", False))
    has_seg = (str(cfg.get("task", "classification")).lower() == "segmentation") or bool(cfg.get("has_seg", False))
    is_mtl = bool(has_cls and has_seg)
    seg_only = has_seg and not has_cls
    health_on = decision_health_active_from_cfg(cfg)

    # two prepare_batch functions
    prepare_batch_std = make_prepare_batch(
        task=task,
        debug=cfg.get("debug"),
        binary_label_check=True,
        num_classes=int(cfg.get("num_classes", 2)),
    )

    # loss (generic, uses class_counts for weights/pos_weight)
    loss_fn = model.get_loss_fn(task=cfg.get("task"), cfg=cfg, class_counts=cfg.get("class_counts"))

    # Build the canonical PosProbCfg once from cfg
    ppcfg = PosProbCfg.from_cfg(cfg)
    # Build engines
    trainer = build_trainer(
        device=device,
        max_epochs=cfg["epochs"],
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_fn,
        prepare_batch=prepare_batch_std,
        # class_counts=class_counts,
    )
    # backbone warm-freeze hook
    if cfg.get("freeze_backbone_warmup", False):
        for p in model.backbone_parameters():
            p.requires_grad = False
        unfreeze_at = int(cfg.get("freeze_backbone_epochs", 2))

        @trainer.on(Events.EPOCH_COMPLETED)
        def _unfreeze_backbone(engine):
            if engine.state.epoch >= unfreeze_at:
                for p in model.backbone_parameters():
                    p.requires_grad = True
                engine.remove_event_handler(_unfreeze_backbone, Events.EPOCH_COMPLETED)

    # Attach single SoT probs to trainer
    ppcfg.attach_to_engine_state(trainer)
    # Build evaluator and attach the same PosProbCfg
    std_evaluator = build_evaluator(
        device=device,
        data_loader=val_loader,
        network=model,
        prepare_batch=prepare_batch_std,
        metrics=None,
        include_seg=has_seg,
        ppcfg=ppcfg,
    )
    # ensure evaluator uses the same instance
    ppcfg.attach_to_engine_state(std_evaluator)
    # Seed the evaluator’s live threshold once
    std_evaluator.state.threshold = float(cfg.get("cls_threshold", 0.5))

    # start from base rate instead of 0.5
    if bool(cfg.get("init_threshold_from_prior", False)):
        std_evaluator.state.threshold = float(pos_prior)

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

    # One-liner: compose the whole validation stack on std_evaluator
    def get_live_thr() -> float:
        return float(getattr(std_evaluator.state, "threshold", cfg.get("cls_threshold", 0.5)))

    attach_val_stack(
        evaluator=std_evaluator,
        num_classes=int(cfg.get("num_classes", 2)),
        ppcfg=ppcfg,
        has_cls=has_cls,
        cls_decision=str(cfg.get("cls_decision", "threshold")),
        threshold_getter=get_live_thr,
        positive_index=int(cfg.get("positive_index", 1)),
        # has_seg=has_seg and not bool(cfg.get("two_pass_val", False)),   # seg metrics here only when 2-pass is OFF
        has_seg=has_seg,
        seg_num_classes=cfg.get("seg_num_classes"),
        seg_threshold=cfg.get("seg_threshold"),
        seg_ignore_index=cfg.get("seg_ignore_index"),
        seg_prefix=("" if seg_only else "seg/"),
        enable_threshold_search=True,
        calibration_method=str(cfg.get("calibration_method", "bal_acc")),
        enable_pos_score_debug_once=bool(cfg.get("debug_pos_once", False) or cfg.get("debug", False)),
        pos_debug_bins=int(cfg.get("pos_hist_bins", 20)),
        pos_debug_tag=str(cfg.get("pos_debug_tag", "debug_pos")),
    )

    # scheduler/watch setup (defaults)
    two_pass_enabled = bool(cfg.get("two_pass_val", False))
    if is_mtl:
        default_watch = "multi"     # weighted blend logged as evaluator.state.metrics["multi"]
        default_plateau = "multi"
    elif has_seg:
        default_watch = "dice"
        default_plateau = "dice"
    else:
        default_watch = "auc"
        default_plateau = "auc"

    # user overrides still respected
    watch_metric = str(cfg.get("watch_metric", default_watch))
    plateau_metric_cfg = str(cfg.get("plateau_metric", default_plateau))

    # Normalize keys for scheduler/checkpoint:
    # - Evaluator stores "multi"/"dice"/"auc" (no "val/" prefix)
    # - Trainer mirrors under "val/*"
    # We will step ReduceLROnPlateau from the evaluator, so pass the *bare* key ("multi"|"dice"|"auc")
    plateau_key_for_eval = plateau_metric_cfg.replace("val/", "").replace("val_", "")

    # For checkpoint, conventionally use "val/<watch_metric>"
    if watch_metric.startswith("val/"):
        watch_key_for_ckpt = watch_metric
    else:
        watch_key_for_ckpt = f"val/{watch_metric}"

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
            cfg=cfg,
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def _validate_and_log(engine):
        ep = int(engine.state.epoch)

        # two-pass (optional)
        seg_metrics_from_tp: dict = {}
        cls_metrics_from_tp: dict = {}
        t_proposed = None
        if two_pass is not None:
            t, cls_metrics_from_tp, seg_metrics_from_tp = two_pass.validate(
                epoch=ep, model=model, val_loader=val_loader, base_rate=None
            )
            t_proposed = t
            # guarded adoption of calibrator threshold
            if t_proposed is not None and ep >= cal_warmup and ep >= thr_warmup:
                cand = float(min(max(float(t_proposed), thr_min), thr_max))
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
                    cls_metrics_from_tp["threshold"] = cand

        # run the evaluator (attach_val_stack already attached seg/cls metrics)
        std_evaluator.run(val_loader)

        # collect metrics and flatten them for W&B
        metrics = std_evaluator.state.metrics or {}

        payload = {"trainer/epoch": ep}
        payload.update(_flatten_metrics_for_wandb(metrics, prefix="val/"))

        # Two-pass: overwrite seg metrics
        seg_only = bool(has_seg and not has_cls)
        if has_seg and seg_metrics_from_tp:
            # Log calibrated seg metrics to W&B
            payload.update(_flatten_metrics_for_wandb(seg_metrics_from_tp, prefix="val/"))

            # Mirror into evaluator & engine for schedulers/console/checkpoints
            seg_ns = "" if seg_only else "seg/"
            for k, v in seg_metrics_from_tp.items():
                std_evaluator.state.metrics[f"{seg_ns}{k}"] = v
                engine.state.metrics[f"val/{k}"] = (float(v.item()) if torch.is_tensor(v) and v.numel() == 1 else v)

            # Optionally compute a combined multi score for MTL (seg + cls)
            if has_cls:
                # seg score: prefer dice, fall back to mean_dice/iou/jaccard
                seg_score = None
                for k in ("dice", "mean_dice", "iou", "jaccard"):
                    if k in seg_metrics_from_tp:
                        v = seg_metrics_from_tp[k]
                        seg_score = (float(v.item()) if torch.is_tensor(v) and v.numel() == 1 else float(v))
                        break
                if seg_score is not None:
                    # cls score: prefer AUC, fallback to ACC; prefer two-pass if provided
                    cls_score = None
                    for ck in ("auc", "acc"):
                        if ck in cls_metrics_from_tp:
                            cv = cls_metrics_from_tp[ck]
                            cls_score = (float(cv.item()) if torch.is_tensor(cv) and cv.numel() == 1 else float(cv))
                            break
                    if cls_score is None:
                        cv = metrics.get("auc", metrics.get("acc", 0.0))
                        if torch.is_tensor(cv):
                            cls_score = float(cv.mean().item()) if cv.numel() > 1 else float(cv.item())
                        else:
                            try:
                                cls_score = float(cv)
                            except Exception:
                                cls_score = 0.0

                    w = float(cfg.get("multi_weight", 0.5))
                    multi = w * seg_score + (1.0 - w) * cls_score
                    payload["val/multi"] = multi
                    std_evaluator.state.metrics["multi"] = multi
                    engine.state.metrics["val/multi"] = multi

        # Mirror a single watch metric into engine.state.metrics for schedulers/checkpoints
        # Defaults: cls->auc, seg->dice, mtl->multi
        _default_watch = "multi" if (has_cls and has_seg) else ("dice" if has_seg else "auc")
        watch_key = str(cfg.get("watch_metric", _default_watch))
        watch_lookup = watch_key.replace("val/", "").replace("val_", "")
        if watch_lookup in metrics:
            v = metrics[watch_lookup]
            if isinstance(v, torch.Tensor):
                v = float(v.detach().mean().item()) if v.numel() > 1 else float(v.item())
            else:
                try:
                    v = float(v)
                except Exception:
                    v = None
            if v is not None:
                engine.state.metrics[f"val/{watch_lookup}"] = v
                # convenient alias when watching seg dice
                if watch_lookup == "seg/dice":
                    engine.state.metrics["val/dice"] = v  # convenient alias

        # validation loss (if we compute it separately)
        try:
            vloss = float(_compute_loss(model, val_loader, loss_fn, prepare_batch_std, device))
            payload["val/loss"] = vloss
            engine.state.metrics["val/loss"] = vloss
        except Exception:
            pass

        # console logging (if we have a pretty-printer)
        if console_log:
            _console_print_metrics(ep, metrics)

        # push to W&B
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
        # plateau_metric=str(cfg.get("plateau_metric", "val/loss")),
        plateau_metric=plateau_key_for_eval,
        plateau_mode=str(cfg.get("plateau_mode", "max")),
        plateau_source="trainer",  # plateau_source="evaluator",
    )

    # Best checkpoint (model only, extend 'objects' as needed)
    attach_best_checkpoint(
        trainer, std_evaluator, model, optimizer,
        save_dir=CHECKPOINT_DIR,
        filename_prefix="best",
        # watch_metric=watch_metric,
        watch_metric=watch_key_for_ckpt,
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

    # trainer.run()
    trainer.run(train_loader)  # max_epochs=cfg["epochs"]

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
            cfg=cfg,
        )

        # infer counts
        from data_utils import infer_class_counts
        num_classes = int(cfg.get("out_channels", cfg.get("num_classes", 2)))
        ds = getattr(train_loader, "dataset", None)
        class_counts = getattr(ds, "class_counts", None)
        if class_counts is None:
            class_counts = infer_class_counts(train_loader, num_classes=num_classes, max_batches=16)
            try:
                # cache on the dataset so engine_utils.build_trainer can pick it up
                setattr(ds, "class_counts", class_counts)
            except Exception:
                pass

        if hasattr(train_loader, "dataset"):
            # ensure it's a plain list of ints/floats, length == num_classes
            train_loader.dataset.class_counts = [int(x) for x in class_counts]

        # logger.info(f"class_counts={class_counts}")
        # Run a single-batch probe when debugging
        if cfg.get("debug", False) or cfg.get("sanity_check_loaders", True):
            from data_utils import one_shot_loader_sanity
            one_shot_loader_sanity(train_loader, val_loader, test_loader, strict=True)

        if cfg.get("print_dataset_sizes", True):
            print(f"[INFO] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

        import warnings
        # import re
        warnings.filterwarnings("ignore", message="Divide by zero (a_min == a_max)")

        # Only now touch CUDA/TF32 settings and select device
        enforce_fp32_policy()
        device = auto_device()
        cfg["device"] = device

        # derive class_counts from TRAIN set (not val)
        C = int(cfg.get("num_classes", cfg.get("out_channels", 2)))
        train_ds = getattr(train_loader, "dataset", None)
        import torch
        agg = torch.zeros(C, dtype=torch.long)  # keep on CPU
        with torch.no_grad():
            for batch in train_loader:
                y = get_label_indices_from_batch(batch, num_classes=C)  # [B] or [B,*spatial]

                # If a one-hot seg mask slipped through, collapse channel dim
                if isinstance(y, torch.Tensor) and y.ndim >= 4 and y.size(1) == C:
                    y = y.argmax(dim=1)
                # Ensure 1-D, integer, valid IDs
                y = torch.as_tensor(y, dtype=torch.long).reshape(-1).cpu()  # 1-D for bincount
                ignore_idx = cfg.get("ignore_index", None)
                if ignore_idx is not None:
                    y = y[y != int(ignore_idx)]
                y = y[(y >= 0) & (y < C)]

                if y.numel():
                    agg += torch.bincount(y, minlength=C)

        counts = agg.tolist()
        print(f"[INFO] computed train class_counts={counts}")
        cfg["class_counts"] = counts

        # Run training with explicit loaders & device
        run(cfg, train_loader, val_loader, test_loader, device)
