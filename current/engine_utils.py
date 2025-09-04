# engine_utils.py
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

import math
import torch
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union, Mapping

from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver
from monai.engines import SupervisedTrainer
from ignite.engine import Events, Engine
from utils.safe import to_float_scalar, is_finite_float, to_tensor
from metrics_utils import (
    extract_cls_logits_from_any,
    extract_seg_logits_from_any,
    coerce_loss_dict,
    make_metrics,
    make_cls_val_metrics,
    make_seg_val_metrics
)


# small engine-local helpers
def _attach_device_state(engine, device, non_blocking: bool = False):
    """Stash device + dataloader pin_memory hint on the engine.state."""
    engine.state.device = device
    engine.state.non_blocking = bool(non_blocking)
    return engine


def _to_device(t: Optional[torch.Tensor], device, non_blocking: bool | None = False) -> Optional[torch.Tensor]:
    """Move tensor to device. Accepts None for non_blocking and coerces to False."""
    if t is None:
        return None
    return t.to(device=device, non_blocking=bool(non_blocking))


def get_first(d: Dict[str, Any], *keys: str) -> Any:
    """Return first present value among keys (no truthiness on tensors)."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def coerce_batch_dict(batch: Union[Dict[str, Any], Sequence[Any]]) -> Dict[str, Any]:
    """
    Support dict, (x, y), and (x, y, m). If y is a dict, merge into output.
    Returns {"image": x, "label": ..., "mask": ...} when present.
    """
    if isinstance(batch, dict):
        return batch
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    n = len(batch)
    if n not in (2, 3):
        raise TypeError(f"Unsupported tuple/list batch length: {n} (expected 2 or 3)")

    x, *rest = batch
    out: Dict[str, Any] = {"image": x}
    if n == 2:
        (y,) = rest
        return {**out, **y} if isinstance(y, dict) else {**out, "label": y}
    y, m = rest
    return ({**out, **y, "mask": m} if isinstance(y, dict) else {**out, "label": y, "mask": m})


def normalize_target(
    y: torch.Tensor,
    *,
    kind: Literal["label", "mask_indices", "mask_channels"],
    float_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Unify label & mask normalization:
      - label         -> [B] long
      - mask_indices  -> [B,H,W] long
      - mask_channels -> [B,1 or C,H,W] float
    """
    y = to_tensor(y)

    if kind == "label":
        # [B,1] -> [B], one-hot/logits -> argmax, then squeeze to [B]
        if y.ndim >= 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        elif y.ndim >= 2 and y.shape[-1] > 1:
            y = y.argmax(dim=-1)
        if y.ndim > 1:
            y = y.squeeze()
        return y.long().view(-1)

    if kind == "mask_indices":
        if y.ndim == 2:              # [H,W] -> [1,H,W]
            y = y.unsqueeze(0)
        if y.ndim == 4:              # [B,1,H,W] -> [B,H,W]; [B,C,H,W] -> argmax
            y = y[:, 0] if y.shape[1] == 1 else y.argmax(dim=1)
        return y.long()

    # kind == "mask_channels"
    if y.ndim == 2:
        y = y.unsqueeze(0).unsqueeze(0)     # [H,W] -> [1,1,H,W]
    elif y.ndim == 3:
        y = y.unsqueeze(1)                  # [B,H,W] -> [B,1,H,W]
    return y.to(dtype=float_dtype)


def _normalize_task(task: Union[str, Task]) -> Task:
    if isinstance(task, Task):
        return task
    t = (task or "").lower()
    if t in {"cls", "classification", "classify"}:
        return Task.CLASSIFICATION
    if t in {"seg", "segmentation", "segment"}:
        return Task.SEGMENTATION
    return Task.MULTITASK


def normalize_mask(
    y: torch.Tensor,
    *,
    target: Literal["indices", "channels"] = "indices",
    num_classes: int | None = None,
    float_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Standardize segmentation targets.

    target="indices":
        Returns [B,H,W] int64 class indices.
        Accepts [H,W], [B,H,W], [B,1,H,W], [B,C,H,W] (argmax if C>1).

    target="channels":
        Returns [B,1,H,W] float for binary, or [B,C,H,W] float for multi-class
        (C = num_classes). If input is indices and num_classes>1, one-hot is produced.
    """
    y = to_tensor(y)

    # Squeeze/add batch/channel dims to predictable rank
    if y.ndim == 2:                # [H, W]
        y = y.unsqueeze(0)         # [1, H, W]
    if y.ndim == 3:                # [B, H, W]  (indices typical)
        pass
    elif y.ndim == 4:              # [B, C, H, W]  (channelled)
        pass
    else:
        raise ValueError(f"normalize_mask: unsupported shape {tuple(y.shape)}")

    if target == "indices":
        # If channelled, reduce to indices
        if y.ndim == 4:
            C = y.shape[1]
            if C == 1:
                y = y[:, 0]        # [B,H,W]
            else:
                # logits/probs or one-hot -> indices
                y = y.argmax(dim=1)  # [B,H,W]
        # Ensure long dtype
        return y.long()

    # target == "channels"
    if y.ndim == 3:
        # indices -> channels
        if (num_classes is not None) and (num_classes > 1):
            # one-hot [B,H,W, C] -> [B,C,H,W]
            y = torch.nn.functional.one_hot(y.long(), num_classes=num_classes).permute(0, 3, 1, 2)
            return y.to(dtype=float_dtype)
        # binary -> [B,1,H,W] float
        y = y.unsqueeze(1)  # [B,1,H,W]
        return y.to(dtype=float_dtype)

    # y.ndim == 4 (already channelled): just cast to float
    return y.to(dtype=float_dtype)


def _debug_print(
    task: Task,
    x: torch.Tensor,
    y_cls: Optional[torch.Tensor],
    y_seg: Optional[torch.Tensor],
    num_classes: int,
    binary_label_check: bool,
) -> None:
    print(f"[prepare_batch] task={task.value}")
    print(f"  image: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    if y_cls is not None:
        mn = int(y_cls.min().item()) if y_cls.numel() else -1
        mx = int(y_cls.max().item()) if y_cls.numel() else -1
        print(f"  label: shape={tuple(y_cls.shape)}, dtype={y_cls.dtype}, min={mn}, max={mx}")
        if binary_label_check and task in (Task.CLASSIFICATION, Task.MULTITASK):
            u = torch.unique(y_cls.detach())
            if u.numel() > 0 and not (u.min().item() >= 0 and u.max().item() < num_classes):
                raise ValueError(f"labels out of range for num_classes={num_classes}: {u.tolist()}")
    if y_seg is not None:
        uniq = torch.unique(y_seg.detach())
        print(f"  mask:  shape={tuple(y_seg.shape)}, dtype={y_seg.dtype}, unique[:6]={uniq[:6].tolist()}")


def read_current_lr(optimizer, scheduler=None) -> Optional[float]:
    """
    Prefer scheduler.get_last_lr() when available; otherwise read from optimizer.
    Works for epoch-based (_LRScheduler) and ReduceLROnPlateau.
    """
    try:
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            last = scheduler.get_last_lr()
            if isinstance(last, (list, tuple)) and len(last) > 0:
                return float(last[0])
            if isinstance(last, (float, int)):
                return float(last)
    except Exception:
        pass
    try:
        return float(optimizer.param_groups[0]["lr"])
    except Exception:
        return None


def is_ignite_engine(obj) -> bool:
    try:
        from ignite.engine import Engine as IgniteEngine
        return isinstance(obj, IgniteEngine)
    except Exception:
        return hasattr(obj, "add_event_handler") and hasattr(obj, "on")


def build_trainer(
    device,
    max_epochs,
    train_data_loader,
    network,
    optimizer,
    loss_function,
    prepare_batch,
    train_metrics: Optional[Dict[str, Any]] = None,
) -> SupervisedTrainer:
    assert callable(loss_function), "loss_function must be callable"
    assert callable(prepare_batch), "prepare_batch must be callable"

    def _iteration_update(engine: Engine, batch):
        network.train()
        dev = getattr(engine.state, "device", device)
        nb = getattr(engine.state, "non_blocking", bool(getattr(train_data_loader, "pin_memory", False)))

        x, targets = prepare_batch(batch, dev, nb)

        optimizer.zero_grad(set_to_none=True)
        preds = network(x)

        # Normalize any loss return shape to a tensor dict with a 'loss' key
        loss_out = loss_function(preds, targets)
        loss_map = coerce_loss_dict(loss_out)  # tensors only, ensures 'loss' exists
        total = loss_map["loss"]
        if not torch.is_tensor(total):
            raise TypeError("'loss' must be a torch.Tensor.")

        total.backward()
        optimizer.step()

        # Convert to plain floats for StatsHandler/W&B/etc.
        out: Dict[str, float] = {}
        # be strict for the primary loss (catch reduction errors)
        main_loss_f = to_float_scalar(total, strict=True)
        if main_loss_f is None:
            main_loss_f = to_float_scalar(total, strict=False)
        out["loss"] = 0.0 if main_loss_f is None else main_loss_f

        for k, v in loss_map.items():
            if k == "loss":
                continue
            f = to_float_scalar(v, strict=False)
            if f is not None:
                out[k] = f

        # optional: log LR (scheduler-aware; no deprecated get_lr())
        sched = getattr(engine.state, "_scheduler", None)
        lr_val = read_current_lr(optimizer, sched)
        if lr_val is not None:
            out["lr"] = lr_val

        return out  # engine.state.output becomes this dict

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_data_loader,
        network=network,
        optimizer=optimizer,
        loss_function=loss_function,
        prepare_batch=prepare_batch,
        iteration_update=_iteration_update,
    )

    # attach device/non_blocking on state AFTER the trainer exists
    _attach_device_state(trainer, device, getattr(train_data_loader, "pin_memory", False))

    # attach training metrics (names MUST be bare, e.g., "auc", "accuracy")
    if train_metrics:
        # idempotency guard (avoid double-attach if rebuild/reuse)
        if not getattr(trainer.state, "_train_metrics_attached", False):
            for name, metric in train_metrics.items():
                metric.attach(trainer, name)
            setattr(trainer.state, "_train_metrics_attached", True)

    return trainer


def build_evaluator(
    *,
    device: torch.device,
    data_loader,                        # required (val loader)
    network: torch.nn.Module,
    prepare_batch: Callable,            # (batch, device, non_blocking) -> (x, targets)
    metrics: Dict[str, Any] | None = None,
    include_seg: bool = True,
    inferer: Optional[Callable] = None,
    non_blocking: Optional[bool] = None,
    trainer_for_logging: Optional[Engine] = None,
) -> Engine:
    """Custom evaluator that returns an out_map dict compatible with metrics."""
    nb_default = bool(non_blocking) if non_blocking is not None else False

    def _eval_step(engine: Engine, batch: Mapping[str, Any]) -> Dict[str, Any]:
        dev = getattr(engine.state, "device", device)
        nb = getattr(engine.state, "non_blocking", nb_default)
        x, targets = prepare_batch(batch, dev, nb)

        # parse targets (can be tensor label, dict with label/mask, etc.)
        y = m = None
        if isinstance(targets, dict):
            y = targets.get("label")
            m = targets.get("mask")
        else:
            y = targets

        with torch.inference_mode():
            net_out = network(x) if inferer is None else inferer(x, network)

        out_map: Dict[str, Any] = {}

        # classification head
        try:
            logits = extract_cls_logits_from_any(net_out)
            if torch.is_tensor(logits):
                logits = logits.float()
            else:
                logits = torch.as_tensor(logits, dtype=torch.float32, device=dev)
            out_map["y_pred"] = logits
            out_map["logits"] = logits
            out_map["cls_out"] = logits
            if y is not None:
                out_map["y"] = torch.as_tensor(y, device=dev).long().view(-1)
        except Exception:
            pass

        # segmentation head
        if include_seg:
            try:
                seg_logits = extract_seg_logits_from_any(net_out)
                out_map["seg_out"] = seg_logits
                out_map["seg_logits"] = seg_logits
            except Exception:
                pass

        # optional packed label dict
        label_dict: Dict[str, Any] = {}
        if y is not None:
            label_dict["label"] = out_map.get("y", torch.as_tensor(y, device=dev).long().view(-1))
        if include_seg and (m is not None):
            label_dict["mask"] = torch.as_tensor(m, device=dev).long()
        if label_dict:
            out_map["label"] = label_dict

        return out_map

    evaluator = Engine(_eval_step)
    pin = bool(getattr(data_loader, "pin_memory", False))
    # Persist device hints on the engine (spawn-safe)
    setattr(evaluator.state, "device", device)
    setattr(evaluator.state, "non_blocking", pin)

    if metrics:
        if not getattr(evaluator.state, "_val_metrics_attached", False):
            for name, metric in metrics.items():
                metric.attach(evaluator, name)
            setattr(evaluator.state, "_val_metrics_attached", True)

    # optional: expose for downstream hooks
    evaluator.state.data_loader = data_loader
    return evaluator


def attach_engine_metrics(
    *,
    trainer: Engine,
    evaluator: Engine,
    tasks: Sequence[Literal["classification", "segmentation"]] = ("classification",),
    num_classes: int,
    cls_decision: str = "threshold",
    cls_threshold: float = 0.5,
    positive_index: int = 1,
    # new, optional seg controls (default to classification settings when not provided)
    seg_num_classes: Optional[int] = None,
    seg_threshold: Optional[float] = None,
    seg_ignore_index: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build and attach metrics to trainer and evaluator with bare names.
    Returns (train_metrics, val_metrics).
    """
    task_set = set(tasks)

    # training metrics (keep light; classification-only by default)
    train_metrics = make_metrics(
        tasks=task_set,
        num_classes=num_classes,
        cls_decision=cls_decision,
        cls_threshold=cls_threshold,
        positive_index=positive_index,
    )
    if not getattr(trainer.state, "_train_metrics_attached", False):
        for k, m in train_metrics.items():
            m.attach(trainer, k)
        setattr(trainer.state, "_train_metrics_attached", True)

    # validation metrics: start with classification
    val_metrics = make_cls_val_metrics(
        num_classes=num_classes,
        decision=cls_decision,
        threshold=cls_threshold,
        positive_index=positive_index,
    )

    # optional: add segmentation validation metrics
    if "segmentation" in task_set:
        seg_nc = int(seg_num_classes) if seg_num_classes is not None else int(num_classes)
        seg_thr = float(seg_threshold) if seg_threshold is not None else float(cls_threshold)
        val_metrics.update(
            make_seg_val_metrics(
                num_classes=seg_nc,
                threshold=seg_thr,
                ignore_index=seg_ignore_index,
            )
        )

    if not getattr(evaluator.state, "_val_metrics_attached", False):
        for k, m in val_metrics.items():
            m.attach(evaluator, k)
        setattr(evaluator.state, "_val_metrics_attached", True)

    return train_metrics, val_metrics


# prepare_batch & scheduler
PrepareBatch = Callable[
    [Union[Dict[str, Any], Sequence[Any]], Optional[torch.device], bool],
    Tuple[torch.Tensor, Any]
]


class Task(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    MULTITASK = "multitask"


def get_label_indices_from_batch(batch: Mapping[str, Any], *, num_classes: int | None = None) -> torch.Tensor:
    """
    Return labels as a 1D LongTensor of class indices [B].
    Accepts keys: 'label', 'labels', or 'y'. Handles one-hot and [B,1].
    """
    if not isinstance(batch, Mapping):
        raise TypeError(f"Expected dict-like batch, got {type(batch)}")

    if "label" in batch and batch["label"] is not None:
        y = batch["label"]
    elif "labels" in batch and batch["labels"] is not None:
        y = batch["labels"]
    elif "y" in batch and batch["y"] is not None:
        y = batch["y"]
    else:
        raise KeyError(f"No label key in batch; keys={list(batch.keys())}")

    if hasattr(y, "as_tensor"):
        y = y.as_tensor()
    y = torch.as_tensor(y)

    # Normalize to indices [B]
    if y.ndim >= 2 and y.shape[-1] > 1:
        y = y.argmax(dim=-1)
    if y.ndim >= 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if y.ndim == 0:
        y = y.view(1)
    y = y.long().view(-1)

    if num_classes is not None:
        if (y < 0).any() or (y >= num_classes).any():
            mn, mx = int(y.min()), int(y.max())
            raise ValueError(f"Label indices out of range 0..{num_classes-1} (min={mn}, max={mx})")

    return y


def make_prepare_batch(
    task: Union[str, Task],
    *,
    debug: bool = False,
    binary_label_check: bool = True,
    num_classes: int = 2,
    seg_target: Literal["indices", "channels"] = "indices",  # = "channels", if default loss is Dice/BCE-style
    seg_float_dtype: Optional[torch.dtype] = None,
) -> PrepareBatch:

    t = _normalize_task(task)
    _printed_once = {"flag": False}

    def _prepare_batch(batch, device: Optional[torch.device] = None, non_blocking: bool | None = False):
        dev = device or torch.device("cpu")
        nb = bool(non_blocking) if non_blocking is not None else False
        bdict = coerce_batch_dict(batch)

        # inputs
        x = get_first(bdict, "image", "images", "x")
        if x is None:
            raise KeyError("Batch missing image tensor (expected one of: 'image', 'images', 'x').")
        x = _to_device(to_tensor(x).float(), dev, nb)

        # labels (classification)
        y_cls = get_first(bdict, "label", "classification", "class", "target", "y")
        if y_cls is not None:
            y_cls = normalize_target(to_tensor(y_cls), kind="label")
            y_cls = _to_device(y_cls, dev, nb)
            # enforce 1D long indices for CE / scalar for BCE-with-logits
            if y_cls.ndim >= 2 and y_cls.shape[-1] > 1:
                y_cls = y_cls.argmax(dim=-1)
            y_cls = y_cls.long().view(-1)

        # masks (segmentation)
        y_seg = get_first(bdict, "mask", "seg", "segmentation", "mask_label", "mask_true")
        if y_seg is not None:
            target_fmt = "channels" if seg_target == "channels" else "indices"
            y_seg = normalize_mask(
                to_tensor(y_seg),
                target=target_fmt,
                num_classes=num_classes,
                float_dtype=seg_float_dtype or torch.float32,
            )
            y_seg = _to_device(y_seg, dev, nb)

        # select targets
        if t is Task.CLASSIFICATION:
            if y_cls is None:
                raise ValueError("prepare_batch(classification): missing 'label'.")
            targets = {"label": y_cls}
        elif t is Task.SEGMENTATION:
            if y_seg is None:
                raise ValueError("prepare_batch(segmentation): missing 'mask'.")
            targets = {"mask": y_seg}
        else:
            if y_cls is None or y_seg is None:
                raise ValueError("prepare_batch(multitask): need both 'label' and 'mask'.")
            targets = {"label": y_cls, "mask": y_seg}

        if debug and not _printed_once["flag"]:
            _debug_print(t, x, y_cls, y_seg, num_classes, binary_label_check)
            _printed_once["flag"] = True

        return x, targets

    return _prepare_batch


def attach_scheduler(
    cfg: Any,
    trainer: Engine,
    evaluator: Optional[Engine],
    optimizer,
    train_loader: Optional[Any] = None,
) -> Optional[object]:
    """
    Creates and attaches LR scheduler to Ignite events.
    Returns the underlying scheduler (or None).
    """
    strategy = str(cfg.get("lr_strategy", "none")).lower()

    if strategy in ("none", "", "off"):
        return None

    if strategy == "cosine":
        T_max = int(cfg.get("T_max", trainer.state.max_epochs))
        eta_min = float(cfg.get("eta_min", 0.0))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        @trainer.on(Events.EPOCH_COMPLETED)
        def _step_cosine(_):
            sched.step()
        return sched

    if strategy == "warmcos":
        # simple warmup to cosine
        T_max = int(cfg.get("T_max", trainer.state.max_epochs))
        eta_min = float(cfg.get("eta_min", 0.0))
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        warmup_factor = float(cfg.get("warmup_start_factor", 0.1))

        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        @trainer.on(Events.EPOCH_STARTED)
        def _warmup(engine):
            e = engine.state.epoch
            if e <= warmup_epochs:
                # linear warmup from factor*base_lr to base_lr
                for pg in optimizer.param_groups:
                    base = pg.get("initial_lr", pg["lr"])
                    pg["lr"] = base * (warmup_factor + (1 - warmup_factor) * (e / max(1, warmup_epochs)))

        @trainer.on(Events.EPOCH_COMPLETED)
        def _step_cosine_after_warmup(engine):
            if engine.state.epoch > warmup_epochs:
                cos.step()

        @trainer.on(Events.STARTED)
        def _seed_initial_lr(_):
            for pg in optimizer.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]

        return cos

    if strategy in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
        mode = str(cfg.get("plateau_mode", "max")).lower()
        patience = int(cfg.get("patience", 5))
        factor = float(cfg.get("factor", 0.5))
        threshold = float(cfg.get("threshold", 1e-4))

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor,
            threshold=threshold,
        )
        logger.info("Use attach_lr_scheduling(...) to step ReduceLROnPlateau from trainer after eval.")
        return sched

    raise ValueError(f"Unknown lr_strategy='{strategy}'")


def _resolve_metric(metrics: dict | None, key: str) -> float | None:
    """Resolve a metric like 'val/loss' with common aliases, in a deterministic order."""
    if not isinstance(metrics, dict) or not metrics:
        return None
    base = key.replace("val/", "").replace("val_", "")
    candidates = [
        key,                       # e.g., "val/loss"
        key.replace("/", "_"),     # "val_loss"
        key.replace("_", "/"),     # "val/loss" (no-op if already)
        f"val/{base}",             # "val/loss" from "loss"
        f"val_{base}",             # "val_loss"
        base,                      # "loss"
    ]
    seen = set()
    for k in candidates:
        if k in seen:
            continue
        seen.add(k)
        if k in metrics:
            try:
                v = float(metrics[k])
                if v == v:  # not NaN
                    return v
            except Exception:
                pass
    return None


def attach_lr_scheduling(
    trainer: Engine,
    evaluator: Engine | None,
    optimizer,
    scheduler,
    *,
    plateau_metric: str | None = None,
    plateau_source: Literal["auto", "evaluator", "trainer"] = "auto",
) -> None:
    if scheduler is None:
        return

    from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

    # Avoid re-attaching handlers (idempotency).
    # Key by scheduler object id so different schedulers can be attached in the same process if needed.
    _guard_key = f"_lr_sched_attached_{id(scheduler)}"
    if getattr(trainer.state, _guard_key, False):
        return
    setattr(trainer.state, _guard_key, True)

    # Make scheduler discoverable for loggers (used by read_current_lr)
    try:
        setattr(trainer.state, "_scheduler", scheduler)
    except Exception:
        pass

    # Attach a single LR logger (once per trainer), regardless of scheduler type.
    if not getattr(trainer.state, "_lr_logger_attached", False):
        @trainer.on(Events.ITERATION_COMPLETED)
        def _log_lr(engine):
            try:
                engine.state.metrics["opt/lr"] = optimizer.param_groups[0]["lr"]
            except Exception:
                pass
        setattr(trainer.state, "_lr_logger_attached", True)

    # Iteration cadence (OneCycle)
    if isinstance(scheduler, OneCycleLR):
        @trainer.on(Events.ITERATION_COMPLETED)
        def _step_onecycle(_):
            scheduler.step()
        return

    # Plateau cadence — prefer evaluator.COMPLETED; else fall back to trainer.EPOCH_COMPLETED
    if isinstance(scheduler, ReduceLROnPlateau):
        key = plateau_metric or "val/loss"

        def _step_with(metrics: dict | None):
            score = _resolve_metric(metrics, key)
            if score is not None:
                try:
                    scheduler.step(score)
                except Exception:
                    pass

        use_eval = (plateau_source == "evaluator") or (plateau_source == "auto" and evaluator is not None)
        if use_eval and evaluator is not None:
            @evaluator.on(Events.COMPLETED)
            def _step_plateau_eval(_):
                _step_with(getattr(evaluator.state, "metrics", None))
        else:
            @trainer.on(Events.EPOCH_COMPLETED)
            def _step_plateau_trainer(engine: Engine):
                # Assumes validation has populated trainer.state.metrics by this point.
                _step_with(getattr(engine.state, "metrics", None))
        return

    # Generic epoch-based schedulers (Cosine, SequentialLR warmcos, StepLR, MultiStep, Exponential, etc.)
    @trainer.on(Events.EPOCH_COMPLETED)
    def _step_epoch(_):
        scheduler.step()


def _make_score_fn(metric_key: str, mode: str = "max"):
    sign = 1.0 if str(mode).lower() == "max" else -1.0

    def _score(engine):
        v = to_float_scalar(engine.state.metrics.get(metric_key), strict=False)
        if not is_finite_float(v):
            # Return worst possible so Checkpoint won't save on bad/NaN metrics
            return -math.inf if sign > 0 else math.inf
        return sign * v
    return _score


def make_warmup_safe_score_fn(
    *, metric_key: str, mode: str, trainer: Engine, warmup_epochs: int
):
    """
    Returns a score_fn(engine)->float that:
      - returns NaN during warmup so EarlyStopping ignores early epochs
      - otherwise delegates to make_score_fn(metric_key, mode)
    """
    base = _make_score_fn(metric_key=metric_key, mode=mode)
    warm = int(warmup_epochs)

    def _score(engine: Engine) -> float:
        tr_epoch = int(getattr(trainer.state, "epoch", 0) or 0)
        if tr_epoch < warm:
            return float("nan")
        return base(engine)

    return _score


def attach_early_stopping(
    *,
    trainer: Engine,
    evaluator: Engine,
    metric_key: str = "val/auc",
    mode: str = "max",
    patience: int = 5,
    warmup_epochs: int = 1,
) -> EarlyStopping:
    """
    Attaches EarlyStopping to the evaluator, warmup-safe and tensor-proof.
    """
    if patience <= 0:
        return None  # disabled

    def _auc_score_fn(engine):
        return float(engine.state.metrics.get("auc", 0.0))
    score_fn = make_warmup_safe_score_fn(metric_key=metric_key, mode=mode, trainer=trainer, warmup_epochs=warmup_epochs)  # noqa: F841
    # handler = EarlyStopping(patience=int(patience), score_function=score_fn, trainer=trainer)
    handler = EarlyStopping(patience=int(patience), score_function=_auc_score_fn, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)
    return handler


def attach_best_checkpoint(
    trainer,
    evaluator,
    model,
    optimizer,
    save_dir: str,
    *,
    filename_prefix: str = "best",
    watch_metric: str = "val/auc",
    watch_mode: str = "max",
    n_saved: int = 1,
) -> None:
    """
    Save 'best' checkpoints based on `watch_metric` computed by the evaluator.
    """
    from ignite.handlers import global_step_from_engine
    saver = DiskSaver(save_dir, create_dir=True, require_empty=False, atomic=True)
    to_save = {"model": model}
    if optimizer is not None:
        to_save["optimizer"] = optimizer
    to_save["trainer"] = trainer  # optional but handy for global_step_from_engine

    checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=saver,
        filename_prefix=filename_prefix,
        n_saved=n_saved,
        score_function=_make_score_fn(watch_metric, watch_mode),
        score_name=watch_metric.replace("/", "_"),
        global_step_transform=global_step_from_engine(trainer),
    )

    # Save after each evaluator run. If we run evaluator from a trainer hook,
    # attach to evaluator.COMPLETED so it only triggers once metrics are ready.
    evaluator.add_event_handler(Events.COMPLETED, checkpoint)


def _best_threshold(
    logits: torch.Tensor, y_true: torch.Tensor, mode: str = "bal_acc"
) -> tuple[float, dict]:
    """
    logits: (N,) raw logits for positive class
    y_true: (N,) {0,1}
    Returns (threshold, stats_dict)
    """
    with torch.no_grad():
        p = torch.sigmoid(logits.float()).cpu()
        y = y_true.long().cpu()
        # candidate thresholds: unique preds + endpoints
        cand = torch.quantile(p, torch.linspace(0, 1, 101)).unique()
        best_thr, best_score, best_stats = 0.5, -1.0, {}

        for thr in cand:
            y_hat = (p >= thr).long()
            tp = int(((y_hat == 1) & (y == 1)).sum())
            tn = int(((y_hat == 0) & (y == 0)).sum())
            fp = int(((y_hat == 1) & (y == 0)).sum())
            fn = int(((y_hat == 0) & (y == 1)).sum())

            prec = tp / max(tp + fp, 1)
            rec1 = tp / max(tp + fn, 1)   # recall for class 1
            rec0 = tn / max(tn + fp, 1)   # recall for class 0
            bal_acc = 0.5 * (rec0 + rec1)
            f1 = 2 * prec * rec1 / max(prec + rec1, 1e-8)
            pos_rate = (tp + fp) / max(len(y), 1)

            score = bal_acc if mode == "bal_acc" else f1
            if score > best_score:
                best_score = float(score)
                best_thr = float(thr)
                best_stats = dict(
                    tp=tp, tn=tn, fp=fp, fn=fn,
                    prec=float(prec), rec=float(rec1),
                    rec_0=float(rec0), rec_1=float(rec1),
                    bal_acc=float(bal_acc), f1=float(f1),
                    pos_rate=float(pos_rate),
                )

        return best_thr, best_stats


def attach_val_threshold_search(
    *,
    evaluator: Engine,
    mode: str = "bal_acc",   # "bal_acc" or "f1"
    logger=None,
):
    """
    Collects logits/targets across the validation epoch, selects a per-epoch threshold,
    and writes: val_threshold, val_prec, val_recall, val_cls_confmat, etc. into metrics.
    """
    buf = {"logits": [], "targets": []}

    @evaluator.on(Events.EPOCH_STARTED)
    def _reset_buffers(_):
        buf["logits"].clear()
        buf["targets"].clear()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def _collect(engine):
        # engine.state.output must be (logits, y_true) or {"y_pred": logits, "y": y_true}
        out = engine.state.output
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits, y_true = out[0], out[1]
        elif isinstance(out, dict) and "y_pred" in out and "y" in out:
            logits, y_true = out["y_pred"], out["y"]
        else:
            return
        buf["logits"].append(logits.detach().flatten())
        buf["targets"].append(y_true.detach().flatten())

    @evaluator.on(Events.EPOCH_COMPLETED)
    def _choose_threshold(engine):
        if not buf["logits"]:
            return
        logits = torch.cat(buf["logits"], dim=0)
        y_true = torch.cat(buf["targets"], dim=0)

        thr, stats = _best_threshold(logits, y_true, mode=mode)
        m = engine.state.metrics
        m["threshold"] = thr

        # write confusion-matrix-like fields compatible with logs
        tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
        m["cls_confmat"] = [[tn, fp], [fn, tp]]
        m["cls_confmat_00"] = float(tn)
        m["cls_confmat_01"] = float(fp)
        m["cls_confmat_10"] = float(fn)
        m["cls_confmat_11"] = float(tp)
        m["prec"] = stats["prec"]
        m["recall"] = stats["rec"]
        m["recall_0"] = stats["rec_0"]
        m["recall_1"] = stats["rec_1"]
        m["bal_acc"] = stats["bal_acc"]
        m["f1"] = stats["f1"]
        m["pos_rate"] = stats["pos_rate"]

        if logger is not None:
            try:
                logger("val/threshold", float(thr))
            except Exception:
                pass


def maybe_update_best(engine: Engine, *, watch_metric: str, mode: str, state: dict) -> bool:
    """
    Update state['best'] with a finite float from engine.state.metrics[watch_metric].
    Returns True if updated.
    """
    cur = to_float_scalar((engine.state.metrics or {}).get(watch_metric), strict=False)
    if not is_finite_float(cur):
        return False
    sign = 1.0 if str(mode).lower() == "max" else -1.0
    best = state.get("best", -float("inf") if sign > 0 else float("inf"))
    improved = (cur > best) if sign > 0 else (cur < best)
    if improved:
        state["best"] = float(cur)
    return improved


__all__ = [
    "Task",
    "PrepareBatch",
    "make_prepare_batch",
    "build_trainer",
    "build_evaluator",
    "attach_engine_metrics",
    "attach_scheduler",
    "attach_lr_scheduling",
    "attach_early_stopping",
    "attach_best_checkpoint",
    "make_warmup_safe_score_fn",
    "get_label_indices_from_batch",
    "maybe_update_best",
]
