# engine_utils.py
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union

from monai.engines import SupervisedTrainer
from ignite.engine import Events, Engine

from protocols import CalibratorProtocol
from evaluator_two_pass import make_two_pass_evaluator, TwoPassEvaluator
from metrics_utils import (
    to_tensor,
    extract_cls_logits_from_any,
    extract_seg_logits_from_any,
    to_float_scalar,
    coerce_loss_dict,
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
        if k in d:
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
        return y.long()

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


def attach_wandb_loggers(
    trainer: Engine,
    *,
    evaluator: Optional[Engine] = None,
    train_prefix: str = "train",
    val_prefix: str = "val",
    log_train: bool = True,
    log_val: bool = True,
    debug: bool = False,
    make_logger=None,  # factory for the handler; if None we'll try to import it
):
    """
    Attach W&B loggers with correct step semantics.
    - TRAIN: iteration-level logs from engine.state.output, step = global_step
    - VAL:   epoch-level logs from engine.state.metrics, step = epoch

    Idempotent: re-calling will remove previous handlers and re-attach.
    Returns a (train_cb, val_cb) tuple (None if not attached).
    """
    # lazy import if a factory wasn't provided
    if make_logger is None:
        try:
            from wandb_utils import make_wandb_logger as make_logger
        except Exception as e:
            raise ImportError(
                "attach_wandb_loggers needs a make_wandb_logger(factory). "
                "Pass it via make_logger=... or ensure wandb_utils.make_wandb_logger is importable."
            ) from e

    # Let W&B know which field is the step for each namespace (best-effort)
    try:
        import wandb
        wandb.define_metric(f"{train_prefix}/*", step_metric="global_step")
        wandb.define_metric(f"{val_prefix}/*", step_metric="epoch")
    except Exception:
        pass

    # detach old handlers (idempotent)
    old_tr = getattr(trainer.state, "_wandb_train_cb", None)
    if old_tr is not None:
        try:
            trainer.remove_event_handler(old_tr, Events.ITERATION_COMPLETED)
        except Exception:
            pass
        trainer.state._wandb_train_cb = None

    if evaluator is not None:
        old_val = getattr(evaluator.state, "_wandb_val_cb", None)
        if old_val is not None:
            try:
                evaluator.remove_event_handler(old_val, Events.COMPLETED)
            except Exception:
                pass
            evaluator.state._wandb_val_cb = None

    # attach new handlers
    train_cb = None
    val_cb = None

    if log_train:
        train_cb = make_logger(
            prefix=train_prefix,
            source="output",
            step_by="global",
            trainer=trainer,
            evaluator=evaluator,
            debug=debug,
        )
        trainer.add_event_handler(Events.ITERATION_COMPLETED, train_cb)
        trainer.state._wandb_train_cb = train_cb

    if log_val and evaluator is not None:
        val_cb = make_logger(
            prefix=val_prefix,
            source="metrics",
            step_by="epoch",
            trainer=trainer,
            evaluator=evaluator,
            debug=debug,
        )
        evaluator.add_event_handler(Events.COMPLETED, val_cb)
        evaluator.state._wandb_val_cb = val_cb

    return train_cb, val_cb


def build_trainer(
    device,
    max_epochs,
    train_data_loader,
    network,
    optimizer,
    loss_function,
    prepare_batch,
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

        # # optional: log LR
        # try:
        #     out["lr"] = float(optimizer.param_groups[0]["lr"])
        # except Exception:
        #     pass

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
    return trainer


def attach_two_pass_validation(
    *,
    trainer: Engine,
    model,
    val_loader,
    device: torch.device,
    cfg: Any,
    calibrator: Optional[CalibratorProtocol] = None,
    log_to_wandb: bool = True,
    wandb_prefix: str = "val/",
) -> TwoPassEvaluator:
    """
    Attach a TwoPassEvaluator at EPOCH_COMPLETED.
    Evaluator does not log; trainer logs exactly once per epoch with step=epoch.
    """
    ev = make_two_pass_evaluator(
        calibrator=calibrator,
        task=str(getattr(cfg, "task", cfg.get("task", "multitask"))),
        positive_index=int(getattr(cfg, "positive_index", cfg.get("positive_index", 1))),
    )

    # Remove any previous hook to avoid duplicate logging / step desync
    old = getattr(trainer.state, "_val_hook_2pass", None)
    if old is not None:
        try:
            trainer.remove_event_handler(old, Events.EPOCH_COMPLETED)
        except Exception:
            pass
        trainer.state._val_hook_2pass = None

    base_rate = getattr(cfg, "base_rate", cfg.get("base_rate", None))

    # Optional: declare metric step domains once
    wb = None
    if log_to_wandb:
        try:
            import wandb as _wandb
            wb = _wandb
            if not getattr(trainer.state, "_wb_val_defined", False):
                wb.define_metric("trainer/epoch")
                wb.define_metric(f"{wandb_prefix}*", step_metric="trainer/epoch")
                trainer.state._wb_val_defined = True
        except Exception:
            wb = None  # keep training even if wandb not available

    def _to_scalar(v):
        # Safe conversion for numpy/torch scalars
        try:
            import torch
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return float(v.detach().cpu().item())
                return v.detach().cpu().tolist()
        except Exception:
            pass
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    def _build_payload(epoch: int, cls_m: Dict[str, Any], seg_m: Dict[str, Any], t: float) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"trainer/epoch": int(epoch)}
        payload[f"{wandb_prefix}cal_thr"] = float(t)
        for k, v in cls_m.items():
            payload[f"{wandb_prefix}{k}"] = _to_scalar(v)
        for k, v in seg_m.items():
            key = f"{wandb_prefix}{k}"
            if key not in payload:
                payload[key] = _to_scalar(v)
        return payload

    @trainer.on(Events.EPOCH_COMPLETED)
    def _run_two_pass(engine: Engine):
        epoch = int(engine.state.epoch or 0)

        # Ensure model has a device attribute for evaluator (harmless if already set)
        if not hasattr(model, "device"):
            try:
                model.device = device
            except Exception:
                pass

        # --- Evaluate (no logging here) ---
        t, cls_metrics, seg_metrics = ev.validate(
            epoch=epoch, model=model, val_loader=val_loader, base_rate=base_rate
        )

        # Update engine metrics (so schedulers/handlers can read them)
        engine.state.metrics = engine.state.metrics or {}
        m = engine.state.metrics
        m.update({k: _to_scalar(v) for k, v in cls_metrics.items()})
        m.update({k: _to_scalar(v) for k, v in seg_metrics.items()})
        m["cal_thr"] = float(t)

        # --- Single-source logging: epoch step only ---
        if log_to_wandb and wb is not None:
            payload = _build_payload(epoch, cls_metrics, seg_metrics, t)
            wb.log(payload, step=epoch)

    trainer.state._val_hook_2pass = _run_two_pass
    return ev


# def attach_two_pass_validation(
#     *,
#     trainer: Engine,
#     model,
#     val_loader,
#     device: torch.device,
#     cfg: Any,
#     calibrator: Optional[CalibratorProtocol] = None,
#     # base_rate=None
# ) -> TwoPassEvaluator:
#     """
#     Attach a TwoPassEvaluator run at EPOCH_COMPLETED.
#     Returns the evaluator instance (keeps stateful threshold across epochs).
#     """
#     ev = make_two_pass_evaluator(
#         calibrator=calibrator,
#         task=str(getattr(cfg, "task", cfg.get("task", "multitask"))),
#         positive_index=int(getattr(cfg, "positive_index", cfg.get("positive_index", 1))),
#     )

#     old = getattr(trainer.state, "_val_hook_2pass", None)
#     if old is not None:
#         try:
#             trainer.remove_event_handler(old, Events.EPOCH_COMPLETED)
#         except Exception:
#             pass
#         trainer.state._val_hook_2pass = None

#     base_rate = getattr(cfg, "base_rate", cfg.get("base_rate", None))

#     @trainer.on(Events.EPOCH_COMPLETED)
#     def _run_two_pass(_):
#         epoch = int(trainer.state.epoch or 0)
#         if not hasattr(model, "device"):
#             try:
#                 model.device = device
#             except Exception:
#                 pass

#         t, cls_metrics, seg_metrics = ev.validate(
#             epoch=epoch,
#             model=model,
#             val_loader=val_loader,
#             base_rate=base_rate,
#         )

#         trainer.state.metrics = trainer.state.metrics or {}
#         m = trainer.state.metrics
#         m.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in cls_metrics.items()})
#         m.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in seg_metrics.items()})
#         m["cal_thr"] = float(t)

#     trainer.state._val_hook_2pass = _run_two_pass
#     return ev


# evaluator builder (uses shared extract_* helpers; removed _extract_outputs)
def build_evaluator(
    *,
    device,
    data_loader=None,
    network,
    prepare_batch,
    metrics: dict | None = None,
    postprocessing=None,
    inferer=None,
    non_blocking: bool | None = None,
    include_seg: bool = True,
    **kwargs,
) -> Engine:

    if data_loader is None:
        data_loader = kwargs.pop("val_data_loader", None)
    kwargs.pop("decollate", None)
    if data_loader is None:
        raise ValueError("build_evaluator: data_loader (or val_data_loader) must be provided")

    nb_default = bool(non_blocking) if non_blocking is not None else False

    def _eval_step(engine, batch):
        dev = getattr(engine.state, "device", device)
        nb = getattr(engine.state, "non_blocking", nb_default)
        x, targets = prepare_batch(batch, dev, nb)

        # targets can be label (Tensor[B]), mask (Tensor[B,H,W]) or dict({"label":..., "mask":...})
        y = m = None
        if isinstance(targets, dict):
            y = targets.get("label", None)
            m = targets.get("mask", None)
        else:
            y = targets

        with torch.no_grad():
            net_out = network(x) if inferer is None else inferer(x, network)

        # Use shared robust extractors
        logits = None
        try:
            logits = extract_cls_logits_from_any(net_out)
        except Exception:
            pass

        seg_logits = None
        if include_seg:
            try:
                seg_logits = extract_seg_logits_from_any(net_out)
            except Exception:
                pass

        out_map: Dict[str, Any] = {}

        # classification head
        if logits is not None:
            logits = to_tensor(logits).float()
            out_map["y_pred"] = logits
            out_map["logits"] = logits
            out_map["cls_out"] = logits
            if y is not None:
                out_map["y"] = to_tensor(y).long().view(-1)

        # segmentation head (only if allowed and mask exists)
        if include_seg and (seg_logits is not None):
            out_map["seg_out"] = to_tensor(seg_logits)
            out_map["seg_logits"] = out_map["seg_out"]

        # pack label dict only with keys we actually have
        label_dict: Dict[str, Any] = {}
        if y is not None:
            label_dict["label"] = y
        if include_seg and (m is not None):
            label_dict["mask"] = m
        if label_dict:
            out_map["label"] = label_dict

        return out_map

    evaluator = Engine(_eval_step)
    pin = getattr(data_loader, "pin_memory", False)
    _attach_device_state(evaluator, device=device, non_blocking=pin)

    if metrics:
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

    evaluator.state.data_loader = data_loader
    return evaluator


# prepare_batch & scheduler
PrepareBatch = Callable[
    [Union[Dict[str, Any], Sequence[Any]], Optional[torch.device], bool],
    Tuple[torch.Tensor, Any]
]


class Task(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    MULTITASK = "multitask"


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
            targets = y_cls
        elif t is Task.SEGMENTATION:
            if y_seg is None:
                raise ValueError("prepare_batch(segmentation): missing 'mask'.")
            targets = y_seg
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


# helper to wire schedulers
def attach_lr_scheduling(trainer: Engine, evaluator: Engine | None, optimizer, scheduler, *,
                         plateau_metric: str | None = None, plateau_mode: str = "max") -> None:
    if scheduler is None:
        return

    from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

    # Make scheduler discoverable for loggers (used by read_current_lr)
    try:
        setattr(trainer.state, "_scheduler", scheduler)
    except Exception:
        pass

    if isinstance(scheduler, OneCycleLR):
        @trainer.on(Events.ITERATION_COMPLETED)
        def _step_onecycle(_):
            scheduler.step()
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        if not plateau_metric:
            raise ValueError("ReduceLROnPlateau needs plateau_metric='val_loss' or 'val_auc', etc.")

        @trainer.on(Events.EPOCH_COMPLETED)
        def _step_plateau(_):
            # prefer evaluator metrics if provided, otherwise trainer metrics
            source = evaluator if evaluator is not None else trainer
            metrics = getattr(source.state, "metrics", {}) or {}
            # key = plateau_metric or ("val_auc" if str(plateau_mode).lower() == "max" else "val_loss")
            # candidates = {key, key.replace("/", "_"), key.replace("_", "/")}
            # val = next((metrics.get(k) for k in candidates if k in metrics), None)
            key = plateau_metric or ("val_auc" if str(plateau_mode).lower() == "max" else "val_loss")
            # try common variants: "val/auc" -> "val_auc" -> "auc", and vice-versa
            candidates = {key, key.replace("/", "_"), key.replace("_", "/")}
            base = key
            # strip common "val" prefixes
            base = base.replace("val/", "").replace("val_", "")
            # also take the last token after a separator just in case
            base = base.split("/")[-1].split("_")[-1]
            candidates.update({base, f"val_{base}", f"val/{base}"})
            val = next((metrics.get(k) for k in candidates if k in metrics), None)
            # val = metrics.get(plateau_metric, None)
            if val is not None:
                try:
                    scheduler.step(float(val))
                except Exception:
                    pass
            return
            # no-op if metric missing; do NOT call scheduler.step() without a metric
        # IMPORTANT: do not add the generic epoch step for ReduceLROnPlateau
        return

    # Generic epoch-based schedulers (Cosine, StepLR, MultiStep, Exponential, etc.)
    @trainer.on(Events.EPOCH_COMPLETED)
    def _step_epoch(_):
        scheduler.step()


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

        # write confusion-matrix-like fields compatible with your logs
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


__all__ = [
    "Task",
    "PrepareBatch",
    "make_prepare_batch",
    "build_trainer",
    "build_evaluator",
    "attach_scheduler",
    "attach_wandb_loggers",
    "attach_lr_scheduling",
]
