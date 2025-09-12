# engine_utils.py
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
import numpy as np
import math
import torch
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union, Mapping
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver
from ignite.engine import Events, Engine
from utils.safe import to_float_scalar, is_finite_float, to_tensor
from posprob import PosProbCfg
from optim_factory import grad_norms_by_group  # zero_grad_if_nan
from metrics_utils import (
    extract_cls_logits_from_any,
    extract_seg_logits_from_any,
    make_metrics,
    make_cls_val_metrics,
    make_seg_val_metrics,
    coerce_loss_dict
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


def _pick_logits_from_output(out: Any) -> Optional[torch.Tensor]:
    # Returns raw logits only; never probabilities. Decisions must use decision_from_logits(...).
    if isinstance(out, (tuple, list)) and len(out) >= 1:
        return out[0]
    if isinstance(out, dict):
        for k in ("y_pred", "logits", "cls_out"):
            v = out.get(k, None)
            if v is not None:
                return v
    return None


# def ppcfg_from_logits(logits, positive_index=1):
#     if logits.ndim == 1 or (logits.ndim >= 2 and logits.shape[-1] == 1):
#         scheme = "bce1"
#     elif logits.ndim >= 2 and logits.shape[-1] >= 2:
#         scheme = "ce"
#     else:
#         scheme = "bce1"
#     return PosProbCfg(positive_index=int(positive_index), scheme=scheme)


def decision_from_logits(
    logits: torch.Tensor,
    threshold: float,
    *,
    positive_index: int = 1,
    to_pos_prob: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Return int64 predictions computed as (p_pos >= threshold), never on raw logits.
    Prefer passing to_pos_prob = evaluator.state.to_pos_prob for single SoT.
    """
    if to_pos_prob is None:
        # Fallback: construct a minimal PosProbCfg (kept for backward compat)
        # pp = PosProbCfg(
        #     binary_single_logit=(logits.ndim in (1, 2) and (logits.ndim == 1 or logits.shape[-1] == 1)),
        #     positive_index=int(positive_index))
        # ppcfg = ppcfg_from_logits(logits, positive_index)
        if logits.ndim == 1 or (logits.ndim >= 2 and logits.shape[-1] == 1):
            scheme = "bce1"          # single-logit BCEWithLogits
        elif logits.ndim >= 2 and logits.shape[-1] >= 2:
            scheme = "ce"             # multi-logit CrossEntropy-style
        else:
            scheme = "bce1"           # safe fallback
        pp = PosProbCfg(positive_index=int(positive_index), scheme=scheme)
        probs = pp.logits_to_pos_prob(logits)
    else:
        probs = to_pos_prob(logits)
    return (probs.view(-1) >= float(threshold)).long()


def build_trainer(
    device,
    max_epochs,
    train_data_loader,
    network,
    optimizer,
    loss_function,
    prepare_batch,
    train_metrics: Optional[Dict[str, Any]] = None,
) -> Engine:
    assert callable(loss_function), "loss_function must be callable"
    assert callable(prepare_batch), "prepare_batch must be callable"

    # Defaults; caller may override after construction via trainer.state.*
    _grad_dbg_steps_default = 100
    _grad_dbg_every_default = 10
    _logit_dbg_steps_default = 20
    _skip_warn_every_default = 20

    def _iteration_update(engine: Engine, batch):
        network.train()

        dev = getattr(engine.state, "device", device)
        nb = getattr(engine.state, "non_blocking", bool(getattr(train_data_loader, "pin_memory", False)))

        x, targets = prepare_batch(batch, dev, nb)

        optimizer.zero_grad(set_to_none=True)
        preds = network(x)

        # early logits-spread probe to catch constant scores
        try:
            it = int(getattr(engine.state, "iteration", 0) or 0)
            dbg_logit_until = int(getattr(engine.state, "logit_debug_steps", _logit_dbg_steps_default))
            if it <= dbg_logit_until:
                from metrics_utils import extract_cls_logits_from_any
                logits = extract_cls_logits_from_any(preds)
                if torch.is_tensor(logits):
                    logits = logits.detach().float().view(-1)
                    if logits.numel() > 0:
                        lmin, lmed, lmax = logits.min().item(), logits.median().item(), logits.max().item()
                        lstd = logits.std(unbiased=False).item()
                        # stash to output for logging
                        engine.state._last_logit_probe = (lmin, lmed, lmax, lstd)
        except Exception:
            pass

        # Normalize any loss return to a dict with a 'loss' tensor
        loss_out = loss_function(preds, targets)
        loss_map = coerce_loss_dict(loss_out)
        total = loss_map["loss"]
        if not torch.is_tensor(total):
            raise TypeError("'loss' must be a torch.Tensor.")

        total.backward()

        # Gradient diagnostics (first N steps, every K iters)
        it = int(getattr(engine.state, "iteration", 0) or 0)
        dbg_until = int(getattr(engine.state, "grad_debug_steps", _grad_dbg_steps_default))
        dbg_every = int(getattr(engine.state, "grad_debug_every", _grad_dbg_every_default))

        out: Dict[str, float] = {}

        if it <= dbg_until and it % max(1, dbg_every) == 0:
            try:
                norms = grad_norms_by_group(optimizer)
                out["grad/gnorm_total"] = float(norms.get("total", 0.0))
                gi = 0
                for k, v in norms.items():
                    if k.startswith("g"):
                        out[f"grad/gnorm_g{gi}"] = float(v)
                        gi += 1
            except Exception:
                pass

        # Safe NaN/Inf guard: ignore None grads; zero only non-finite grads
        bad = False
        try:
            for group in optimizer.param_groups:
                for p in group.get("params", ()):
                    g = p.grad
                    if g is None:
                        continue  # do NOT treat missing grads as bad
                    if not torch.isfinite(g).all():
                        g.detach().zero_()
                        bad = True
        except Exception:
            # If anything goes wrong, be conservative: attempt a step
            bad = False

        did_step = not bad
        if did_step:
            optimizer.step()
            engine.state._skipped_steps = 0
        else:
            # count consecutive skips and warn occasionally
            skips = int(getattr(engine.state, "_skipped_steps", 0) or 0) + 1
            engine.state._skipped_steps = skips
            warn_every = int(getattr(engine.state, "skip_warn_every", _skip_warn_every_default))
            if skips % max(1, warn_every) == 0:
                logger.warning("[trainer] skipped %d optimizer.step() due to non-finite grads.", skips)

        optimizer.zero_grad(set_to_none=True)

        # Scalars for loggers/StatsHandler/etc.
        from utils.safe import to_float_scalar  # already imported at top file
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

        # Expose current LR (scheduler-aware)
        sched = getattr(engine.state, "_scheduler", None)
        lr_val = read_current_lr(optimizer, sched)
        if lr_val is not None:
            out["lr"] = float(lr_val)

        # Expose whether we actually stepped + (optional) logits probe
        out["opt/did_step"] = 1.0 if did_step else 0.0
        probe = getattr(engine.state, "_last_logit_probe", None)
        if probe is not None:
            lmin, lmed, lmax, lstd = probe
            out["logits/min"] = float(lmin)
            out["logits/med"] = float(lmed)
            out["logits/max"] = float(lmax)
            out["logits/std"] = float(lstd)

        return out

    trainer = Engine(_iteration_update)

    # mirror SupervisedTrainer’s state
    trainer.state.max_epochs = int(max_epochs)
    trainer.state.epoch = 0

    # stash device + pin_memory hint on state
    _attach_device_state(trainer, device, getattr(train_data_loader, "pin_memory", False))

    # unified scorer on trainer.state (parity with evaluator)
    if not hasattr(trainer.state, "ppcfg"):
        trainer.state.ppcfg = PosProbCfg.from_cfg(getattr(trainer.state, "cfg", {}))
        trainer.state.to_pos_prob = trainer.state.ppcfg.logits_to_pos_prob

    # attach training metrics (bare names)
    if train_metrics and not getattr(trainer.state, "_train_metrics_attached", False):
        for name, metric in train_metrics.items():
            metric.attach(trainer, name)
        setattr(trainer.state, "_train_metrics_attached", True)

    # tweakables (can be overridden by caller after build)
    # trainer.state.grad_debug_steps = 200
    # trainer.state.grad_debug_every = 5
    # trainer.state.logit_debug_steps = 20
    # trainer.state.skip_warn_every = 20

    return trainer


def build_evaluator(
    *,
    device: torch.device,
    data_loader,  # required (val loader)
    network: torch.nn.Module,
    prepare_batch: Callable,  # (batch, device, non_blocking) -> (x, targets)
    metrics: Dict[str, Any] | None = None,
    include_seg: bool = True,
    inferer: Optional[Callable] = None,
    non_blocking: Optional[bool] = None,
    trainer_for_logging: Optional[Engine] = None,
    ppcfg: Optional[PosProbCfg] = None,
) -> Engine:
    """Custom evaluator that returns an out_map dict compatible with metrics."""
    nb_default = bool(non_blocking) if non_blocking is not None else False

    # single source-of-truth scorer
    if ppcfg is None:
        # ppcfg = PosProbCfg.from_cfg(cfg)
        ppcfg = PosProbCfg(positive_index=1, scheme="bce1")
        # ppcfg = PosProbCfg()  # defaults: positive_index=1, binary_single_logit=True, mode="auto"

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
            out_map["y_pred"] = logits  # Keep logits here. Ignite metrics' output transforms will convert to probs/decisions.
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
    # bind unified scorer to engine.state
    setattr(evaluator.state, "ppcfg", ppcfg)
    setattr(evaluator.state, "to_pos_prob", ppcfg.logits_to_pos_prob)

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
    seg_num_classes: Optional[int] = None,
    seg_threshold: Optional[float] = None,
    seg_ignore_index: Optional[int] = None,
    ppcfg: Optional[PosProbCfg] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build and attach metrics to trainer and evaluator with bare names.
    Returns (train_metrics, val_metrics).
    """
    task_set = set(tasks)

    # single PosProbCfg for both trainer/evaluator metrics
    if ppcfg is None:
        # ppcfg = PosProbCfg.from_cfg(cfg)
        # ppcfg.positive_index = int(positive_index)
        ppcfg = PosProbCfg(positive_index=int(positive_index), scheme="bce1")

    # training metrics (keep light; classification-only by default)
    train_metrics = make_metrics(
        tasks=task_set,
        num_classes=num_classes,
        cls_decision=cls_decision,
        cls_threshold=cls_threshold,
        positive_index=positive_index,
        ppcfg=ppcfg,
    )
    if not getattr(trainer.state, "_train_metrics_attached", False):
        for k, m in train_metrics.items():
            m.attach(trainer, k)
        setattr(trainer.state, "_train_metrics_attached", True)

    # Seed the evaluator with an initial threshold (single SoT)
    if getattr(evaluator.state, "threshold", None) is None:
        evaluator.state.threshold = float(cls_threshold)

    dev = getattr(trainer.state, "device", None) or getattr(evaluator.state, "device", None) or torch.device("cpu")

    # validation metrics: start with classification
    val_metrics = make_cls_val_metrics(
        num_classes=num_classes,
        decision=cls_decision,
        # threshold=cls_threshold,
        threshold=lambda: float(getattr(evaluator.state, "threshold", cls_threshold)),
        positive_index=positive_index,
        ppcfg=ppcfg,
        device=dev,
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
        mode = str(cfg.get("plateau_mode", "min")).lower()
        patience = int(cfg.get("patience", 5))
        factor = float(cfg.get("factor", 0.5))
        threshold = float(cfg.get("plateau_threshold", 1e-4))

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor,
            threshold=threshold,
        )
        setattr(sched, "_plateau_metric", str(cfg.get("plateau_metric", "val/loss")))
        logger.info(
            "ReduceLROnPlateau created: mode=%s, patience=%d, factor=%.3g, threshold=%.3g, metric=%s",
            mode, patience, factor, threshold, getattr(sched, "_plateau_metric")
        )
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
    plateau_mode: str | None = None,
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


def _best_threshold_from_probs(p: torch.Tensor, y_true: torch.Tensor, mode: str = "bal_acc") -> tuple[float, dict]:
    """
    p: (N,) positive-class probabilities in [0,1]
    y_true: (N,) {0,1}
    Returns (threshold, stats_dict)
    """
    with torch.no_grad():
        p = p.detach().float().clamp_(0.0, 1.0).view(-1)     # keep device, in-place clamp
        y = y_true.detach().long().view(-1).to(p.device)
        # candidate thresholds: unique preds + endpoints
        # stay on the same device; quantile works on CUDA too
        cand = torch.quantile(p, torch.linspace(0, 1, 101, device=p.device)).unique()

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


# threshold calibration helpers (logits-aware)
@dataclass
class CalCfg:
    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_posrate_min: float = 0.05
    thr_posrate_max: float = 0.95
    thr_min_tp: int = 0
    thr_min_tn: int = 0
    cal_auc_floor: float = 0.50
    cal_warmup_epochs: int = 1
    cal_max_delta: float = 0.10
    # low-spread guard
    cal_min_std: float = 1e-4
    cal_min_iqr: float = 1e-3
    # smoothing & stability
    ema_beta: float = 0.3            # 0..1, fraction of movement applied this epoch
    thr_hysteresis: float = 0.02     # inner pos-rate corridor margin


def _safe_auc_np(labels: np.ndarray, scores: np.ndarray) -> float:
    y = labels.astype(np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, scores))
    except Exception:
        # rank-based fallback (ties handled)
        order = np.argsort(scores)
        s_sorted = scores[order]
        ranks = np.empty_like(s_sorted, dtype=np.float64)
        i = 0
        while i < len(s_sorted):
            j = i
            while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
                j += 1
            ranks[i:j + 1] = (i + j + 2) / 2.0
            i = j + 1
        r = np.empty_like(scores, dtype=np.float64)
        r[order] = ranks
        sum_pos = float(r[y == 1].sum())
        return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _bounded_step(old: float, new: float, *, max_delta: float, lo: float, hi: float) -> float:
    step = np.clip(new - old, -float(max_delta), float(max_delta))
    return float(np.clip(old + step, float(lo), float(hi)))


def _rate_match_threshold(probs: np.ndarray, target_rate: float) -> float:
    q = float(np.clip(1.0 - target_rate, 0.0, 1.0))
    return float(np.quantile(probs, q))


def _spread(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    std = float(np.std(x))
    q10 = float(np.quantile(x, 0.10))
    q90 = float(np.quantile(x, 0.90))
    return std, (q90 - q10)


def pick_threshold_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    prev_thr: float,
    epoch: int,
    cfg: CalCfg,
    to_pos_prob: callable,           # e.g., evaluator.state.to_pos_prob
    base_rate: float | None = None,  # optional
    mode: str = "bal_acc",           # or "f1"
) -> tuple[float, dict]:
    """
    Returns (new_threshold, stats dict). Computes on logits -> probs inside.
    Applies spread guards, pos-rate envelope, AUC floor, min TP/TN, and bounded movement.
    """
    # logits -> probs, labels -> np
    with torch.no_grad():
        p = to_pos_prob(logits).detach().reshape(-1).clamp(0, 1).cpu().numpy()
        y = labels.detach().reshape(-1).long().cpu().numpy()

    n = p.size
    stats_out: dict = {"n": int(n)}
    if n == 0:
        return float(prev_thr), stats_out

    # spread guard
    std, iqr = _spread(p)
    stats_out["spread_std"] = std
    stats_out["spread_iqr"] = iqr

    # degenerate/flat probs → soft pos-rate clamp into envelope using current distribution
    if std < cfg.cal_min_std and iqr < cfg.cal_min_iqr:
        pos_rate_prev = float((p >= prev_thr).mean())
        # if outside envelope, set quantile threshold to project inside bounds
        if pos_rate_prev < cfg.thr_posrate_min or pos_rate_prev > cfg.thr_posrate_max:
            target = cfg.thr_posrate_min if pos_rate_prev < cfg.thr_posrate_min else cfg.thr_posrate_max
            thr_star = _rate_match_threshold(p, target)
            thr_star = _bounded_step(prev_thr, thr_star, max_delta=cfg.cal_max_delta, lo=cfg.thr_min, hi=cfg.thr_max)
            stats_out.update({"reason": "low_spread_clamp", "pos_rate_prev": pos_rate_prev, "thr_star": float(thr_star)})
            return float(thr_star), stats_out
        # already in bounds: keep previous
        stats_out.update({"reason": "low_spread_keep", "pos_rate_prev": pos_rate_prev})
        return float(prev_thr), stats_out

    # candidate search on probs: quantile grid (101 pts)
    qs = np.unique(np.quantile(p, np.linspace(0, 1, 101)))
    best_thr, best_score = prev_thr, -1.0
    best = {}

    for t in qs:
        y_hat = (p >= t).astype(np.int64)
        tp = int(((y_hat == 1) & (y == 1)).sum())
        tn = int(((y_hat == 0) & (y == 0)).sum())
        fp = int(((y_hat == 1) & (y == 0)).sum())
        fn = int(((y_hat == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec1 = tp / max(tp + fn, 1)
        rec0 = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (rec0 + rec1)
        f1 = 2 * prec * rec1 / max(prec + rec1, 1e-8)
        score = bal_acc if mode == "bal_acc" else f1
        if score > best_score:
            best_score = score
            best_thr = float(t)
            best = dict(tp=tp, tn=tn, fp=fp, fn=fn, prec=prec, rec=rec1, rec_0=rec0, rec_1=rec1,
                        bal_acc=bal_acc, f1=f1, pos_rate=float((y_hat == 1).mean()))

    # guardrails
    # 1) warmup
    if epoch < int(cfg.cal_warmup_epochs):
        stats_out.update({"reason": "warmup", "cand": best_thr})
        return float(prev_thr), stats_out

    # 2) envelope clamp
    pr = float((p >= best_thr).mean())
    if pr < cfg.thr_posrate_min or pr > cfg.thr_posrate_max:
        stats_out.update({"reason": "posrate_envelope_reject", "cand": best_thr, "pos_rate": pr})
        return float(prev_thr), stats_out

    # 3) AUC floor
    auc_val = _safe_auc_np(y, p)
    stats_out["auc"] = auc_val
    if not (isinstance(auc_val, float) and np.isfinite(auc_val) and auc_val >= cfg.cal_auc_floor):
        stats_out.update({"reason": "auc_floor_reject", "cand": best_thr})
        return float(prev_thr), stats_out

    # 4) min counts
    if cfg.thr_min_tp > 0 or cfg.thr_min_tn > 0:
        y_hat = (p >= best_thr).astype(np.int64)
        tp = int(((y == 1) & (y_hat == 1)).sum())
        tn = int(((y == 0) & (y_hat == 0)).sum())
        if tp < cfg.thr_min_tp or tn < cfg.thr_min_tn:
            stats_out.update({"reason": "min_counts_reject", "cand": best_thr, "tp": tp, "tn": tn})
            return float(prev_thr), stats_out

    # 5) bounded movement and global bounds
    thr_clamped = float(np.clip(best_thr, cfg.thr_min, cfg.thr_max))
    new_thr = _bounded_step(prev_thr, thr_clamped, max_delta=cfg.cal_max_delta, lo=cfg.thr_min, hi=cfg.thr_max)

    # 6) optional rate match to base_rate (then bounded)
    if base_rate is not None:
        pr_now = float((p >= new_thr).mean())
        # if too far from base, nudge toward rate-matched
        rate_tol = getattr(cfg, "cal_rate_tolerance", 0.15)
        if abs(pr_now - float(base_rate)) > rate_tol:
            t_dm = _rate_match_threshold(p, float(base_rate))
            new_thr = _bounded_step(new_thr, t_dm, max_delta=cfg.cal_max_delta, lo=cfg.thr_min, hi=cfg.thr_max)
            stats_out["rate_match_applied"] = True

    # 7) hysteresis on decision rate (dampen flips when comfortably inside corridor)
    pmin, pmax = float(cfg.thr_posrate_min), float(cfg.thr_posrate_max)
    margin = float(getattr(cfg, "thr_hysteresis", 0.02))
    pos_rate_candidate = float((p >= new_thr).mean())
    if (pmin + margin) <= pos_rate_candidate <= (pmax - margin):
        # Within inner corridor → keep previous threshold (no change)
        stats_out["hysteresis_hold"] = True
        new_thr = prev_thr

    # 8) EMA smoothing (apply a fraction of the movement)
    beta = float(getattr(cfg, "ema_beta", 0.3))
    if beta > 0.0:
        new_thr = float(prev_thr + beta * (new_thr - prev_thr))

    # 9) final clamp
    new_thr = float(np.clip(new_thr, cfg.thr_min, cfg.thr_max))

    stats_out.update({"reason": "accepted", "cand": best_thr, "final": new_thr, **best})
    return float(new_thr), stats_out


def attach_val_threshold_search(
    *,
    evaluator: Engine,
    mode: str = "bal_acc",   # "bal_acc" or "f1"
    logger=None,
    positive_index: int = 1,
    score_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Collect positive-class probabilities each val epoch, pick a threshold on the same scale,
    and write confusion/summary stats into evaluator.state.metrics.
    Adds guards to avoid locking into degenerate thresholds when pos_rate is 0 or 1.
    """
    buf = {"scores": [], "targets": []}
    state = {"last_good_thr": 0.5}  # noqa F841 retains last non-degenerate threshold in (0,1)

    @evaluator.on(Events.EPOCH_STARTED)
    def _reset_buffers(_):
        buf["scores"].clear()
        buf["targets"].clear()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def _collect(engine):
        out = engine.state.output
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits, y_true = out[0], out[1]
        elif isinstance(out, dict) and "y_pred" in out and "y" in out:
            logits, y_true = out["y_pred"], out["y"]
        else:
            return
        if logits is None:
            return
        to_pos_prob = getattr(engine.state, "to_pos_prob", None)
        if to_pos_prob is None:
            return
        # prob = to_pos_prob(logits)
        # buf["scores"].append(prob.detach().reshape(-1))
        # buf["targets"].append(torch.as_tensor(y_true).detach().reshape(-1).long())
        p = to_pos_prob(logits).detach().reshape(-1).clamp_(0.0, 1.0).cpu()  # <-- .cpu()
        y = torch.as_tensor(y_true).detach().reshape(-1).long().cpu()        # <-- .cpu()
        buf["scores"].append(p)
        buf["targets"].append(y)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def _choose_threshold(engine: Engine):
        if not buf["scores"]:
            return

        # We already stored CPU probs & labels in _collect — switch to logits-first:
        # If we prefer to keep _collect as logits, change that collection and adapt here.
        probs = torch.cat(buf["scores"], dim=0).clamp(0.0, 1.0)
        y_true = torch.cat(buf["targets"], dim=0).long()

        # Reconstruct logits-like via logit(p) for the helper, so the same path is exercised.
        # (Or, better: collect logits in _collect and skip this inversion.)
        eps = 1e-6
        p_np = probs.numpy()
        logits_np = np.log(np.clip(p_np, eps, 1 - eps)) - np.log(np.clip(1 - p_np, eps, 1 - eps))
        logits = torch.from_numpy(logits_np)
        labels = y_true

        # prev = float(getattr(evaluator.state, "threshold", 0.5))
        prev = float(engine.state.threshold) if hasattr(engine.state, "threshold") else float(getattr(engine.state, "cls_threshold", 0.5))
        # Write it back once so future epochs never fall back to cfg
        if not hasattr(engine.state, "threshold"):
            engine.state.threshold = prev
        # Single source of truth scorer
        to_pos_prob = getattr(engine.state, "to_pos_prob", None)
        if to_pos_prob is None:
            return

        # Pull config knobs if attached on engine.state or via a namespaced object
        cfg_obj = SimpleNamespace(
            thr_min=getattr(engine.state, "thr_min", 0.05),
            thr_max=getattr(engine.state, "thr_max", 0.95),
            thr_posrate_min=getattr(engine.state, "thr_posrate_min", 0.05),
            thr_posrate_max=getattr(engine.state, "thr_posrate_max", 0.95),
            thr_min_tp=getattr(engine.state, "thr_min_tp", 0),
            thr_min_tn=getattr(engine.state, "thr_min_tn", 0),
            cal_auc_floor=getattr(engine.state, "cal_auc_floor", 0.50),
            cal_warmup_epochs=getattr(engine.state, "thr_warmup_epochs", 1),
            cal_max_delta=getattr(engine.state, "cal_max_delta", 0.10),
            cal_min_std=getattr(engine.state, "cal_min_std", 1e-4),
            cal_min_iqr=getattr(engine.state, "cal_min_iqr", 1e-3),
            cal_rate_tolerance=getattr(engine.state, "cal_rate_tolerance", 0.15),
            ema_beta=getattr(engine.state, "cal_ema_beta", 0.3),
            thr_hysteresis=getattr(engine.state, "thr_hysteresis", 0.02),
        )
        # calcfg = CalCfg(**cfg_obj.__dict__)
        import inspect
        # 1) Only pass fields CalCfg actually accepts
        _allowed = set(inspect.signature(CalCfg).parameters.keys())
        _d = dict(cfg_obj.__dict__)  # shallow copy
        # 2) Backward-compat: map legacy keys -> current field names
        #    Adjust this mapping to whatever CalCfg expects in codebase.
        legacy_map = {
            "cal_rate_tolerance": "rate_tolerance",   # or "rate_tol" if that's current name
            "rate_tol_override": "rate_tolerance",     # let explicit override win
            "cal_ema_beta": "ema_beta",
        }
        for src, dst in legacy_map.items():
            if src in _d and dst not in _d:
                _d[dst] = _d[src]
        # 3) Drop any keys CalCfg doesn't take
        _d = {k: v for k, v in _d.items() if k in _allowed}
        calcfg = CalCfg(**_d)

        # optional base rate (use ground-truth mean if not injected)
        base_rate = float((labels.numpy() == 1).mean())

        new_thr, s = pick_threshold_from_logits(
            logits=logits,
            labels=labels,
            prev_thr=prev,
            epoch=int(getattr(engine.state, "epoch", 0) or 0),
            cfg=calcfg,
            to_pos_prob=to_pos_prob,
            base_rate=base_rate,
            mode=str((score_kwargs or {}).get("mode", "bal_acc")),
        )

        # summaries
        probs_final = to_pos_prob(torch.from_numpy(logits_np)).numpy()  # for consistent metrics dump
        y_hat = (probs_final >= new_thr).astype(np.int64)
        tp = int(((labels.numpy() == 1) & (y_hat == 1)).sum())
        tn = int(((labels.numpy() == 0) & (y_hat == 0)).sum())
        fp = int(((labels.numpy() == 0) & (y_hat == 1)).sum())
        fn = int(((labels.numpy() == 1) & (y_hat == 0)).sum())
        pos_rate = float(y_hat.mean())

        m = engine.state.metrics
        m["score_mean"] = float(p_np.mean())
        m["score_std"] = float(p_np.std())
        m["gt_pos_rate"] = float((labels.numpy() == 1).mean())

        m["search_threshold"] = float(new_thr)
        m["search_confmat"] = [[tn, fp], [fn, tp]]
        m["search_confmat_00"] = float(tn)
        m["search_confmat_01"] = float(fp)
        m["search_confmat_10"] = float(fn)
        m["search_confmat_11"] = float(tp)
        m["search_pos_rate"] = float(pos_rate)
        if "auc" in s:
            m["search_auc"] = float(s["auc"])

        evaluator.state.threshold = float(new_thr)
        try:
            if hasattr(evaluator, "trainer"):
                evaluator.trainer.state.threshold = float(new_thr)
        except Exception:
            pass


def attach_pos_score_debug_once(
    evaluator: Engine,
    *,
    positive_index: int = 1,
    score_kwargs: dict | None = None,
    bins: int = 20,
    print_fn: Callable[[str], None] = print,
    tag: str = "debug_pos",
) -> None:
    """
    Collects positive-class scores for ONE validation run and:
      - prints: min, q10, median, q90, max, pos@0.5
      - writes to evaluator.state.metrics:
          {tag}_min, {tag}_q10, {tag}_median, {tag}_q90, {tag}_max,
          {tag}_rate@0.5, {tag}_hist (list)
    Then detaches itself so it never runs again.

    Works with evaluator outputs of shape:
      - dict with 'y_pred'/'logits'/'cls_out'
      - tuple/list like (logits, y)
    """
    # Idempotency guard (avoid double-attach)
    guard_key = f"_{tag}_once_attached"
    if getattr(evaluator.state, guard_key, False):
        return
    setattr(evaluator.state, guard_key, True)

    buf: Dict[str, list] = {"scores": []}

    @evaluator.on(Events.EPOCH_STARTED)
    def _reset(_):
        buf["scores"].clear()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def _collect(engine: Engine):
        out = engine.state.output
        logits = _pick_logits_from_output(out)
        if logits is None:
            return
        # Single SoT: evaluator.state.to_pos_prob
        to_pos_prob = getattr(engine.state, "to_pos_prob", None)
        if to_pos_prob is None:
            return
        try:
            s = to_pos_prob(logits).detach().reshape(-1).cpu()
            buf["scores"].append(s)
        except Exception:
            pass

    @evaluator.on(Events.COMPLETED)
    def _summarize(engine: Engine):
        if not buf["scores"]:
            return
        import torch

        scores = torch.cat(buf["scores"], dim=0).clamp(0.0, 1.0)
        n = int(scores.numel())
        mn = float(scores.min().item())
        mx = float(scores.max().item())
        med = float(scores.median().item())
        q10 = float(scores.quantile(0.10).item())
        q90 = float(scores.quantile(0.90).item())
        pos50 = float((scores >= 0.5).float().mean().item())
        hist = torch.histc(scores, bins=int(max(1, bins)), min=0.0, max=1.0).tolist()

        # Print once (easy to see in logs)
        try:
            print_fn(
                f"[pos-score] n={n} min={mn:.4f} q10={q10:.4f} med={med:.4f} "
                f"q90={q90:.4f} max={mx:.4f} pos@0.5={pos50:.4f}"
            )
        except Exception:
            pass

        # Expose in metrics (logger will prefix with 'val/' in validation context)
        m = engine.state.metrics
        m[f"{tag}_min"] = mn
        m[f"{tag}_q10"] = q10
        m[f"{tag}_median"] = med
        m[f"{tag}_q90"] = q90
        m[f"{tag}_max"] = mx
        m[f"{tag}_rate@0.5"] = pos50
        m[f"{tag}_hist"] = hist

        # Detach handlers so this runs only once
        try:
            evaluator.remove_event_handler(_reset, Events.EPOCH_STARTED)
            evaluator.remove_event_handler(_collect, Events.ITERATION_COMPLETED)
            evaluator.remove_event_handler(_summarize, Events.COMPLETED)
        except Exception:
            pass

    @evaluator.on(Events.ITERATION_COMPLETED)
    def _log_logits_stats(engine):
        out = engine.state.output
        logits = _pick_logits_from_output(out)
        if logits is None:
            return
        t = logits.detach().view(-1)
        # print once early; or write to metrics as needed
        if getattr(engine.state, "_logged_logits_once", False):
            return
        print(f"[debug] logits mean={t.mean().item():.4f} std={t.std().item():.4f} min={t.min().item():.4f} max={t.max().item():.4f}")
        engine.state._logged_logits_once = True


def attach_val_stack(
    *,
    evaluator: Engine,
    num_classes: int,
    ppcfg: PosProbCfg,
    # classification controls
    has_cls: bool = True,
    cls_decision: str = "threshold",
    threshold_getter: Callable[[], float] | float = 0.5,
    positive_index: int = 1,
    # segmentation controls
    has_seg: bool = False,
    seg_num_classes: Optional[int] = None,
    seg_threshold: Optional[float] = None,
    seg_ignore_index: Optional[int] = None,
    # calibration / debug
    enable_threshold_search: bool = True,
    calibration_method: str = "bal_acc",  # or "f1"
    enable_pos_score_debug_once: bool = False,
    pos_debug_bins: int = 20,
    pos_debug_tag: str = "debug_pos",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Compose the 'val stack' on a single evaluator:
      - attach classification metrics via unified PosProbCfg
      - optionally attach segmentation metrics
      - optionally attach one-shot pos-score debug
      - optionally attach per-epoch threshold search
    Returns a dict of metrics that were attached (for reference).
    """
    attached: Dict[str, Any] = {}
    # Resolve a concrete device for Ignite metrics
    dev = device or getattr(evaluator.state, "device", None) or torch.device("cpu")

    # classification metrics
    if has_cls:
        # normalize threshold_getter to a callable
        if not callable(threshold_getter):
            _thr_val = float(threshold_getter)
            threshold_getter = lambda: _thr_val  # noqa: E731

        cls_metrics = make_cls_val_metrics(
            num_classes=int(num_classes),
            ppcfg=ppcfg,  # unified scorer
            positive_index=int(positive_index),
            cls_decision=str(cls_decision),
            cls_threshold=threshold_getter,
            device=dev,
        )
        for k, m in cls_metrics.items():
            m.attach(evaluator, k)
        attached.update(cls_metrics)

    # segmentation metrics
    if has_seg:
        try:
            seg_nc = int(seg_num_classes) if seg_num_classes is not None else int(num_classes)
            seg_thr = float(seg_threshold) if seg_threshold is not None else 0.5
            seg_metrics = make_seg_val_metrics(
                num_classes=seg_nc,
                threshold=seg_thr,
                ignore_index=seg_ignore_index,
            )
            # namespace seg/* to avoid key collisions
            for k, m in seg_metrics.items():
                m.attach(evaluator, f"seg/{k}")
            attached.update({f"seg/{k}": v for k, v in seg_metrics.items()})
        except Exception:
            pass

    # optional one-shot positive-score debug
    if enable_pos_score_debug_once:
        attach_pos_score_debug_once(
            evaluator=evaluator,
            positive_index=int(positive_index),
            bins=int(pos_debug_bins),
            tag=str(pos_debug_tag),
        )

    # optional per-epoch threshold search (writes summaries into evaluator.state.metrics)
    if enable_threshold_search and has_cls:
        attach_val_threshold_search(
            evaluator=evaluator,
            mode=str(calibration_method),
            positive_index=int(positive_index),
        )

    return attached


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
    "attach_engine_metrics",
    "attach_scheduler",
    "attach_lr_scheduling",
    "attach_pos_score_debug_once",
    "attach_val_threshold_search",
    "attach_val_stack",
    "attach_early_stopping",
    "attach_best_checkpoint",
    "build_trainer",
    "build_evaluator",
    "decision_from_logits",
    "get_label_indices_from_batch",
    "make_prepare_batch",
    "make_warmup_safe_score_fn",
    "maybe_update_best",
]
