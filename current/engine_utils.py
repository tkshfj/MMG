# engine_utils.py
from __future__ import annotations
import torch
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, Union
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from ignite.metrics import Metric


def build_trainer(
    device: Any,
    max_epochs: int,
    train_data_loader: Any,
    network: Any,
    optimizer: Any,
    loss_function: Callable,
    prepare_batch: Callable,
) -> SupervisedTrainer:
    """Build a MONAI SupervisedTrainer (task/model agnostic)."""
    assert callable(loss_function), "loss_function must be callable"
    assert callable(prepare_batch), "prepare_batch must be callable"
    return SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_data_loader,
        network=network,
        optimizer=optimizer,
        loss_function=loss_function,
        prepare_batch=prepare_batch,
    )


def build_evaluator(
    device: Any,
    val_data_loader: Any,
    network: Any,
    prepare_batch: Callable,
    metrics: Optional[Dict[str, Metric]] = None,
    key_val_metric: Optional[Metric] = None,
    *,
    decollate: bool = False,
    postprocessing: Optional[Callable] = None,
    inferer: Optional[Callable] = None,
) -> SupervisedEvaluator:
    assert callable(prepare_batch), "prepare_batch must be callable"

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_data_loader,
        network=network,
        key_val_metric=None,
        additional_metrics={},
        prepare_batch=prepare_batch,
        decollate=decollate,
        postprocessing=postprocessing if decollate else None,
        inferer=inferer,
    )

    # Explicitly attach metrics so evaluator.state.metrics is guaranteed
    if key_val_metric is not None:
        key_val_metric.attach(evaluator, "key_val_metric")
    if metrics:
        for name, m in metrics.items():
            m.attach(evaluator, name)

    return evaluator


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
    seg_target: Literal["indices", "channels"] = "indices",  # NEW: choose target form
    seg_float_dtype: Optional[torch.dtype] = None,           # used if seg_target="channels"
) -> PrepareBatch:
    """
    Returns prepare_batch(batch, device, non_blocking) -> (x, targets).
      classification: (x, label[B] Long)
      segmentation  : (x, mask[B,H,W] Long)                # default ('indices')
      multitask     : (x, {"label": [B] Long, "mask": [B,H,W] Long})
    If seg_target="channels", returns mask as [B, 1 or C, H, W] float (for pure Dice, etc.).
    """
    t = _normalize_task(task)
    _printed_once = {"flag": False}

    def _prepare_batch(
        batch: Union[Dict[str, Any], Sequence[Any]],
        device: Optional[torch.device] = None,
        non_blocking: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        dev = device or torch.device("cpu")
        bdict = _to_batch_dict(batch)

        # inputs
        x = _pick(bdict, "image", "images", "x")
        if x is None:
            raise KeyError("Batch missing image tensor (expected one of: 'image', 'images', 'x').")
        x = _to_device(_to_tensor(x).float(), dev, non_blocking)

        # labels (classification)
        y_cls = _pick(bdict, "label", "classification", "class", "target", "y")
        if y_cls is not None:
            y_cls = _normalize_labels(_to_tensor(y_cls))
            y_cls = _to_device(y_cls, dev, non_blocking)

        # masks (segmentation)
        y_seg = _pick(bdict, "mask", "seg", "segmentation", "mask_label", "mask_true")
        if y_seg is not None:
            y_seg = _normalize_masks(
                _to_tensor(y_seg),
                target=seg_target,
                float_dtype=seg_float_dtype or torch.float32,
            )
            y_seg = _to_device(y_seg, dev, non_blocking)

        # select targets by task
        if t is Task.CLASSIFICATION:
            if y_cls is None:
                raise ValueError("prepare_batch(classification): missing 'label'.")
            targets = y_cls

        elif t is Task.SEGMENTATION:
            if y_seg is None:
                raise ValueError("prepare_batch(segmentation): missing 'mask'.")
            targets = y_seg

        else:  # MULTITASK
            if y_cls is None or y_seg is None:
                raise ValueError("prepare_batch(multitask): need both 'label' and 'mask'.")
            targets = {"label": y_cls, "mask": y_seg}

        # one-time debug
        if debug and not _printed_once["flag"]:
            _debug_print(t, x, y_cls, y_seg, num_classes, binary_label_check)
            _printed_once["flag"] = True

        return x, targets

    return _prepare_batch


def _normalize_task(task: Union[str, Task]) -> Task:
    if isinstance(task, Task):
        return task
    t = (task or "").lower()
    if t in {"cls", "classification", "classify"}:
        return Task.CLASSIFICATION
    if t in {"seg", "segmentation", "segment"}:
        return Task.SEGMENTATION
    return Task.MULTITASK


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            return d[k]
    return None


def _to_tensor(x: Any) -> torch.Tensor:
    # unwrap MetaTensor
    try:
        from monai.data.meta_tensor import MetaTensor
        if isinstance(x, MetaTensor):
            x = x.as_tensor()
    except Exception:
        pass
    if isinstance(x, torch.Tensor):
        return x
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
    except Exception:
        pass
    return torch.as_tensor(x)


def _to_device(t: torch.Tensor, device: torch.device, non_blocking: bool) -> torch.Tensor:
    return t.to(device=device, non_blocking=non_blocking)


def _tuple_batch_to_dict(seq: Sequence[Any]) -> Dict[str, Any]:
    # Supports (x, y) or (x, y, m)
    n = len(seq)
    if n == 2:
        x, y = seq
        if isinstance(y, dict):
            out = {"image": x}
            out.update(y)
            return out
        return {"image": x, "label": y}
    if n == 3:
        x, y, m = seq
        if isinstance(y, dict):
            out = {"image": x, "mask": m}
            out.update(y)
            return out
        return {"image": x, "label": y, "mask": m}
    raise TypeError(f"Unsupported tuple/list batch length: {n} (expected 2 or 3)")


def _to_batch_dict(batch: Union[Dict[str, Any], Sequence[Any]]) -> Dict[str, Any]:
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (tuple, list)):
        return _tuple_batch_to_dict(batch)
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _normalize_labels(y: torch.Tensor) -> torch.Tensor:
    """Return class indices [B] Long."""
    if y.ndim >= 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    elif y.ndim >= 2 and y.shape[-1] > 1:  # one-hot or logits
        y = y.argmax(dim=-1)
    if y.ndim > 1:
        y = y.squeeze()
    return y.long()


def _normalize_masks(
    y: torch.Tensor,
    *,
    target: Literal["indices", "channels"],
    float_dtype: torch.dtype,
) -> torch.Tensor:
    """
    indices:  return [B,H,W] Long (class indices). Accepts [H,W], [B,H,W], [B,1,H,W], [B,C,H,W].
    channels: return [B,1 or C,H,W] float (for Dice-only learners).
    """
    if target == "indices":
        # [H,W] -> [1,H,W]
        if y.ndim == 2:
            y = y.unsqueeze(0)
        # [B,1,H,W] -> [B,H,W], [B,C,H,W] -> argmax
        if y.ndim == 4:
            y = y[:, 0] if y.shape[1] == 1 else y.argmax(dim=1)
        # else assume [B,H,W]
        return y.long()

    # channels (float)
    if y.ndim == 2:          # [H,W] -> [1,1,H,W]
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.ndim == 3:        # [B,H,W] -> [B,1,H,W]
        y = y.unsqueeze(1)
    # else assume [B,1 or C,H,W]
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


__all__ = ["Task", "PrepareBatch", "make_prepare_batch"]
