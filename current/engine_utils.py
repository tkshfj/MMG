# engine_utils.py
import torch
from typing import Any, Callable, Dict, Optional, Tuple
from functools import partial
from monai.engines import SupervisedTrainer, SupervisedEvaluator

_ALLOWED = {"classification", "segmentation", "multitask"}
_ALIASES = {"cls": "classification", "seg": "segmentation"}


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
    metrics: Optional[Dict[str, Any]] = None,
    key_val_metric: Optional[Any] = None,
    *,
    decollate: bool = False,
    postprocessing: Optional[Callable] = None,
    inferer: Optional[Callable] = None,
) -> SupervisedEvaluator:
    """
    Build a MONAI SupervisedEvaluator.
    - `metrics` are passed as `additional_metrics`.
    - `key_val_metric` can be used by handlers (e.g., early stopping / model selection).
    - Set `decollate=False` to keep batched tensors in `engine.state.output`.
    """
    assert callable(prepare_batch), "prepare_batch must be callable"

    # Warn once if a postprocess is supplied but will be ignored due to decollate=False
    if postprocessing is not None and not decollate and not getattr(build_evaluator, "_warned_pp", False):
        print("[build_evaluator] NOTE: postprocessing provided but decollate=False; postprocessing will be skipped.")
        build_evaluator._warned_pp = True

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_data_loader,
        network=network,
        key_val_metric=key_val_metric,
        additional_metrics=metrics,
        prepare_batch=prepare_batch,
        decollate=decollate,
        postprocessing=postprocessing if decollate else None,
        inferer=inferer,
    )
    return evaluator


# def build_evaluator(
#     device: Any,
#     val_data_loader: Any,
#     network: Any,
#     prepare_batch: Callable,
#     metrics: Optional[Dict[str, Any]] = None,
#     key_val_metric: Optional[Any] = None,
#     decollate: Optional[Callable] = None,
#     postprocessing: Optional[Callable] = None,
#     inferer: Optional[Callable] = None,
# ) -> SupervisedEvaluator:
#     """
#     Build a MONAI SupervisedEvaluator.
#     - `metrics` are passed as `additional_metrics` (preferred to manual attach).
#     - `key_val_metric` can be used by handlers (e.g., early stopping / model selection).
#     """
#     assert callable(prepare_batch), "prepare_batch must be callable"
#     return SupervisedEvaluator(
#         device=device,
#         val_data_loader=val_data_loader,
#         network=network,
#         key_val_metric=key_val_metric,
#         additional_metrics=metrics,
#         prepare_batch=prepare_batch,
#         decollate=False,
#         postprocessing=None,
#         inferer=None,
#     )


def _normalize_task(task: str) -> str:
    t = (task or "").strip().lower()
    if t in _ALIASES:
        # print the alias notice only once to avoid log spam
        if not getattr(_normalize_task, "_alias_warned", False):
            print(f"[prepare_batch] NOTE: task='{task}' -> '{_ALIASES[t]}' (alias)")
            _normalize_task._alias_warned = True
        t = _ALIASES[t]
    if t not in _ALLOWED:
        raise ValueError(f"prepare_batch: invalid task='{task}'. Must be one of {_ALLOWED}.")
    return t


def make_prepare_batch(task: str) -> Callable:
    """
    Factory returning a `(batch, device, non_blocking)` callable for MONAI.
    Normalizes `task` once up front; avoids per-iteration normalization/logging.
    """
    task = _normalize_task(task)
    return partial(prepare_batch, task=task)


def prepare_batch(
    batch: Dict[str, Any],
    device=None,
    non_blocking: bool = False,
    task: str = "multitask",
) -> Tuple[torch.Tensor, Any, tuple, dict]:
    """
    Returns (inputs, targets, args, kwargs) for MONAI SupervisedTrainer/Evaluator.

    task:
      - "classification": targets = LongTensor (B,)
      - "segmentation" : targets = FloatTensor (B,1,H,W)
      - "multitask"    : targets = {"label": LongTensor(B,), "mask": FloatTensor(B,1,H,W)}
    """
    # task is assumed normalized if make_prepare_batch() was used; still safe if not.
    if task not in _ALLOWED:
        task = _normalize_task(task)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # inputs
    x = batch["image"].to(device=device, non_blocking=non_blocking)

    # targets
    y_cls = batch.get("label")
    if y_cls is not None:
        if y_cls.dtype != torch.long:
            y_cls = y_cls.long()
        y_cls = y_cls.to(device=device, non_blocking=non_blocking)

    y_seg = batch.get("mask")
    if y_seg is not None:
        if y_seg.ndim == 3:  # (B,H,W) -> (B,1,H,W)
            y_seg = y_seg.unsqueeze(1)
        if y_seg.dtype != torch.float32:
            y_seg = y_seg.float()
        y_seg = y_seg.to(device=device, non_blocking=non_blocking)

    # pick targets by task
    if task == "classification":
        if y_cls is None:
            raise ValueError("prepare_batch(task='classification'): 'label' missing in batch.")
        targets = y_cls
    elif task == "segmentation":
        if y_seg is None:
            raise ValueError("prepare_batch(task='segmentation'): 'mask' missing in batch.")
        targets = y_seg
    else:  # multitask
        if y_cls is None or y_seg is None:
            raise ValueError("prepare_batch(task='multitask'): need both 'label' and 'mask'.")
        targets = {"label": y_cls, "mask": y_seg}

    # one-time debug print
    if not getattr(prepare_batch, "_printed", False):
        print(f"[prepare_batch] task={task} | keys={list(batch.keys())}")
        print(f"  image: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
        if y_cls is not None:
            print(f"  label: shape={tuple(y_cls.shape)}, dtype={y_cls.dtype}, "
                  f"min={y_cls.min().item()}, max={y_cls.max().item()}")
        if y_seg is not None:
            uniq = torch.unique(y_seg.detach().flatten())[:6].tolist()
            print(f"  mask:  shape={tuple(y_seg.shape)}, dtype={y_seg.dtype}, unique[:6]={uniq}")
        prepare_batch._printed = True

    return x, targets, (), {}


# def build_trainer(
#     device: Any,
#     max_epochs: int,
#     train_data_loader: Any,
#     network: Any,
#     optimizer: Any,
#     loss_function: Callable,
#     prepare_batch: Callable,
#     output_transform: Optional[Callable] = None,
#     metrics: Optional[Dict[str, Any]] = None,
# ) -> SupervisedTrainer:
#     """
#     Build a MONAI SupervisedTrainer using configuration-supplied components.
#     All arguments are protocol-driven, so this trainer is fully model and task agnostic.
#     - If metrics are supplied, they will be attached at construction (optional; can also be attached externally).
#     - If output_transform is not supplied, the default will be used.
#     """
#     assert callable(loss_function), "loss_function must be callable"
#     assert callable(prepare_batch), "prepare_batch must be callable"
#     return SupervisedTrainer(
#         device=device,
#         max_epochs=max_epochs,
#         train_data_loader=train_data_loader,
#         network=network,
#         optimizer=optimizer,
#         loss_function=loss_function,
#         prepare_batch=prepare_batch,
#     )

# def build_evaluator(
#     device: Any,
#     val_data_loader: Any,
#     network: Any,
#     prepare_batch: Callable,
#     metrics: Dict[str, Any],
#     output_transform: Optional[Callable] = None,
# ) -> SupervisedEvaluator:
#     """
#     Build a MONAI SupervisedEvaluator using configuration-supplied components.
#     Metrics and output_transform should always be obtained from the model class.
#     """
#     assert callable(prepare_batch), "prepare_batch must be callable"
#     evaluator = SupervisedEvaluator(
#         device=device,
#         val_data_loader=val_data_loader,
#         network=network,
#         prepare_batch=prepare_batch,
#     )
    # DEBUG: Attach metrics after construction
    # if metrics:
    #     print("[DEBUG] Attaching metrics to evaluator:")
    #     for name, metric in metrics.items():
    #         print(f"  {name} -> {type(metric)}")
    #         metric.attach(evaluator, name)
    #     print("------")
    # return evaluator


# # standard MONAI/Ignite returning dict
# def _normalize_task(task: str) -> str:
#     t = (task or "").strip().lower()
#     if t in _ALIASES:
#         print(f"[prepare_batch] NOTE: task='{task}' -> '{_ALIASES[t]}' (alias)")
#         t = _ALIASES[t]
#     if t not in _ALLOWED:
#         raise ValueError(f"prepare_batch: invalid task='{task}'. Must be one of {_ALLOWED}.")
#     return t


# def prepare_batch(batch: Dict[str, Any],
#                   device=None,
#                   non_blocking: bool = False,
#                   task: str = "multitask") -> Tuple[torch.Tensor, Any, tuple, dict]:
#     """
#     Returns (inputs, targets, args, kwargs) for MONAI SupervisedTrainer/Evaluator.

#     task:
#       - "classification": targets = LongTensor (B,)
#       - "segmentation" : targets = FloatTensor (B,1,H,W)
#       - "multitask"    : targets = {"label": LongTensor(B,), "mask": FloatTensor(B,1,H,W)}
#     """
#     task = _normalize_task(task)
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # inputs
#     x = batch["image"].to(device=device, non_blocking=non_blocking)

#     # targets
#     y_cls = batch.get("label")
#     if y_cls is not None:
#         if y_cls.dtype != torch.long:
#             y_cls = y_cls.long()
#         y_cls = y_cls.to(device=device, non_blocking=non_blocking)

#     y_seg = batch.get("mask")
#     if y_seg is not None:
#         if y_seg.ndim == 3:  # (B,H,W) -> (B,1,H,W)
#             y_seg = y_seg.unsqueeze(1)
#         if y_seg.dtype != torch.float32:
#             y_seg = y_seg.float()
#         y_seg = y_seg.to(device=device, non_blocking=non_blocking)

#     # pick targets by task
#     if task == "classification":
#         if y_cls is None:
#             raise ValueError("prepare_batch(task='classification'): 'label' missing in batch.")
#         targets = y_cls
#     elif task == "segmentation":
#         if y_seg is None:
#             raise ValueError("prepare_batch(task='segmentation'): 'mask' missing in batch.")
#         targets = y_seg
#     else:  # multitask
#         if y_cls is None or y_seg is None:
#             raise ValueError("prepare_batch(task='multitask'): need both 'label' and 'mask'.")
#         targets = {"label": y_cls, "mask": y_seg}

#     # one-time debug print
#     if not getattr(prepare_batch, "_printed", False):
#         print(f"[prepare_batch] task={task} | keys={list(batch.keys())}")
#         print(f"  image: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
#         if y_cls is not None:
#             print(f"  label: shape={tuple(y_cls.shape)}, dtype={y_cls.dtype}, "
#                   f"min={y_cls.min().item()}, max={y_cls.max().item()}")
#         if y_seg is not None:
#             uniq = torch.unique(y_seg.detach().flatten())[:6].tolist()
#             print(f"  mask:  shape={tuple(y_seg.shape)}, dtype={y_seg.dtype}, unique[:6]={uniq}")
#         prepare_batch._printed = True

#     return x, targets, (), {}


# # def prepare_batch(batch, device=None, non_blocking=False, task="multitask"):
# #     print("[DEBUG] Using dict-version prepare_batch")
# #     # Flat dict: 'image', 'label', 'mask'
# #     images = batch["image"].to(device=device, non_blocking=non_blocking)
# #     label = batch.get("label")
# #     mask = batch.get("mask")
# #     if label is not None:
# #         label = label.to(device=device, non_blocking=non_blocking)
# #     if mask is not None:
# #         mask = mask.to(device=device, non_blocking=non_blocking)
# #     # Return flat tensors, not nested dicts
# #     # Training loop/model should expect and return these
# #     batch_out = {"image": images}
# #     if label is not None:
# #         batch_out["label"] = label
# #     if mask is not None:
# #         batch_out["mask"] = mask
# #     return batch_out

# # For a pure PyTorch custom loop returning (img, label, mask) unpacking (tuple)
# # def prepare_batch(batch, device=None, non_blocking=False, task="multitask"):
# #     images = batch["image"].to(device=device, non_blocking=non_blocking)
# #     label = batch.get("label")
# #     mask = batch.get("mask")
# #     if label is not None:
# #         label = label.to(device=device, non_blocking=non_blocking)
# #     if mask is not None:
# #         mask = mask.to(device=device, non_blocking=non_blocking)
# #     if task == "classification":
# #         return images, label
# #     elif task == "segmentation":
# #         return images, mask
# #     elif task == "multitask":
# #         return images, label, mask
# #     else:
# #         return images
