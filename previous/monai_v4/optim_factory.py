# optim_factory.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence
import torch
from torch.optim import Optimizer


# Scheduler spec
@dataclass
class SchedulerSpec:
    """
    Wrapper describing how to drive the scheduler in Ignite.
    - step_cadence: 'epoch' | 'iteration' | 'plateau' | None
    - needs_train_len: True for OneCycle (created later when steps_per_epoch is known)
    """
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]  # ReduceLROnPlateau ok too
    step_cadence: Optional[str]
    needs_train_len: bool = False

    # stash for late OneCycle construction
    _onecycle_kwargs: Optional[Dict[str, float]] = None


# Param-group handling
def _extract_param_groups(parameters: Any) -> List[Dict[str, Any]] | List[Dict[str, Any]]:
    """
    Accepts:
      - a model (has .parameters()) -> single group with all trainable params
      - an iterable of Tensors -> single group
      - a list of dict param groups (possibly with per-group lrs)
    Returns a list usable by torch optimizer.
    """
    # model
    if hasattr(parameters, "named_parameters") and callable(parameters.named_parameters):
        params = [p for p in parameters.parameters() if getattr(p, "requires_grad", False)]
        if not params:
            raise ValueError("get_optimizer: model has no trainable parameters.")
        return [{"params": params}]

    # already param groups?
    if isinstance(parameters, Sequence) and len(parameters) > 0:
        first = parameters[0]
        if isinstance(first, dict):
            return list(parameters)  # type: ignore[return-value]
        # iterable of tensors
        return [{"params": list(parameters)}]

    raise ValueError("get_optimizer: unsupported parameters input.")


# Optimizer factory
def get_optimizer(cfg: Mapping[str, Any], parameters: Any, *, verbose: bool = False) -> Optimizer:
    """
    Config-driven optimizer:
      - optimizer: adamw|adam|sgd|rmsprop  (case-insensitive)
      - lr: float (required unless every param group already provides lr)
      - weight_decay: float
      - momentum: float (for sgd/rmsprop)
    """
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = cfg.get("lr", None)
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    momentum = float(cfg.get("momentum", 0.9))

    opts = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    if name not in opts:
        raise ValueError(f"Unknown optimizer: {name}")

    param_groups = _extract_param_groups(parameters)
    opt_kwargs: Dict[str, Any] = {"weight_decay": weight_decay}
    if name in ("sgd", "rmsprop"):
        opt_kwargs["momentum"] = momentum

    if not (isinstance(param_groups[0], dict) and all("lr" in g for g in param_groups)):
        if lr is None:
            raise ValueError("get_optimizer: provide `cfg.lr` when passing raw params or models without per-group LRs.")
        optimizer = opts[name](param_groups, lr=float(lr), **opt_kwargs)
    else:
        optimizer = opts[name](param_groups, **opt_kwargs)

    if verbose and hasattr(parameters, "parameters"):
        total = sum(p.numel() for p in parameters.parameters() if getattr(p, "requires_grad", False))
        print("[optim] num trainable params:", total)
        for i, g in enumerate(optimizer.param_groups):
            n = sum(getattr(p, "numel", lambda: 0)() for p in g["params"])
            print(f"[optim] group {i}: {len(g['params'])} tensors, {n} params, lr={g.get('lr')}")

    return optimizer


# Scheduler factory
def get_scheduler(cfg: Mapping[str, Any], optimizer: Optimizer) -> SchedulerSpec:
    """
    Returns a SchedulerSpec telling the training loop how to step the scheduler.
    lr_strategy:
      - none/constant/single -> no scheduler
      - cosine -> CosineAnnealingLR (epoch cadence)
      - onecycle -> OneCycleLR (iteration cadence; constructed later)
      - plateau -> ReduceLROnPlateau (step after validation)
      - warmcos -> Linear warmup then cosine (SequentialLR, epoch cadence)
    """
    strat = str(cfg.get("lr_strategy", "none") or "none").lower()

    if strat in ("none", "constant", "single", ""):
        return SchedulerSpec(None, None)

    if strat == "cosine":
        T_max = int(cfg.get("T_max") or cfg.get("epochs") or 50)
        eta_min = float(cfg.get("eta_min", 0.0))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        return SchedulerSpec(scheduler=sch, step_cadence="epoch")

    if strat == "onecycle":
        # Create later in attach() when steps_per_epoch is known
        max_lr = float(cfg.get("max_lr", cfg.get("lr", 1e-3)))
        spec = SchedulerSpec(scheduler=None, step_cadence="iteration", needs_train_len=True)
        spec._onecycle_kwargs = dict(
            max_lr=max_lr,
            pct_start=float(cfg.get("pct_start", 0.3)),
            div_factor=float(cfg.get("div_factor", 25.0)),
            final_div_factor=float(cfg.get("final_div_factor", 1e4)),
        )
        return spec

    if strat == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.get("monitor_mode", "min")).lower(),
            patience=int(cfg.get("patience", 3)),
            factor=float(cfg.get("factor", 0.5)),
        )
        return SchedulerSpec(scheduler=sch, step_cadence="plateau")

    if strat == "warmcos":
        warmup_epochs = int(cfg.get("warmup_epochs", 3))
        warmup_start_factor = float(cfg.get("warmup_start_factor", 0.1))
        total_epochs = int(cfg.get("epochs", 50))
        T_max = max(1, total_epochs - warmup_epochs)

        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start_factor, total_iters=max(1, warmup_epochs)
        )
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=float(cfg.get("eta_min", 0.0))
        )
        sch = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cos], milestones=[warmup_epochs]
        )
        return SchedulerSpec(scheduler=sch, step_cadence="epoch")

    # Fallback
    return SchedulerSpec(None, None)
