# optim_factory.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


# Scheduler spec
@dataclass
class SchedulerSpec:
    """Wrapper describing how to drive the scheduler in Ignite."""
    # scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]  # ReduceLROnPlateau ok too
    scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]]
    step_cadence: Optional[str]  # 'epoch' | 'iteration' | 'plateau' | None
    needs_train_len: bool = False  # True for OneCycle
    # stash for late OneCycle construction
    _onecycle_kwargs: Optional[Dict[str, float]] = None


# Param-group handling
def _extract_param_groups(parameters: Any) -> List[Dict[str, Any]]:
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
    # iterable of tensors or a list of param-group dicts
    from collections.abc import Iterable
    if isinstance(parameters, Iterable):
        params_list = list(parameters)
        if len(params_list) == 0:
            raise ValueError("get_optimizer: empty parameters iterable.")
        first = params_list[0]
        if isinstance(first, dict):
            return list(params_list)  # list of param group dicts
        return [{"params": params_list}]  # list/iterable of tensors

    raise ValueError("get_optimizer: unsupported parameters input.")


def _dedupe_params(params):
    """Preserve order, remove duplicates by id()."""
    seen = set()
    out = []
    for p in params:
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            out.append(p)
    return out


def _disjoint(a: List[torch.nn.Parameter], b: List[torch.nn.Parameter]) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Return a,b with duplicates removed from b if they appeared in a (by id)."""
    a_ids = {id(p) for p in a}
    b = [p for p in b if id(p) not in a_ids]
    return a, b


def _tag_param_names(model: torch.nn.Module) -> None:
    """Attach a best-effort name and owning module to each Parameter (for diagnostics/filters)."""
    for module_name, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            full = f"{module_name}.{name}" if module_name else name
            setattr(p, "_param_name", full)
            setattr(p, "_owner_module", module)


def _consume_model_param_groups(parameters: Any) -> Optional[Dict[str, List[torch.nn.Parameter]]]:
    """
    If the model exposes param_groups(), normalize it to {name: [params,...]}.
    Filters to requires_grad params and dedupes each group.
    """
    if not hasattr(parameters, "param_groups") or not callable(getattr(parameters, "param_groups")):
        return None
    pg = parameters.param_groups()
    groups: Dict[str, List[torch.nn.Parameter]] = {}
    if isinstance(pg, dict):
        for k, v in pg.items():
            lst = list(v)
            lst = [p for p in lst if getattr(p, "requires_grad", False)]
            groups[str(k)] = _dedupe_params(lst)
    elif isinstance(pg, (list, tuple)):
        # Accept list of lists or list of dicts with "params"
        for i, g in enumerate(pg):
            if isinstance(g, dict) and "params" in g:
                lst = list(g["params"])
            else:
                lst = list(g)
            lst = [p for p in lst if getattr(p, "requires_grad", False)]
            groups[f"g{i}"] = _dedupe_params(lst)
    else:
        return None
    return groups if any(groups.values()) else None


def _split_decay_by_name(named_params: Iterable[Tuple[str, torch.nn.Parameter]]
                         ) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Split into (decay, no_decay) following AdamW practice using PARAM NAMES:
    no_decay: biases, *norm* weights, embeddings
    decay:    everything else
    """
    decay: List[torch.nn.Parameter] = []
    nodecay: List[torch.nn.Parameter] = []
    for n, p in named_params:
        if not getattr(p, "requires_grad", False):
            continue
        n_low = n.lower()
        # biases and 1D params
        if n_low.endswith(".bias") or p.ndim <= 1:
            nodecay.append(p)
            continue
        # common normalization / embedding identifiers in names
        if any(k in n_low for k in ("bn", "norm", "layernorm", "ln", "gn", "embedding", "embeddings", "pos_embed")):
            nodecay.append(p)
            continue
        decay.append(p)
    return _dedupe_params(decay), _dedupe_params(nodecay)


# Optimizer factory
def get_optimizer(cfg: Mapping[str, Any], parameters: Any, *, verbose: bool = False) -> Optimizer:
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

    # handle split param groups here if a model was passed in and we can identify head vs backbone
    pg_mode = str(cfg.get("param_groups", "single")).lower()
    if pg_mode in ("split", "auto") and hasattr(parameters, "parameters"):
        base_lr = float(cfg.get("lr", 1e-3))
        # Allow boosting (>1) or down-scaling (<1) of head LR; clamp to a safe range
        head_lr_scale = float(cfg.get("head_lr_scale", 1.0))
        head_lr_scale = float(min(max(head_lr_scale, 0.05), 10.0))  # [0.05, 10]
        head_wd_scale = float(cfg.get("head_wd_scale", 1.0))
        head_wd_scale = float(max(head_wd_scale, 0.0))              # allow 0..∞ (0 disables WD on head)
        use_decay_split = bool(cfg.get("decay_split", True))         # split decay/no-decay per group

        # Prefer model.param_groups() if available (exact control from the model)
        _tag_param_names(parameters)  # helpful for other tools; not required here
        model_groups = _consume_model_param_groups(parameters)
        head_params = model_groups.get("head") if model_groups is not None else None
        backbone_params = model_groups.get("backbone") if model_groups is not None else None

        # Prefer explicit helpers on the model (if param_groups() not provided or incomplete)        backbone_params = None
        if head_params is None and hasattr(parameters, "head_parameters") and callable(getattr(parameters, "head_parameters")):
            head_params = list(parameters.head_parameters())
        if backbone_params is None and hasattr(parameters, "backbone_parameters") and callable(getattr(parameters, "backbone_parameters")):
            backbone_params = list(parameters.backbone_parameters())

        # Fallback: if only head is known, derive backbone as the complement
        if head_params is not None and backbone_params is None:
            head_ids = {id(p) for p in head_params}
            backbone_params = [p for p in parameters.parameters() if getattr(p, "requires_grad", False) and id(p) not in head_ids]

        # Name-based fallback if helpers missing
        if head_params is None or backbone_params is None:
            head_keys = tuple(k.lower() for k in cfg.get("head_keys", ("classification_head", "head", "classifier", "mlp_head", "fc", "cls")))
            heads, backs = [], []
            for n, p in parameters.named_parameters():
                if not getattr(p, "requires_grad", False):
                    continue
                (heads if any(k in n.lower() for k in head_keys) else backs).append(p)
            if heads and backs:
                head_params, backbone_params = heads, backs

        # Clean up and sanity-check splits
        if head_params is not None and backbone_params is not None:
            head_params = _dedupe_params([p for p in head_params if getattr(p, "requires_grad", False)])
            backbone_params = _dedupe_params([p for p in backbone_params if getattr(p, "requires_grad", False)])
            head_params, backbone_params = _disjoint(head_params, backbone_params)

        # If we can split, build per-group settings; optionally split decay/no-decay inside each
        if head_params and backbone_params:
            wd = float(cfg.get("weight_decay", 1e-4))
            param_groups: List[Dict[str, Any]] = []
            if use_decay_split:
                # Use names for reliable filtering
                bb_named = [(n, p) for n, p in parameters.named_parameters() if p in set(backbone_params)]
                hd_named = [(n, p) for n, p in parameters.named_parameters() if p in set(head_params)]
                bb_decay, bb_no = _split_decay_by_name(bb_named)
                hd_decay, hd_no = _split_decay_by_name(hd_named)
                if bb_decay:
                    param_groups.append({"params": bb_decay, "lr": base_lr, "weight_decay": wd})
                if bb_no:
                    param_groups.append({"params": bb_no, "lr": base_lr, "weight_decay": 0.0})
                if hd_decay:
                    param_groups.append({"params": hd_decay, "lr": base_lr * head_lr_scale, "weight_decay": wd * head_wd_scale})
                if hd_no:
                    param_groups.append({"params": hd_no, "lr": base_lr * head_lr_scale, "weight_decay": 0.0})
            else:
                param_groups.extend([
                    {"params": backbone_params, "lr": base_lr, "weight_decay": wd},
                    {"params": head_params, "lr": base_lr * head_lr_scale, "weight_decay": wd * head_wd_scale},
                ])
            opt_kwargs: Dict[str, Any] = {"weight_decay": weight_decay}
            if name in ("sgd", "rmsprop"):
                opt_kwargs["momentum"] = momentum
            # Per-group LR/WD already set above; pass no global lr/wd overrides.
            optimizer = opts[name](param_groups, **({k: v for k, v in opt_kwargs.items() if k != "weight_decay"}))
            if verbose:
                _print_param_group_summary(optimizer, parameters)
            return optimizer
        elif pg_mode in ("split", "auto") and verbose:
            print("[optim] split requested but could not identify both head and backbone; falling back to single group.")

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
        _print_param_group_summary(optimizer, parameters)

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
        plateau_thr = float(
            cfg.get("plateau_threshold", cfg.get("lr_plateau_threshold", 1e-4))
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.get("plateau_mode", "min")).lower(),
            patience=int(cfg.get("patience", 3)),
            factor=float(cfg.get("factor", 0.5)),
            threshold=plateau_thr,
            threshold_mode=str(cfg.get("threshold_mode", "rel")).lower(),  # 'rel' or 'abs'
            cooldown=int(cfg.get("cooldown", 0)),
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


# diagnostics
def _print_param_group_summary(optimizer: Optimizer, model: Optional[torch.nn.Module] = None) -> None:
    total = 0
    if model is not None:
        total = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))
        print("[optim] num trainable params:", total)
    for i, g in enumerate(optimizer.param_groups):
        n = 0
        for p in g["params"]:
            try:
                n += p.numel()
            except Exception:
                pass
        lr = g.get("lr")
        wd = g.get("weight_decay")
        print(f"[optim] group {i}: {len(g['params'])} tensors, {n} params, lr={lr}, wd={wd}")


def grad_norms_by_group(optimizer: Optimizer) -> Dict[str, float]:
    """
    Returns L2 grad norms per optimizer group and total. Call after loss.backward().
    Example:
        norms = grad_norms_by_group(optim); log_fn({f\"grad/gnorm_g{i}\": v for i,v in norms.items()})
    """
    out: Dict[str, float] = {}
    total_sq = 0.0
    for i, g in enumerate(optimizer.param_groups):
        s = 0.0
        for p in g["params"]:
            if p.grad is not None:
                v = p.grad.data
                s += float(v.pow(2).sum().item())
        out[f"g{i}"] = (s ** 0.5) if s > 0 else 0.0
        total_sq += s
    out["total"] = (total_sq ** 0.5) if total_sq > 0 else 0.0
    return out


def zero_grad_if_nan(optimizer: Optimizer) -> bool:
    """
    Zero grads in-place and return True if a NaN/Inf was found in any group.
    Handy safety net before optimizer.step().
    """
    bad = False
    for g in optimizer.param_groups:
        for p in g["params"]:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    bad = True
    if bad:
        optimizer.zero_grad(set_to_none=True)
    return bad


def make_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, train_loader=None):
    """
    Pure factory: builds and returns a torch scheduler (no event wiring).
    Supported: none, cosine, step, multistep, exponential, plateau, onecycle, linear_warmup_cosine
    """
    name = str(cfg.get("lr_scheduler", "none") or "none").lower()

    if name in ("none", "", "off", "false"):
        return None

    if name in ("cosine", "cosineannealing", "cos"):
        T_max = int(cfg.get("T_max", cfg.get("epochs", 40)))
        eta_min = float(cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    if name in ("step", "steplr"):
        step_size = int(cfg.get("step_size", 10))
        gamma = float(cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name in ("multistep", "multisteplr", "multi_step"):
        milestones = cfg.get("milestones", [30, 60, 90])
        if isinstance(milestones, str):
            import re
            milestones = [int(x) for x in re.findall(r"\d+", milestones)]
        gamma = float(cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if name in ("exp", "exponential"):
        gamma = float(cfg.get("gamma", 0.95))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if name in ("plateau", "reducelronplateau", "reduce"):
        mode = str(cfg.get("plateau_mode", "min"))
        factor = float(cfg.get("factor", 0.5))
        patience = int(cfg.get("patience", 3))
        threshold = float(cfg.get("plateau_threshold", cfg.get("lr_plateau_threshold", 1e-4)))
        # verbose = bool(cfg.get("plateau_verbose", False))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
        )

    if name in ("onecycle", "one_cycle"):
        if train_loader is None:
            raise ValueError("OneCycleLR requires train_loader to compute steps_per_epoch.")
        max_lr = float(cfg.get("max_lr", cfg.get("lr", 1e-3)))
        steps_per_epoch = len(train_loader)
        epochs = int(cfg.get("epochs", 40))
        pct_start = float(cfg.get("pct_start", 0.3))
        anneal_strategy = str(cfg.get("anneal_strategy", "cos")).lower()
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=float(cfg.get("div_factor", 25.0)),
            final_div_factor=float(cfg.get("final_div_factor", 1e4)),
        )

    if name in ("linear_warmup_cosine", "warmup_cosine", "warmcos", "warm_cos", "warmupcos"):
        warmup_epochs = int(cfg.get("warmup_epochs", 5))
        total_epochs = int(cfg.get("epochs", 40))

        def lr_lambda(current_epoch: int):
            if current_epoch < warmup_epochs:
                # linear warmup to 1.0
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            # cosine decay afterwards
            progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown lr_scheduler '{name}'")

# def _split_decay(params: Iterable[torch.nn.Parameter]) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
#     """
#     Split into (decay, no_decay) following AdamW practice:
#     - no_decay: biases, LayerNorm/GroupNorm/BatchNorm weights, embeddings (if any)
#     - decay: everything else
#     """
#     decay, no_decay = [], []
#     norm_types = (
#         torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d,
#         torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.InstanceNorm1d,
#         torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d
#     )
#     for p in params:
#         if not getattr(p, "requires_grad", False):
#             continue
#         name = getattr(p, "_param_name", "")  # optional (see _tag_param_names)
#         is_bias = name.endswith(".bias") if name else (p.ndim == 1)
#         owner = getattr(p, "_owner_module", None)
#         is_norm = isinstance(owner, norm_types) if owner is not None else False
#         if is_bias or is_norm:
#             no_decay.append(p)
#         else:
#             decay.append(p)
#     return _dedupe_params(decay), _dedupe_params(no_decay)


# def _split_decay(named_params):
#     decay, nodecay = [], []
#     for n, p in named_params:
#         if not p.requires_grad:
#             continue
#         if p.ndim <= 1 or any(k in n.lower() for k in ("bias","bn","norm","ln","layernorm","embedding")):
#             nodecay.append(p)
#         else:
#             decay.append(p)
#     return decay, nodecay
