# optim_utils.py
from typing import Any
from torch.optim import AdamW, Adam, SGD, RMSprop


def get_optimizer(
    name: str,
    parameters,
    lr: float | None = None,
    *,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    verbose: bool = False,
    **kwargs: Any,
):
    name = (name or "adamw").lower()
    opts = {"adamw": AdamW, "adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
    if name not in opts:
        raise ValueError(f"Unknown optimizer: {name}")

    if hasattr(parameters, "named_parameters"):  # a model
        params = [p for p in parameters.parameters() if p.requires_grad]
        if not params:
            raise ValueError("get_optimizer: model has no trainable parameters.")
        param_groups = [{"params": params}]
    else:
        param_groups = list(parameters)
        if not param_groups:
            raise ValueError("get_optimizer: empty parameter list.")

    opt_cls = opts[name]
    opt_kwargs: dict[str, Any] = {"weight_decay": float(weight_decay)}
    if name in ("sgd", "rmsprop"):
        opt_kwargs["momentum"] = float(momentum)

    needs_global_lr = not (
        isinstance(param_groups[0], dict) and all("lr" in g for g in param_groups)
    )
    if needs_global_lr:
        if lr is None:
            raise ValueError("get_optimizer: please provide `lr` when passing raw params.")
        optimizer = opt_cls(param_groups, lr=float(lr), **opt_kwargs)
    else:
        optimizer = opt_cls(param_groups, **opt_kwargs)

    if verbose and hasattr(parameters, "parameters"):
        total = sum(p.numel() for p in parameters.parameters() if p.requires_grad)
        print("num trainable params:", total)
        for i, g in enumerate(optimizer.param_groups):
            n = sum(p.numel() for p in g["params"])
            print(f"group {i}: {len(g['params'])} tensors, {n} params, lr={g.get('lr')}")

    return optimizer
