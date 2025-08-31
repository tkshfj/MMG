# utils/safe.py
from __future__ import annotations

from typing import Any, Mapping

import math
import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor


def to_float_scalar(v: Any, *, strict: bool = False) -> float | None:
    """
    Convert many numeric-like shapes to a float scalar.
    - strict=True: only accept scalar-like (0-D tensor/np scalar/python number)
    - strict=False: also collapse tensors/arrays/lists/dicts via mean()
    """
    # Tensors
    if torch.is_tensor(v):
        if v.numel() == 0:
            return 0.0 if not strict else None
        if strict and v.ndim > 0:
            return None
        return float(v.detach().float().mean().item())
    # NumPy arrays & scalars
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return 0.0 if not strict else None
        if strict and v.ndim > 0:
            return None
        return float(v.mean()) if v.ndim > 0 else float(v.item())
    if isinstance(v, (np.floating, np.integer, float, int)):
        return float(v)
    # Containers (only if non-strict)
    if not strict:
        if isinstance(v, (list, tuple)):
            vals = [to_float_scalar(x, strict=False) for x in v]
            vals = [x for x in vals if x is not None]
            return float(np.mean(vals)) if vals else None
        if isinstance(v, Mapping):
            vals = [to_float_scalar(x, strict=False) for x in v.values()]
            vals = [x for x in vals if x is not None]
            return float(np.mean(vals)) if vals else None
    return None


def is_finite_float(x: float | None) -> bool:
    """Check if x is a finite float (not None, not NaN, not Inf)."""
    return x is not None and math.isfinite(x)


def to_tensor(x: Any) -> torch.Tensor:
    """Convert MetaTensor/array/scalar to torch.Tensor; raise if not possible."""
    if isinstance(x, MetaTensor):
        x = x.as_tensor()
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception as e:
        raise TypeError(f"to_tensor: cannot convert type {type(x)}") from e


def to_py(x: Any) -> Any:
    """Convert tensors/arrays/mappings to JSON-serializable Python values."""
    x = to_tensor(x) if isinstance(x, (MetaTensor, torch.Tensor)) else x
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, Mapping):
        return {k: to_py(v) for k, v in x.items()}
    try:
        return float(x)
    except Exception:
        return str(x)


def labels_to_1d_indices(y: Any) -> torch.Tensor:
    """Coerce labels to shape [B] int64, handling one-hot and [B,1] forms."""
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if y.ndim >= 2:
        if y.shape[-1] > 1:  # one-hot
            y = y.argmax(dim=-1)
        else:
            y = y.view(y.shape[0], -1).squeeze(-1)
    return y.view(-1).to(torch.long)
