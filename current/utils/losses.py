# utils/losses.py
from __future__ import annotations
from typing import Sequence, Optional
import torch
from torch import nn
from monai.losses import DiceLoss


def compute_class_weights(
    class_counts: Sequence[int] | torch.Tensor,
    *,
    min_w: float = 0.5,
    max_w: float = 10.0,
    eps: float = 1e-6,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    inv = 1.0 / (counts + eps)
    w = inv / inv.mean()  # normalize around ~1
    return w.clamp_(min=min_w, max=max_w)


def make_seg_loss(
    class_counts: Sequence[int] | torch.Tensor,
    *,
    alpha: float = 1.0,   # CE weight
    beta: float = 0.5,    # Dice weight
    device: Optional[torch.device] = None
):
    w = compute_class_weights(class_counts, device=device)  # [K]
    ce = nn.CrossEntropyLoss(weight=w)
    dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=True, weight=w)

    def _loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return alpha * ce(logits, y.long()) + beta * dice(logits, y.long())
    return _loss
