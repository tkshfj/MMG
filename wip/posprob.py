# posprob.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Literal, Tuple
import torch
try:
    # avoid circular import at module import time
    from metrics_utils import extract_cls_logits_from_any  # noqa: F401
except Exception:
    extract_cls_logits_from_any = None  # type: ignore[misc]


def _as_tensor(x: Any) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def positive_score_from_logits(
    logits: Any,
    *,
    positive_index: int = 1,
    binary_single_logit: Optional[bool] = None,
    binary_bce_from_two_logits: Optional[bool] = None,
    mode: Literal["auto", "bce_single", "bce_two_logit", "ce_softmax"] = "auto",
) -> torch.Tensor:
    """
    Map raw logits -> P(positive) in shape [B] or [...], no thresholding here.
    Supports:
      - BCE single-logit:    logits [...], sigmoid(logit)
      - BCE two-logit:       logits [...,2], sigmoid(logit[:, pos_idx])
      - CE multi-class:      logits [...,C], softmax(...)[..., pos_idx]
    """
    t = _as_tensor(logits)
    last_dim = (t.ndim > 0 and t.shape[-1]) or 1

    if mode == "auto":
        # Prefer explicit hints; else infer from shape
        if binary_single_logit is True or (t.ndim == 1 or last_dim == 1):
            mode = "bce_single"
        elif binary_bce_from_two_logits:
            mode = "bce_two_logit"
        else:
            mode = "ce_softmax" if last_dim >= 2 else "bce_single"

    if mode == "bce_single":
        return torch.sigmoid(t).reshape(t.shape[:-1]) if (t.ndim > 1 and last_dim == 1) else torch.sigmoid(t)

    if mode == "bce_two_logit":
        if last_dim < 2:
            raise ValueError(f"bce_two_logit requires last dim >=2, got {tuple(t.shape)}")
        return torch.sigmoid(t[..., int(positive_index)])

    if mode == "ce_softmax":
        if last_dim < 2:
            raise ValueError(f"ce_softmax requires last dim >=2, got {tuple(t.shape)}")
        probs = torch.softmax(t, dim=-1)
        return probs[..., int(positive_index)]

    raise ValueError(f"Unknown mode='{mode}'")


@dataclass
class PosProbCfg:
    positive_index: int = 1
    binary_single_logit: bool = True
    binary_bce_from_two_logits: bool = False
    mode: Literal["auto", "bce_single", "bce_two_logit", "ce_softmax"] = "auto"

    def pick_logits(self, output: Any) -> torch.Tensor:
        # Accept dicts or raw tensors; cover all common keys from our evaluator
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            # Prefer a robust nested-extractor if available
            if extract_cls_logits_from_any is not None:
                try:
                    t = extract_cls_logits_from_any(output)
                    if torch.is_tensor(t):
                        return t
                except Exception:
                    pass
            # Fallback: common keys we emit
            for k in ("y_pred", "logits", "cls_out", "cls_logits", "class_logits"):
                v = output.get(k, None)
                if torch.is_tensor(v):
                    return v
        # Last resort: coerce
        return _as_tensor(output)

    def pick_labels(self, target: Any) -> torch.Tensor:
        cur = target  # unwrap nested dicts
        while isinstance(cur, dict):
            if "y" in cur and cur["y"] is not None:
                cur = cur["y"]
                continue
            if "label" in cur and cur["label"] is not None:
                cur = cur["label"]
                continue
            if "labels" in cur and cur["labels"] is not None:
                cur = cur["labels"]
                continue
            break  # no usable keys
        return _as_tensor(cur).long().view(-1)

    def logits_to_pos_prob(self, logits: Any) -> torch.Tensor:
        return positive_score_from_logits(
            logits,
            positive_index=self.positive_index,
            binary_single_logit=self.binary_single_logit,
            binary_bce_from_two_logits=self.binary_bce_from_two_logits,
            mode=self.mode,
        )

    # Callable interface for Ignite-friendly OTs
    def __call__(self, output: Any) -> torch.Tensor:
        return self.logits_to_pos_prob(self.pick_logits(output))

    # Convenience
    def probs_and_labels(self, output: Any, target: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        return self(output), self.pick_labels(target)
