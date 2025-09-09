# posprob.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Tuple, Iterable, Optional, Literal
import torch
from utils.safe import to_tensor


def positive_score_from_logits(
    logits: Any,
    *,
    positive_index: int = 1,
    binary_single_logit: Optional[bool] = None,
    binary_bce_from_two_logits: Optional[bool] = None,
    mode: Literal["auto", "bce_single", "ce_softmax", "bce_two_logit"] = "auto",
) -> torch.Tensor:
    """
    Map logits -> P(positive). Exactly one sigmoid in BCE single-logit mode.
    Never constructs [-x, x]. Softmax only for true multi-logit inputs.
    """
    t = to_tensor(logits).float()
    # normalize shape references to last dim
    last_dim = t.shape[-1] if t.ndim > 0 else 1

    # Resolve mode if auto
    if mode == "auto":
        if binary_single_logit is True:
            mode = "bce_single"
        elif binary_bce_from_two_logits is True:
            mode = "bce_two_logit"
        else:
            mode = "bce_single" if (t.ndim == 1 or last_dim == 1) else "ce_softmax"

    # Apply chosen mapping
    if mode == "bce_single":
        if t.ndim >= 2 and last_dim == 1:
            t = t.squeeze(-1)  # [B] or [B,1] → [B]
        # exactly one sigmoid; flip sign if positive_index==0
        return torch.sigmoid(t if int(positive_index) == 1 else -t)

    if mode == "bce_two_logit":
        # Expect [B,2] (or at least a channel dimension)
        if t.ndim == 1:  # already positive logit
            return torch.sigmoid(t)
        return torch.sigmoid(t[..., int(positive_index)])

    if mode == "ce_softmax":
        if last_dim < 2:
            raise ValueError(f"ce_softmax requires C>=2, got shape {tuple(t.shape)}")
        if not (0 <= int(positive_index) < last_dim):
            raise ValueError(f"positive_index={positive_index} out of range for C={last_dim}")
        probs = torch.softmax(t, dim=-1)
        return probs[..., int(positive_index)]

    raise ValueError(
        f"positive_score_from_logits: unsupported shape {tuple(t.shape)} for mode='{mode}'"
    )


@dataclass(frozen=True)
class PosProbCfg:
    binary_single_logit: bool = True
    binary_bce_from_two_logits: bool = False
    positive_index: int = 1
    cls_keys: tuple[str, ...] = ("cls_logits", "logits", "y_pred", "pred", "cls")
    label_keys: tuple[str, ...] = ("label", "y")

    def logits_to_pos_prob(self, logits: torch.Tensor) -> torch.Tensor:
        # Delegate to the canonical helper to avoid divergence
        return positive_score_from_logits(
            logits,
            positive_index=self.positive_index,
            binary_single_logit=self.binary_single_logit,
            binary_bce_from_two_logits=self.binary_bce_from_two_logits,
            mode="auto",
        )

    # extraction helpers
    def _first_present_key(self, mapping: Mapping, keys: Iterable[str]) -> str | None:
        for k in keys:
            if k in mapping:
                return k
        return None

    def pick_logits(self, output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, Mapping):
            k = self._first_present_key(output, self.cls_keys)
            if k is not None:
                return output[k]
        raise KeyError(f"[posprob] Could not find classification logits in keys={list(getattr(output, 'keys', lambda: [])())}")

    def pick_labels(self, target: Any) -> torch.Tensor:
        y = target
        if isinstance(target, Mapping):
            k = self._first_present_key(target, self.label_keys)
            if k is not None:
                y = target[k]
            else:
                y = None
        if y is None:
            raise KeyError("[posprob] Could not find labels in target")
        return torch.as_tensor(y).long().view(-1)

    # public “one-liners” every caller can use
    def scores(self, output: Any) -> torch.Tensor:
        """Return P(positive) in shape [B] regardless of model head shape."""
        logits = self.pick_logits(output)
        return self.logits_to_pos_prob(logits)

    def probs_and_labels(self, output: Any, target: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (P(positive)[B], labels[B])."""
        return self.scores(output), self.pick_labels(target)

    # pass the instance itself anywhere a callable is expected:
    __call__ = scores
