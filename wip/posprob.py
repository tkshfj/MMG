# posprob.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Tuple, Iterable
import torch


@dataclass(frozen=True)
class PosProbCfg:
    binary_single_logit: bool = True
    positive_index: int = 1
    cls_keys: tuple[str, ...] = ("cls_logits", "logits", "y_pred", "pred", "cls")
    label_keys: tuple[str, ...] = ("label", "y")

    # core logit -> prob transform
    def logits_to_pos_prob(self, logits: torch.Tensor) -> torch.Tensor:
        x = logits.float()
        if self.binary_single_logit:
            # Expect [B] or [B,1] → [B]
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.ndim != 1:
                raise ValueError(
                    f"binary_single_logit=True but logits shape is {tuple(x.shape)}; expected [B] or [B,1]"
                )
            return torch.sigmoid(x)

        # Multiclass: [B,C] → take positive_index column
        if x.ndim != 2:
            raise ValueError(f"multiclass logits must be [B,C], got {tuple(x.shape)}")
        if not (0 <= self.positive_index < x.shape[1]):
            raise ValueError(f"positive_index={self.positive_index} out of range for C={x.shape[1]}")
        return torch.softmax(x, dim=1)[:, self.positive_index]

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

    # nice alias so we can pass the instance itself anywhere a callable is expected:
    __call__ = scores

# # posprob.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Any, Iterable, Mapping, Tuple
# import torch

# @dataclass(frozen=True)
# class PosProbCfg:
#     binary_single_logit: bool = True
#     positive_index: int = 1
#     cls_keys: tuple[str, ...] = ("cls_logits", "logits", "y_pred", "pred", "cls")

#     def logits_to_pos_prob(self, logits: torch.Tensor) -> torch.Tensor:
#         x = logits.float()
#         if self.binary_single_logit:
#             # Expect [B] or [B,1] → [B]
#             if x.ndim == 2 and x.shape[1] == 1:
#                 x = x.squeeze(1)
#             elif x.ndim == 1:
#                 pass
#             else:
#                 raise ValueError(
#                     f"binary_single_logit=True but logits shape is {tuple(x.shape)}; expected [B] or [B,1]"
#                 )
#             return torch.sigmoid(x)
#         # Multiclass: [B,C] → take positive_index column
#         if x.ndim != 2:
#             raise ValueError(f"multiclass logits must be [B,C], got {tuple(x.shape)}")
#         if not (0 <= self.positive_index < x.shape[1]):
#             raise ValueError(f"positive_index={self.positive_index} out of range for C={x.shape[1]}")
#         return torch.softmax(x, dim=1)[:, self.positive_index]

#     def pick_logits(self, output: Any) -> torch.Tensor:
#         if torch.is_tensor(output):
#             return output
#         if isinstance(output, Mapping):
#             for k in self.cls_keys:
#                 if k in output:
#                     return output[k]
#         raise KeyError(f"[posprob] Could not find classification logits in {list(getattr(output, 'keys', lambda: [])())}")

#     def pick_labels(self, target: Any) -> torch.Tensor:
#         y = target
#         if isinstance(target, Mapping):
#             y = target.get("label", target.get("y", None))
#         if y is None:
#             raise KeyError("[posprob] Could not find labels in target")
#         y = torch.as_tensor(y).long().view(-1)
#         return y

#     def output_to_probs_and_labels(self, output: Any, target: Any) -> Tuple[torch.Tensor, torch.Tensor]:
#         logits = self.pick_logits(output)
#         probs = self.logits_to_pos_prob(logits)
#         y = self.pick_labels(target)
#         return probs, y
