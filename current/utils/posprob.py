# utils/posprob.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Tuple
import torch

try:
    # Optional helper to consistently extract logits from arbitrary model outputs
    from metrics_utils import extract_cls_logits_from_any as _extract_logits  # noqa: F401
except Exception:
    _extract_logits = None  # type: ignore


def _as_tensor(x: Any) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def logits_to_pos_prob(
    logits: Any,
    *,
    scheme: Literal["bce1", "bce2", "ce"] = "bce1",
    positive_index: int = 1,
) -> torch.Tensor:
    """
    Map raw logits to positive-class probability (no thresholding).

    scheme:
      - "bce1": single-logit BCEWithLogits (shape [...], sigmoid)
      - "bce2": two-logit BCE style (shape [...,2], sigmoid on pos logit)
      - "ce":   CrossEntropy style (shape [...,C], softmax then take pos idx)
    """
    t = _as_tensor(logits)
    last = (t.ndim > 0 and t.shape[-1]) or 1

    if scheme == "bce1":
        # Allow [B], [B,1], or [...,1]
        return torch.sigmoid(t).reshape(t.shape[:-1]) if (t.ndim > 1 and last == 1) else torch.sigmoid(t)

    if scheme == "bce2":
        if last < 2:
            raise ValueError(f"'bce2' expects last dim >= 2, got {tuple(t.shape)}")
        return torch.sigmoid(t[..., int(positive_index)])

    if scheme == "ce":
        if last < 2:
            raise ValueError(f"'ce' expects last dim >= 2, got {tuple(t.shape)}")
        return torch.softmax(t, dim=-1)[..., int(positive_index)]

    raise ValueError(f"Unknown scheme='{scheme}'")


@dataclass
class PosProbCfg:
    """One SoT for logits→P(pos)."""
    positive_index: int = 1
    scheme: Literal["bce1", "bce2", "ce"] = "bce1"

    @staticmethod
    def from_cfg(cfg: dict) -> "PosProbCfg":
        head_logits = int(cfg.get("head_logits", 1))
        loss_name = str(cfg.get("cls_loss", "bce")).lower()
        legacy_two = bool(cfg.get("binary_bce_from_two_logits", False))

        if head_logits <= 1:
            scheme = "bce2" if legacy_two else "bce1"
        else:
            scheme = "ce"
            if "bce" in loss_name:
                print("[WARN] head_logits>=2 but cls_loss looks like BCE; proceeding with CE mapping.")
        pos_idx = int(cfg.get("positive_index", 1))
        return PosProbCfg(positive_index=pos_idx, scheme=scheme)

    @staticmethod
    def _pick_logits(output: Any) -> torch.Tensor:
        if _extract_logits is not None:
            try:
                t = _extract_logits(output)
                if torch.is_tensor(t):
                    return t
            except Exception:
                pass
        if isinstance(output, dict):
            for k in ("logits", "cls_logits", "class_logits", "y_pred", "cls_out"):
                v = output.get(k, None)
                if torch.is_tensor(v):
                    return v
        return _as_tensor(output)

    @staticmethod
    def _pick_labels(target: Any) -> torch.Tensor:
        cur = target
        while isinstance(cur, dict):
            for k in ("y", "label", "labels"):
                if k in cur and cur[k] is not None:
                    cur = cur[k]
                    break
            else:
                break
        return _as_tensor(cur).long().view(-1)

    def pick_logits(self, output: Any) -> torch.Tensor:
        return self._pick_logits(output)

    def pick_labels(self, target: Any) -> torch.Tensor:
        return self._pick_labels(target)

    def logits_to_pos_prob(self, logits: Any) -> torch.Tensor:
        return logits_to_pos_prob(logits, scheme=self.scheme, positive_index=self.positive_index)

    # Ignite-friendly
    def __call__(self, output: Any) -> torch.Tensor:
        return self.logits_to_pos_prob(self._pick_logits(output))

    def probs_and_labels(self, output: Any, target: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        return self(output), self._pick_labels(target)

    def attach_to_engine_state(self, engine) -> None:
        setattr(engine.state, "ppcfg", self)
        setattr(engine.state, "to_pos_prob", self.logits_to_pos_prob)

# # posprob.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Any, Literal, Tuple
# import torch

# # Prefer pulling raw logits under the 'logits' key; fall back sanely.
# # We keep the import optional to avoid circulars.
# try:
#     from metrics_utils import extract_cls_logits_from_any as _extract_logits  # noqa: F401
# except Exception:
#     _extract_logits = None  # type: ignore


# def _as_tensor(x: Any) -> torch.Tensor:
#     return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


# # Single pure function: logits -> P(pos)
# def logits_to_pos_prob(
#     logits: Any,
#     *,
#     scheme: Literal["bce1", "bce2", "ce"] = "bce1",
#     positive_index: int = 1,
# ) -> torch.Tensor:
#     """
#     Map raw logits to positive-class probability (no thresholding).

#     scheme:
#       - "bce1": single-logit BCE (shape [...], sigmoid)
#       - "bce2": two-logit BCE style (shape [...,2], sigmoid on pos logit)
#       - "ce":   CrossEntropy style (shape [...,C], softmax then pick pos idx)
#     """
#     t = _as_tensor(logits)
#     last = (t.ndim > 0 and t.shape[-1]) or 1

#     if scheme == "bce1":
#         # Allow [B] or [B,1] or [...,1]
#         return torch.sigmoid(t).reshape(t.shape[:-1]) if (t.ndim > 1 and last == 1) else torch.sigmoid(t)

#     if scheme == "bce2":
#         if last < 2:
#             raise ValueError(f"'bce2' expects last dim >= 2, got {tuple(t.shape)}")
#         return torch.sigmoid(t[..., int(positive_index)])

#     if scheme == "ce":
#         if last < 2:
#             raise ValueError(f"'ce' expects last dim >= 2, got {tuple(t.shape)}")
#         return torch.softmax(t, dim=-1)[..., int(positive_index)]

#     raise ValueError(f"Unknown scheme='{scheme}'")


# def prob_from_logits(
#     logits: torch.Tensor,
#     *,
#     head_logits: int,
#     num_classes: int,
#     positive_index: int = 1
# ) -> torch.Tensor:
#     """Returns a column vector [B,1] of P(y=positive)."""
#     if logits.ndim == 1:
#         logits = logits.unsqueeze(1)
#     if head_logits == 1:
#         # Single-logit = P(pos) by convention
#         return torch.sigmoid(logits)
#     # Multi-logit: P(pos) from softmax
#     if logits.size(1) != num_classes:
#         raise RuntimeError(f"Expected logits.size(1)==num_classes ({num_classes}), got {logits.size(1)}")
#     if not (0 <= positive_index < num_classes):
#         raise RuntimeError(f"positive_index {positive_index} out of range 0..{num_classes-1}")
#     probs = torch.softmax(logits, dim=1)
#     return probs[:, positive_index:positive_index + 1]


# @dataclass
# class PosProbCfg:
#     """
#     One source of truth:
#       - Decide the scheme once (from cfg or head_logits/loss)
#       - Always pick logits (not probs) from model outputs
#       - Convert with logits_to_pos_prob exactly once
#     """
#     positive_index: int = 1
#     scheme: Literal["bce1", "bce2", "ce"] = "bce1"

#     # Construction helpers
#     @staticmethod
#     def from_cfg(cfg: dict) -> "PosProbCfg":
#         """
#         Derive the scheme from config:
#           head_logits = 1  => 'bce1' (single-logit BCEWithLogits)
#           head_logits >= 2 => 'ce'   (CrossEntropy)
#         If a legacy toggle exists (binary_bce_from_two_logits=True), map to 'bce2'.
#         """
#         head_logits = int(cfg.get("head_logits", 1))
#         loss_name = str(cfg.get("cls_loss", "bce")).lower()
#         legacy_two = bool(cfg.get("binary_bce_from_two_logits", False))

#         if head_logits <= 1:
#             scheme = "bce2" if legacy_two else "bce1"
#         else:
#             # 2+ logits: prefer CE; warn if BCE was requested (incompatible in this repo)
#             scheme = "ce"
#             if "bce" in loss_name:
#                 # We do not raise here to keep sweeps alive; the model/loss factory should guard.
#                 print("[WARN] head_logits>=2 but cls_loss looks like BCE; proceeding with CE mapping in PosProbCfg.")

#         pos_idx = int(cfg.get("positive_index", 1))
#         return PosProbCfg(positive_index=pos_idx, scheme=scheme)

#     # IO helpers
#     @staticmethod
#     def _pick_logits(output: Any) -> torch.Tensor:
#         # Prefer the dedicated key via extractor
#         if _extract_logits is not None:
#             try:
#                 t = _extract_logits(output)
#                 if torch.is_tensor(t):
#                     return t
#             except Exception:
#                 pass
#         # Otherwise, prefer explicit 'logits' before any ambiguous keys
#         if isinstance(output, dict):
#             for k in ("logits", "cls_logits", "class_logits", "y_pred", "cls_out"):
#                 v = output.get(k, None)
#                 if torch.is_tensor(v):
#                     return v
#         # Last resort: coerce whatever came back
#         return _as_tensor(output)

#     @staticmethod
#     def _pick_labels(target: Any) -> torch.Tensor:
#         cur = target
#         while isinstance(cur, dict):
#             for k in ("y", "label", "labels"):
#                 if k in cur and cur[k] is not None:
#                     cur = cur[k]
#                     break
#             else:
#                 break
#         return _as_tensor(cur).long().view(-1)

#     # public instance method expected by metrics_utils
#     def pick_logits(self, output: Any) -> torch.Tensor:
#         """Public wrapper so metrics/OTs can ask the cfg to extract raw logits."""
#         return self._pick_logits(output)

#     def pick_labels(self, target: Any) -> torch.Tensor:
#         """Public wrapper so callers can fetch integer class labels [B]."""
#         return self._pick_labels(target)

#     # Core API
#     def logits_to_pos_prob(self, logits: Any) -> torch.Tensor:
#         return logits_to_pos_prob(logits, scheme=self.scheme, positive_index=self.positive_index)

#     # Ignite-friendly callable (accepts the engine output dict/tensor)
#     def __call__(self, output: Any) -> torch.Tensor:
#         return self.logits_to_pos_prob(self._pick_logits(output))

#     # Convenience: pair (probs, labels)
#     def probs_and_labels(self, output: Any, target: Any) -> Tuple[torch.Tensor, torch.Tensor]:
#         return self(output), self._pick_labels(target)

#     # Wiring shortcuts
#     def attach_to_engine_state(self, engine) -> None:
#         """Stash on engine.state so every metric/OT uses the same mapping."""
#         setattr(engine.state, "ppcfg", self)
#         setattr(engine.state, "to_pos_prob", self.logits_to_pos_prob)
