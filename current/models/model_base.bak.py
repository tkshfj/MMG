# model_base.py
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from protocols import ModelRegistryProtocol
from metrics_utils import (
    cls_output_transform,
    auc_output_transform,
    seg_output_transform,
    cls_confmat_output_transform,
    seg_confmat_output_transform,
    make_metrics,
)


# class-balanced weighting (Effective Number of Samples)
def _effective_number_weights(counts, beta: float = 0.999):
    """
    counts: 1D iterable of class counts
    returns mean-normalized weights proportional to (1 - beta) / (1 - beta^n_c)
    """
    import torch
    cnt = torch.as_tensor(counts, dtype=torch.float).clamp_min(1.0)
    beta = float(beta)
    if not (0.0 < beta < 1.0):
        beta = 0.999
    eff_num = (1.0 - torch.pow(beta, cnt)).clamp_min(1e-12)
    w = (1.0 - beta) / eff_num
    # normalize to mean = 1 for stability
    return w * (w.numel() / w.sum())


# Focal Loss that supports binary (BCE) and multi-class (CE)
class FocalLoss(nn.Module):
    """
    Focal loss with logits. Supports:
      - Binary: logits [B] or [B,1], targets [B] in {0,1}
      - Multi-class: logits [B,C], targets [B] in {0..C-1}

    Args:
      gamma: focusing parameter (typical 2.0)
      weight: per-class weights (C,) for multi-class; for binary, interprets as [w_neg, w_pos]
      reduction: 'mean' | 'sum'
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", None if weight is None else weight.clone().detach())
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        # Binary case: logits [B] or [B,1]
        if logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 1):
            logits = logits.flatten()
            y = targets.float().flatten()
            # BCE per-sample
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            p = torch.sigmoid(logits)
            pt = torch.where(y > 0.5, p, 1 - p)  # p_t
            focal = (1.0 - pt).pow(self.gamma) * bce
            # optional class weights [w_neg, w_pos]
            if self.weight is not None and self.weight.numel() == 2:
                w = torch.where(y > 0.5, self.weight[1], self.weight[0])
                focal = focal * w
            return focal.mean() if self.reduction == "mean" else focal.sum()

        # Multi-class case: logits [B,C], targets [B] (long)
        if logits.ndim == 2:
            y = targets.long().view(-1)
            # CE per-sample without reduction
            ce = F.cross_entropy(logits, y, weight=None, reduction="none")
            p = torch.softmax(logits, dim=-1)
            pt = p.gather(dim=1, index=y.view(-1, 1)).squeeze(1)
            focal = (1.0 - pt).pow(self.gamma) * ce
            # optional per-class weights
            if self.weight is not None and self.weight.numel() == logits.size(1):
                w = self.weight.to(focal)
                focal = focal * w.gather(dim=0, index=y)
            return focal.mean() if self.reduction == "mean" else focal.sum()

        raise ValueError(f"FocalLoss: unexpected logits shape {tuple(logits.shape)}")


class BaseModel(ModelRegistryProtocol):
    # config helpers
    def _cfg(self, config: Any | None = None) -> Any:
        return self.config if config is None else config

    @staticmethod
    def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
        from collections.abc import Mapping
        if cfg is None:
            return default
        if isinstance(cfg, Mapping):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    @staticmethod
    def _cfg_set(cfg: Any, key: str, value: Any) -> Any:
        from collections.abc import MutableMapping
        if cfg is None:
            raise ValueError("cfg is None; cannot set a key on None")
        if isinstance(cfg, MutableMapping):
            cfg[key] = value
        else:
            setattr(cfg, key, value)
        return cfg

    def _get(self, key: str, default: Any = None, config: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self._cfg_get(self._cfg(config), key, default)

    # basic API
    def get_supported_tasks(self) -> tuple[str, ...]:
        # Default: wrapper supports all; active task is decided by config/runner
        return ("classification", "segmentation", "multitask")

    def get_num_classes(self, config=None) -> int:
        return int(self._get("num_classes", 2, config))

    # Output transforms
    def get_output_transform(self):
        return self.get_cls_output_transform()

    def get_cls_output_transform(self):
        return cls_output_transform

    def get_cls_confmat_output_transform(self):
        return cls_confmat_output_transform

    def get_auc_output_transform(self):
        return auc_output_transform

    def get_seg_output_transform(self):
        return seg_output_transform

    def get_seg_confmat_output_transform(self):
        return seg_confmat_output_transform

    # task-aware metrics factory
    def get_metrics(
        self,
        config: Optional[dict] = None,
        *,
        task: Optional[str] = None,
        has_cls: Optional[bool] = None,
        has_seg: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Build a task-aware metrics dict for Ignite:
        - classification only: cls metrics (+ loss CE)
        - segmentation only:   seg metrics (+ loss CE over indices)
        - multitask:           both sets (+ unified loss with combined loss_fn)
        Uses explicit decision rule for Acc/Prec/Recall/CM; AUC stays threshold-free.
        """
        cfg = self._cfg(config)

        # infer task flags (explicit args override inferred)
        t = (task if task is not None else self._cfg_get(cfg, "task", "multitask")).lower()
        inferred_has_cls = t in ("classification", "multitask")
        inferred_has_seg = t in ("segmentation", "multitask")
        has_cls = inferred_has_cls if has_cls is None else bool(has_cls)
        has_seg = inferred_has_seg if has_seg is None else bool(has_seg)
        multitask = (t == "multitask") and has_cls and has_seg

        if not (has_cls or has_seg):
            raise ValueError("get_metrics: at least one of classification/segmentation must be enabled.")

        tasks: list[str] = []
        if has_cls:
            tasks.append("classification")
        if has_seg:
            tasks.append("segmentation")

        # class counts
        cls_classes = int(self.get_num_classes(cfg))
        seg_classes = self._get("seg_out_channels", None, cfg)
        if seg_classes is None:
            seg_classes = self._cfg_get(cfg, "out_channels", None)
        if seg_classes is None:
            seg_classes = cls_classes
        seg_classes = max(2, int(seg_classes))

        # optional transform factories (pass through directly if present)
        cls_ot_factory = getattr(self, "get_cls_output_transform", None)
        auc_ot_factory = getattr(self, "get_auc_output_transform", None)
        seg_cm_ot_factory = getattr(self, "get_seg_confmat_output_transform", None)

        # optional combined loss for multitask
        loss_fn = None
        if multitask and hasattr(self, "get_loss_fn"):
            try:
                loss_fn = self.get_loss_fn()  # combined
            except TypeError:
                try:
                    loss_fn = self.get_loss_fn("multitask", cfg)
                except TypeError:
                    loss_fn = None

        # decision settings (config-driven)
        cls_decision = str(cfg.get("cls_decision", "threshold")).lower()   # "argmax" | "threshold"
        cls_threshold = float(cfg.get("cls_threshold", 0.5))
        positive_index = int(cfg.get("positive_index", 1))

        # bind seg_threshold to ConfusionMatrix output_transform
        # Prefer a config-wired partial of seg_confmat_output_transform so the chosen
        # threshold (e.g., 0.35) actually controls discretization for binary seg.
        import functools
        import inspect
        seg_thr = float(cfg.get("seg_threshold", 0.5))
        # Try to reuse a model-provided factory if it supports "threshold", otherwise fall back.
        if callable(seg_cm_ot_factory):
            try:
                params = inspect.signature(seg_cm_ot_factory).parameters
                if "threshold" in params:
                    seg_cm_ot = functools.partial(seg_cm_ot_factory, threshold=seg_thr)
                else:
                    seg_cm_ot = functools.partial(seg_confmat_output_transform, threshold=seg_thr)
            except (TypeError, ValueError):
                seg_cm_ot = functools.partial(seg_confmat_output_transform, threshold=seg_thr)
        else:
            seg_cm_ot = functools.partial(seg_confmat_output_transform, threshold=seg_thr)

        # assemble metrics
        metrics = make_metrics(
            tasks=tasks,
            num_classes=cls_classes,
            seg_num_classes=seg_classes,
            loss_fn=loss_fn,
            cls_ot=cls_ot_factory,  # None | factory | callable(output)->(y_pred, y)
            auc_ot=auc_ot_factory,
            seg_cm_ot=seg_cm_ot,
            multitask=multitask,
            cls_decision=cls_decision,  # explicit decision for Acc/Prec/Recall/CM
            cls_threshold=cls_threshold,
            positive_index=positive_index,
        )

        if not metrics:
            print("[WARN] get_metrics returned empty dict; check task/config.")
        return metrics

    # class weights & CE builder (classification)
    def _get_class_weights(self, device=None, dtype=None):
        """
        Returns a 1D tensor of per-class weights.

        Priority:
        1) cfg['class_weights'] if provided
        - if scalar and num_classes==2, interpreted as [1.0, pos_weight]
        2) If cfg['class_counts'] is provided:
            - cfg['class_balance']=="effective" -> effective-number weights (mean-normalized)
            - else (default "inverse") -> inverse-frequency weights (mean-normalized)
        3) None

        Extra config:
        - class_balance: "inverse" (default) | "effective" | "none"
        - cb_beta: beta for effective-number (default 0.999)
        """
        import ast
        import numpy as np
        import torch

        C = int(self._get("num_classes", 2))
        explicit_w = self._get("class_weights", None)
        counts = self._get("class_counts", None)
        balance = str(self._get("class_balance", "inverse")).lower()
        cb_beta = float(self._get("cb_beta", 0.999))

        def _to_seq(x, cast=float):
            if x is None:
                return None
            if isinstance(x, str):
                try:
                    x = ast.literal_eval(x)
                except Exception:
                    return None
            if isinstance(x, (list, tuple, np.ndarray)):
                try:
                    return [cast(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
                except Exception:
                    return None
            if isinstance(x, (int, float)):
                return [cast(x)]
            return None

        w = None

        # 1) explicit class_weights
        w_seq = _to_seq(explicit_w, float)
        if w_seq is not None:
            if len(w_seq) == 1 and C == 2:
                w = torch.tensor([1.0, float(w_seq[0])], dtype=torch.float)
            else:
                w = torch.as_tensor(w_seq, dtype=torch.float)

        # 2) derive from class_counts
        elif (cnt_seq := _to_seq(counts, float)) is not None:
            cnt = cnt_seq
            if balance == "effective":
                w = _effective_number_weights(cnt, beta=cb_beta).to(dtype=torch.float)
            elif balance == "none":
                w = None
            else:
                # inverse-frequency, mean-normalized
                cnt_t = torch.as_tensor(cnt, dtype=torch.float).clamp_min(1.0)
                inv = 1.0 / cnt_t
                w = inv * (inv.numel() / inv.sum())

        # 3) None (no weights)

        # validate length
        if w is not None and w.numel() != C:
            raise ValueError(f"class weights length {w.numel()} != num_classes {C}")

        if w is not None and (device is not None or dtype is not None):
            w = w.to(device=device if device is not None else w.device, dtype=dtype if dtype is not None else w.dtype)
        return w

    # task-aware loss plumbing
    @staticmethod
    def _pick(d, *keys):
        if isinstance(d, dict):
            for k in keys:
                v = d.get(k, None)
                if v is not None:
                    return v
        return None

    def _pos_neg_indices(self) -> tuple[int, int]:
        """Return (pos, neg) indices based on config. Defaults to pos=1, neg=0."""
        pos = int(self._get("positive_index", 1))
        return pos, 1 - pos

    def _get_cls_logits(self, pred: Dict[str, Any]) -> torch.Tensor:
        import torch
        if isinstance(pred, torch.Tensor):
            return pred
        if isinstance(pred, (list, tuple)):
            for e in pred:
                if isinstance(e, torch.Tensor):
                    return e
                if isinstance(e, dict):
                    for k in ("cls_out", "class_logits", "logits", "y_pred"):
                        v = e.get(k, None)
                        if isinstance(v, torch.Tensor):
                            return v
        if isinstance(pred, dict):
            for k in ("cls_out", "class_logits", "logits", "y_pred"):
                v = pred.get(k, None)
                if isinstance(v, torch.Tensor):
                    return v
        raise KeyError("No classification logits found under ('cls_out','class_logits','logits','y_pred') or tensor-like.")

    def _get_seg_logits(self, pred: Dict[str, Any]) -> torch.Tensor:
        import torch
        if isinstance(pred, torch.Tensor):
            return pred  # already [B, C|1, H, W]
        if isinstance(pred, dict):
            # for k in ("seg_out", "mask_logits", "seg", "segmentation", "logits"):
            for k in ("seg_out", "seg_logits", "mask_logits", "seg", "segmentation", "logits"):
                v = pred.get(k)
                if isinstance(v, torch.Tensor):
                    return v
        if isinstance(pred, (list, tuple)):
            for e in pred:
                v = self._get_seg_logits(e)
                if isinstance(v, torch.Tensor):
                    return v
        raise TypeError(f"_get_seg_logits: cannot find seg logits in {type(pred)}")
        # return pred["seg_out"]  # [B, C|1, H, W]

    def _get_cls_target(self, tgt: Any) -> torch.Tensor:
        # accept dict or tensor
        y = self._pick(tgt, "label", "y") if isinstance(tgt, dict) else tgt
        if hasattr(y, "as_tensor"):
            y = y.as_tensor()
        y = torch.as_tensor(y)
        # normalize to indices [B]
        if y.ndim >= 2 and y.shape[-1] > 1:
            y = y.argmax(dim=-1)
        if y.ndim >= 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if y.ndim == 0:
            y = y.view(1)
        return y.long().view(-1)

    def _get_seg_target(self, tgt: Any) -> torch.Tensor:
        # accept dict or tensor
        y = self._pick(tgt, "mask", "seg") if isinstance(tgt, dict) else tgt
        if hasattr(y, "as_tensor"):
            y = y.as_tensor()
        y = torch.as_tensor(y)
        # normalize to index mask [B,H,W]
        if y.ndim == 4:
            y = y[:, 0] if y.shape[1] == 1 else y.argmax(dim=1)
        elif y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)
        elif y.ndim == 2:
            # Single image [H,W] -> add batch dim for CE ([1,H,W]); dataloaders normally provide [B,H,W]
            y = y.unsqueeze(0)
        return y.long()

    def _seg_logits_for_ce(self, logits: torch.Tensor) -> torch.Tensor:
        # Ensure CE-compatible logits: [B,C,H,W] with C>=2
        if logits.ndim == 3:           # [B,H,W] -> [B,1,H,W]
            logits = logits.unsqueeze(1)
        if logits.shape[1] == 1:       # binary -> 2-channel logits via [-x, x]
            x = logits[:, 0:1]
            logits = torch.cat([-x, x], dim=1)
        return logits

    def _compute_pos_weight(self, device=None) -> Optional[torch.Tensor]:
        """
        Returns a tensor([pos_weight]) for BCEWithLogits, or None.
        Priority:
        1) cfg['pos_weight'] scalar
        2) Derive from per-class CE weights if binary: w_pos / w_neg
        3) Derive from class_counts if provided: (n_neg / n_pos)
        """
        # explicit scalar
        pw = self._get("pos_weight", None)
        if isinstance(pw, (int, float)):
            return torch.tensor([float(pw)], device=device, dtype=torch.float32)
        # derive from CE weights if available
        try:
            w = self._get_class_weights(device=device, dtype=torch.float32)
            if w is not None and w.numel() == 2 and w[0].item() > 0:
                return torch.tensor([w[1].item() / w[0].item()], device=device, dtype=torch.float32)
        except Exception:
            pass
        # derive from class_counts if available
        cnts = self._get("class_counts", None)
        if isinstance(cnts, (list, tuple)) and len(cnts) == 2 and cnts[1] > 0:
            n_neg, n_pos = float(cnts[0]), float(cnts[1])
            return torch.tensor([n_neg / max(1.0, n_pos)], device=device, dtype=torch.float32)
        return None

    def _build_cls_loss(self) -> nn.Module:
        """
        Classification criterion (config keys):
        - cls_loss: "auto"|"ce"|"bce"|"focal"  (default: "auto")
        - label_smoothing: float (for CE)
        - ignore_index: int (for CE)
        - focal_gamma: float (default 2.0)
        - class_weights / class_counts (+ class_balance, cb_beta): see _get_class_weights
        - pos_weight: float (BCE only; overrides derived ratio if given)
        """
        import torch
        import torch.nn as nn

        device = self._get("device", None)
        mode = str(self._get("cls_loss", "auto")).lower()
        ls = self._get("label_smoothing", None)
        ig = self._get("ignore_index", None)

        # shared per-class weights (used by CE or focal)
        weight = None
        try:
            weight = self._get_class_weights(device=device, dtype=torch.float32)
        except Exception:
            weight = None

        # FOCAL: explicit request
        if mode == "focal":
            gamma = float(self._get("focal_gamma", 2.0))
            return FocalLoss(gamma=gamma, weight=weight, reduction="mean")

        # BCE path (explicit)
        if mode == "bce":
            pos_w_tensor = self._compute_pos_weight(device=device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

        # CE path (default / 'auto' / 'ce')
        ce_kwargs = {}
        if weight is not None:
            ce_kwargs["weight"] = weight
        if ls is not None:
            ce_kwargs["label_smoothing"] = float(ls)
        if ig is not None:
            ce_kwargs["ignore_index"] = int(ig)

        return nn.CrossEntropyLoss(**ce_kwargs)

    def _build_seg_loss(self) -> nn.Module:
        # Default seg loss: CE over index masks (compatible with ConfusionMatrix/DiceCE flows)
        ig = self._get("ignore_index", None)
        return nn.CrossEntropyLoss(ignore_index=int(ig) if ig is not None else -100)

    def get_loss_fn(self, task: str, cfg: Optional[dict] = None):
        """
        Return a loss_fn(pred_dict, target) for the given task:
        - classification: CE (multi-class) or BCEWithLogits (binary, 1-logit)
        - segmentation:   CE over logits [B,C|1,H,W] (binary auto-expanded) and mask [B,H,W]
        - multitask:      alpha*cls + beta*seg (alpha/beta from cfg or default 1.0)
        """
        t = (task or "classification").lower()
        device = (cfg or {}).get("device", None)

        # Base criteria
        ce_cls = self._build_cls_loss()
        ce_seg = self._build_seg_loss()

        # Build a BCEWithLogits alternative for binary heads, with consistent pos_weight
        bce_pos_weight = self._compute_pos_weight(device=device)
        bce_cls = torch.nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)

        # Optional override in config: cls_loss: auto|ce|bce
        cls_mode = str(self._get("cls_loss", "auto")).lower()

        def cls_loss(pred: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
            logits = self._get_cls_logits(pred)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits)
            y = self._get_cls_target(tgt)
            mode = cls_mode

            if mode == "auto":
                if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
                    mode = "bce"
                elif logits.dim() == 2 and logits.size(-1) == 2 and bool(self._get("binary_bce_from_two_logits", True)):
                    # Convert 2 logits -> single logit using configured positive_index
                    pos, neg = self._pos_neg_indices()
                    logits = logits[:, pos] - logits[:, neg]  # (B,)
                    mode = "bce"
                else:
                    mode = "ce"

            if mode == "bce":
                yf = y.float().view_as(logits)
                return bce_cls(logits, yf)
            return ce_cls(logits, y.long())

        def seg_loss(pred: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
            logits = self._seg_logits_for_ce(self._get_seg_logits(pred))  # ensure [B,C,H,W]
            y = self._get_seg_target(tgt).long()                           # [B,H,W]
            return ce_seg(logits, y)

        if t in {"classification", "cls"}:
            return cls_loss
        if t in {"segmentation", "seg"}:
            return seg_loss

        # multitask
        alpha = float(self._cfg_get(self._cfg(cfg), "cls_weight", self._cfg_get(self._cfg(cfg), "alpha", 1.0)))
        beta = float(self._cfg_get(self._cfg(cfg), "seg_weight", self._cfg_get(self._cfg(cfg), "beta", 1.0)))

        def _multitask_loss(pred: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
            return alpha * cls_loss(pred, tgt) + beta * seg_loss(pred, tgt)

        return _multitask_loss
