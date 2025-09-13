# model_base.py
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from protocols import ModelRegistryProtocol
from metrics_utils import make_metrics, make_default_cls_output_transforms


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
        if logits.ndim <= 2 and (logits.shape[-1] == 1 or logits.ndim == 1):
            logits = logits.flatten()
            y = targets.float().flatten()
            # BCE per-sample
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            p = torch.sigmoid(logits)
            pt = torch.where(y > 0.5, p, 1 - p)  # y ∈ {0,1}
            # pt = p * y + (1 - p) * (1 - y)
            focal = (1.0 - pt).pow(self.gamma) * bce
            # optional class weights [w_neg, w_pos]
            if self.weight is not None and self.weight.numel() == 2:
                w = torch.where(y > 0.5, self.weight[1], self.weight[0])
                focal = focal * w
            return focal.mean() if self.reduction == "mean" else focal.sum()

        # Multi-class case: logits [B,C], targets [B] (long)
        else:  # multi-class
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
        # return self.config if config is None else config
        # defensive: if someone created an attribute `_cfg`, don't call it
        return getattr(self, "config", None) if config is None else config

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

    # Many registries expect these; provide safe defaults so subclasses are not abstract.
    @classmethod
    def build_model(cls, cfg=None, **kwargs):
        """
        Default class-level builder: instantiate with cfg/kwargs.
        Registry code already knows how to pass cfg/kwargs, so this is a safe no-op default.
        """
        try:
            return cls(cfg, **kwargs)  # e.g., SimpleCNNModel(cfg, **kwargs)
        except TypeError:
            try:
                return cls(**(kwargs or {}))
            except TypeError:
                return cls()

    def get_handler_kwargs(self, *args, **kwargs):
        """
        Default handler kwarg provider for event/logging hooks.
        Subclasses may override to pass model-specific arguments to handlers.
        """
        return {}

    def _resolve_loss_for(self, task: str, cfg: dict):
        counts = getattr(self, "class_counts", None) or self._cfg_get(cfg, "class_counts", None)
        # Prefer the new signature
        try:
            return self.get_loss_fn(task=task, cfg=cfg, class_counts=counts)
        except TypeError:
            pass
        # Older signature: (task, cfg)
        try:
            return self.get_loss_fn(task, cfg)  # type: ignore[misc]
        except TypeError:
            pass
        # Oldest signature: no args (must infer from cfg)
        try:
            return self.get_loss_fn()  # type: ignore[misc]
        except TypeError:
            return None

    def get_metrics(
        self,
        config: Optional[dict] = None,
        *,
        task: Optional[str] = None,
        has_cls: Optional[bool] = None,
        has_seg: Optional[bool] = None,
    ) -> Dict[str, Any]:
        cfg = self._cfg(config)

        # infer task flags
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

        # classes
        cls_classes = int(self.get_num_classes(cfg))
        seg_classes = self._get("seg_out_channels", None, cfg)
        if seg_classes is None:
            seg_classes = self._cfg_get(cfg, "out_channels", None)
        if seg_classes is None:
            seg_classes = cls_classes
        seg_classes = max(2, int(seg_classes))

        # decision settings
        cls_decision = str(cfg.get("cls_decision", "threshold")).lower()  # "threshold" | "argmax"
        cls_threshold = cfg.get("cls_threshold", 0.5)  # float or callable allowed downstream
        positive_index = int(cfg.get("positive_index", 1))
        seg_thr = float(cfg.get("seg_threshold", 0.5))

        # Classification OTs
        cls_ots = make_default_cls_output_transforms(
            decision=cls_decision,
            threshold=cls_threshold,
            positive_index=positive_index,
            binary_single_logit=bool(cfg.get("binary_single_logit", (cls_classes == 2))),
            binary_bce_from_two_logits=bool(cfg.get("binary_bce_from_two_logits", False)),
            mode="auto",
        )
        # AUC uses base OT; no model hook needed
        auc_ot = cls_ots.base

        # optional combined loss for multitask
        loss_fn = self._resolve_loss_for("multitask", cfg) if multitask else None

        # Assemble with explicit OTs
        metrics = make_metrics(
            tasks=tasks,
            num_classes=cls_classes,
            seg_num_classes=seg_classes,
            loss_fn=loss_fn,
            # classification
            cls_ot=cls_ots.thresholded,  # Acc/Prec/Rec
            auc_ot=auc_ot,  # ROC_AUC (probability-based)
            # segmentation
            seg_cm_ot=seg_thr,
            # decisions (for internal CM construction, etc.)
            multitask=multitask,
            cls_decision=cls_decision,
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

        # explicit class_weights
        w_seq = _to_seq(explicit_w, float)
        if w_seq is not None:
            if len(w_seq) == 1 and C == 2:
                w = torch.tensor([1.0, float(w_seq[0])], dtype=torch.float)
            else:
                w = torch.as_tensor(w_seq, dtype=torch.float)

        # derive from class_counts
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

        # None (no weights)
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

    # Backward-compat: keep the old name but route to the robust extractor.
    def _get_cls_logits(self, pred: Dict[str, Any]) -> torch.Tensor:
        return self._extract_cls_logits(pred)

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
        pos_idx = int(self._get("positive_index", 1))
        if isinstance(cnts, (list, tuple)) and len(cnts) >= 2 and cnts[pos_idx] > 0:
            n_pos = float(cnts[pos_idx])
            n_neg = float(sum(cnts)) - n_pos
            return torch.tensor([n_neg / max(1.0, n_pos)], device=device, dtype=torch.float32)
        return None

    def _extract_cls_logits(self, y_pred: Any) -> torch.Tensor:
        """
        Robustly extract classification logits from model output.
        Accepts: tensor, dict, or (list|tuple) of tensors/dicts.
        """
        import torch
        if isinstance(y_pred, torch.Tensor):
            return y_pred
        if isinstance(y_pred, dict):
            for k in ("cls_logits", "cls_out", "class_logits", "logits", "y_pred"):
                v = y_pred.get(k)
                if isinstance(v, torch.Tensor):
                    return v
        if isinstance(y_pred, (list, tuple)):
            for e in y_pred:
                try:
                    return self._extract_cls_logits(e)
                except Exception:
                    continue
        raise KeyError("No classification logits found under ('cls_logits','cls_out','class_logits','logits','y_pred') or tensor-like.")

    def init_bias_from_priors(self, head: nn.Module, class_counts: Optional[List[int]], binary_single_logit: bool, positive_index: int = 1):
        if not class_counts or not hasattr(head, "bias") or head.bias is None:
            return
        counts = torch.tensor(class_counts, dtype=torch.float)
        priors = counts / counts.sum()
        with torch.no_grad():
            if binary_single_logit:
                p_pos = float(priors[positive_index].clamp_(1e-8, 1 - 1e-8))
                head.bias.fill_(torch.tensor(p_pos).log() - torch.tensor(1.0 - p_pos).log())
            else:
                head.bias.copy_(priors.clamp_(1e-8, 1 - 1e-8).log())

    def get_loss_fn(
        self,
        task: Optional[str] = None,
        cfg: Optional[dict] = None,
        class_counts: Optional[List[int]] = None,
    ):
        """
        Return a loss_fn(pred_dict, target) for the resolved task.
        - classification: CE (multi-class) or BCEWithLogits (binary, 1-logit or 2->1 collapsed)
        - segmentation:   CE over logits [B,C|1,H,W] (binary auto-expanded) and mask [B,H,W]
        - multitask:      alpha*cls + beta*seg (alpha/beta from cfg, default 1.0)
        """
        cfg = self._cfg(cfg)
        # Resolve task from arg or config (keeps backward-compat with self.get_loss_fn())
        t = (task or self._cfg_get(cfg, "task", "classification")).lower()

        # Core config knobs
        num_classes = int(self._cfg_get(cfg, "num_classes", self.get_num_classes(cfg)))
        pos_idx = int(self._cfg_get(cfg, "positive_index", 1))
        use_single = bool(self._cfg_get(cfg, "binary_single_logit", (num_classes == 2 and self._cfg_get(cfg, "use_single_logit", True))))
        loss_name = str(self._cfg_get(cfg, "cls_loss", "ce")).lower()
        two_to_one = bool(self._cfg_get(cfg, "binary_bce_from_two_logits", True))

        # Resolve class_counts from param, self, or cfg
        counts = (class_counts or self._get("class_counts", None) or self._cfg_get(cfg, "class_counts", None))

        # Prepare imbalance handling
        class_w = None  # for CE/focal
        pos_w = None  # for BCE

        if isinstance(counts, (list, tuple)):
            cnt = torch.tensor(counts, dtype=torch.float)
            if use_single:
                pos = cnt[pos_idx].clamp_min(1e-8)
                neg = (cnt.sum() - pos).clamp_min(1e-8)
                pos_w = (neg / pos).clamp_max(1e6)
            else:
                inv = 1.0 / cnt.clamp_min(1e-8)
                class_w = (inv / inv.mean()).float()  # length C

        # Prebuild criteria (moved out of inner loop)
        ig = self._cfg_get(cfg, "ignore_index", None)
        ls = self._cfg_get(cfg, "label_smoothing", None)
        ce_kwargs: Dict[str, Any] = {}
        if class_w is not None:
            ce_kwargs["weight"] = class_w  # moved to device at call
        if ls is not None:
            ce_kwargs["label_smoothing"] = float(ls)
        if ig is not None:
            ce_kwargs["ignore_index"] = int(ig)
        focal_gamma = float(self._cfg_get(cfg, "focal_gamma", 2.0))

        # classification loss
        def cls_loss(pred: Dict[str, Any], tgt: Any) -> torch.Tensor:
            logits = self._extract_cls_logits(pred)
            y_idx = self._get_cls_target(tgt)  # [B] long indices

            if use_single:
                # Accept [B], [B,1], or collapse [B,2] -> [B] via pos-neg
                if logits.ndim == 2 and logits.size(-1) == 2 and two_to_one:
                    pos, neg = self._pos_neg_indices()
                    logits = logits[:, pos] - logits[:, neg]
                if logits.ndim == 2 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                # Map indices to binary by positive_index
                y_bin = (y_idx == pos_idx).float().view_as(logits)  # y_idx.float().view_as(logits)
                pw = (None if pos_w is None else pos_w.to(device=logits.device, dtype=logits.dtype))
                return F.binary_cross_entropy_with_logits(
                    logits.float(),
                    y_bin,
                    pos_weight=pw
                )

            # Multi-class CE or focal
            if loss_name == "focal":
                crit = FocalLoss(
                    gamma=focal_gamma,
                    weight=(None if class_w is None else class_w.to(logits)),
                    reduction="mean",
                )
                return crit(logits, y_idx.long())
            # CE with optional weight/ignore_index/label_smoothing
            if ce_kwargs:
                k = dict(ce_kwargs)
                if "weight" in k and k["weight"] is not None:
                    k["weight"] = k["weight"].to(logits)
                return F.cross_entropy(logits, y_idx.long(), **k)
            return F.cross_entropy(logits, y_idx.long())

        # segmentation loss
        def seg_loss(pred: Dict[str, Any], tgt: Any) -> torch.Tensor:
            seg = self._seg_logits_for_ce(self._get_seg_logits(pred))  # -> [B,C>=2,H,W]
            y = self._get_seg_target(tgt).long()                       # -> [B,H,W]
            if ig is not None:
                return F.cross_entropy(seg, y, ignore_index=int(ig))
            return F.cross_entropy(seg, y)

        if t in {"classification", "cls"}:
            return cls_loss
        if t in {"segmentation", "seg"}:
            return seg_loss

        # multitask: alpha*cls + beta*seg
        alpha = float(self._cfg_get(cfg, "alpha", self._cfg_get(cfg, "cls_weight", 1.0)))
        beta = float(self._cfg_get(cfg, "beta", self._cfg_get(cfg, "seg_weight", 1.0)))

        def multitask_loss(pred: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
            return alpha * cls_loss(pred, tgt) + beta * seg_loss(pred, tgt)
        return multitask_loss
