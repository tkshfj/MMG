# model_base.py
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from model_protocol import ModelRegistryProtocol
from metrics_utils import (
    cls_output_transform,
    auc_output_transform,
    seg_output_transform,
    cls_confmat_output_transform,
    seg_confmat_output_transform,
    make_metrics,
)


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

    # Output transforms (unchanged)
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
        config=None,
        *,
        task: Optional[str] = None,
        has_cls: Optional[bool] = None,
        has_seg: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Build a task-aware metrics dict for Ignite:
        - classification only: cls metrics (+ val_loss CE)
        - segmentation only:   seg metrics (+ val_loss CE over indices)
        - multitask:           both sets (+ unified val_loss with combined loss_fn)
        """
        cfg = self._cfg(config)
        t = (task or self._cfg_get(cfg, "task", "multitask")).lower()
        has_cls = t in ("classification", "multitask")
        has_seg = t in ("segmentation", "multitask")
        multitask = (t == "multitask")

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
        seg_classes = int(seg_classes)

        # model-provided transforms (optional)
        cls_ot = getattr(self, "get_cls_output_transform", lambda: None)()
        auc_ot = getattr(self, "get_auc_output_transform", lambda: None)()
        seg_cm_ot = getattr(self, "get_seg_confmat_output_transform", lambda: None)()

        # choose an appropriate loss for val_loss metrics
        loss_fn = None
        if multitask and hasattr(self, "get_loss_fn"):
            try:
                loss_fn = self.get_loss_fn()  # combined, if provided
            except TypeError:
                try:
                    loss_fn = self.get_loss_fn("multitask", cfg)
                except TypeError:
                    loss_fn = None

        metrics = make_metrics(
            tasks=tasks,
            num_classes=cls_classes,
            seg_num_classes=seg_classes,
            loss_fn=loss_fn,
            cls_ot=cls_ot,
            auc_ot=auc_ot,
            seg_cm_ot=seg_cm_ot,
            multitask=multitask,
        )
        if not metrics:
            print("[WARN] get_metrics returned empty dict; check task.")
        return metrics

    # class weights & CE builder (classification)
    def _get_class_weights(self, device=None, dtype=None):
        """
        Returns a 1D tensor of per-class weights (for CrossEntropyLoss `weight=`).
        Priority:
        1) cfg['class_weights'] if provided
            - if scalar and num_classes==2, interpreted as [1.0, pos_weight]
        2) cfg['class_counts'] -> inverse-frequency weights, normalized to mean=1
        3) None
        Accepts strings like "[2000, 600]" via literal_eval.
        """
        import ast
        import numpy as np
        import torch

        C = int(self._get("num_classes", 2))
        explicit_w = self._get("class_weights", None)
        counts = self._get("class_counts", None)

        def _to_seq(x, cast=float):
            if x is None:
                return None
            if isinstance(x, str):
                try:
                    x = ast.literal_eval(x)  # "[1,2]" -> [1,2]
                except Exception:
                    return None
            if isinstance(x, (list, tuple, np.ndarray)):
                try:
                    return [cast(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
                except Exception:
                    return None
            # single scalar
            if isinstance(x, (int, float)):
                return [cast(x)]
            return None

        w = None

        # explicit class_weights
        w_seq = _to_seq(explicit_w, float)
        if w_seq is not None:
            # allow scalar for binary: treat as weight for positive class
            if len(w_seq) == 1 and C == 2:
                w = torch.tensor([1.0, float(w_seq[0])], dtype=torch.float)
            else:
                w = torch.as_tensor(w_seq, dtype=torch.float)

        # derive from class_counts (inverse frequency, mean-normalized)
        elif (cnt_seq := _to_seq(counts, float)) is not None:
            cnt = torch.as_tensor(cnt_seq, dtype=torch.float).clamp_min(1.0)
            inv = 1.0 / cnt
            w = inv * (inv.numel() / inv.sum())
        # nothing provided
        else:
            w = None
        # validate length
        if w is not None and w.numel() != C:
            raise ValueError(f"class weights length {w.numel()} != num_classes {C}")
        # move/cast if requested
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

    def _get_cls_logits(self, pred: Dict[str, Any]) -> torch.Tensor:
        import torch
        # Tensor directly
        if isinstance(pred, torch.Tensor):
            return pred
        # Tuple/list: pick first tensor-like entry
        if isinstance(pred, (list, tuple)):
            for e in pred:
                if isinstance(e, torch.Tensor):
                    return e
                if isinstance(e, dict) and ("cls_out" in e or "class_logits" in e):
                    return e.get("cls_out", e.get("class_logits"))
        # Dict keys
        v = self._pick(pred, "cls_out", "class_logits")
        if isinstance(v, torch.Tensor):
            return v
        raise KeyError("No classification logits found (tensor/tuple) or under 'cls_out'/'class_logits'.")

    def _get_seg_logits(self, pred: Dict[str, Any]) -> torch.Tensor:
        import torch
        if isinstance(pred, torch.Tensor):
            return pred  # already [B, C|1, H, W]
        if isinstance(pred, dict):
            for k in ("seg_out", "mask_logits", "seg", "segmentation", "logits"):
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

    def _build_cls_loss(self) -> nn.Module:
        """
        Classification criterion:
        - default: CrossEntropyLoss with optional class weights, label smoothing, ignore_index
        - if cfg['cls_loss'] == 'bce': BCEWithLogitsLoss with optional pos_weight (binary)
            * pos_weight is derived from class_weights when available (w_pos / w_neg),
            or taken from cfg['pos_weight'] if provided.
        """
        import torch
        import torch.nn as nn

        device = self._get("device", None)
        mode = str(self._get("cls_loss", "auto")).lower()  # 'auto'|'ce'|'bce'
        ls = self._get("label_smoothing", None)
        ig = self._get("ignore_index", None)

        # BCE path (explicitly requested)
        if mode == "bce":
            pos_w_tensor = None

            # Try to derive pos_weight from per-class weights if binary
            try:
                w = self._get_class_weights(device=device, dtype=torch.float32)
                if w is not None and w.numel() == 2 and w[0].item() > 0:
                    pos_w_tensor = torch.tensor([w[1].item() / w[0].item()], device=device)
            except Exception:
                pass

            # Or take explicit cfg['pos_weight']
            if pos_w_tensor is None:
                pos_w = self._get("pos_weight", None)
                if isinstance(pos_w, (int, float)):
                    pos_w_tensor = torch.tensor([float(pos_w)], device=device)

            return nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

        # CE path (default / 'auto' / 'ce')
        weight = None
        try:
            weight = self._get_class_weights(device=device, dtype=torch.float32)
        except Exception:
            weight = None

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
        import torch

        t = (task or "classification").lower()
        device = (cfg or {}).get("device", None)

        # Base criteria
        ce_cls = self._build_cls_loss()
        ce_seg = self._build_seg_loss()

        # Build a BCEWithLogits alternative for binary heads, with pos_weight if available
        bce_pos_weight = None
        try:
            w = self._get_class_weights(device=device, dtype=torch.float32)  # per-class weights for CE
            if w is not None and w.numel() == 2 and w[0].item() > 0:
                # pos_weight (BCE) ~ CE_weight_pos / CE_weight_neg
                bce_pos_weight = torch.tensor([w[1].item() / w[0].item()], device=device)
        except Exception:
            pass
        bce_cls = torch.nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)  # fine if pos_weight is None

        # Optional override in config: cls_loss: auto|ce|bce
        cls_mode = str(self._get("cls_loss", "auto")).lower()

        def cls_loss(pred: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
            # logits = self.extract_logits(pred)          # [B, C] or [B, 1]
            logits = self._get_cls_logits(pred)
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits)
            y = self._get_cls_target(tgt)               # [B] (int) or similar

            # Decide BCE vs CE
            use_bce = (cls_mode == "bce") or (cls_mode == "auto" and logits.dim() == 2 and logits.size(-1) == 1)
            if use_bce:
                # BCE expects float targets with same shape as logits
                yf = y.float().view_as(logits)
                return bce_cls(logits, yf)
            else:
                # CE expects class indices [B] (long)
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
