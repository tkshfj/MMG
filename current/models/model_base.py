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

    # generic logits extractor (used by legacy cls loss)
    def extract_logits(self, y_pred):
        """Default: return logits (first item if a tuple). Subclasses may override."""
        if isinstance(y_pred, (tuple, list)) and len(y_pred) > 0:
            return y_pred[0]
        return y_pred

    # class weights & CE builder (classification)
    def _get_class_weights(self, device=None, dtype=None):
        import torch
        C = int(self._get("num_classes", 2))
        explicit_w = self._get("class_weights", None)
        counts = self._get("class_counts", None)

        if explicit_w is not None:
            w = torch.as_tensor(explicit_w, dtype=torch.float)
        elif counts is not None:
            cnt = torch.as_tensor(counts, dtype=torch.float).clamp_min(1)
            inv = 1.0 / cnt
            w = inv * (inv.numel() / inv.sum())
        else:
            w = None

        if w is not None and w.numel() != C:
            raise ValueError(f"class weights length {w.numel()} != num_classes {C}")
        if w is not None and (device is not None or dtype is not None):
            w = w.to(device=device if device is not None else w.device,
                     dtype=dtype if dtype is not None else w.dtype)
        return w

    def _unwrap_label(self, y_true):
        """Return a 1D label tensor/array from y_true; accept dicts with common keys."""
        _LABEL_KEYS = ("label", "classification", "class", "target", "y")
        if isinstance(y_true, dict):
            for k in _LABEL_KEYS:
                if k in y_true:
                    return y_true[k]
            raise ValueError(f"Expected one of {_LABEL_KEYS} in y_true dict, got {list(y_true.keys())}")
        return y_true

    def _get_ce_criterion(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        label_smoothing: Optional[float] = None,
        ignore_index: Optional[int] = None,
    ) -> nn.Module:
        w = self._get_class_weights(device=device, dtype=dtype)
        kwargs: Dict[str, Any] = {}
        if label_smoothing is not None:
            kwargs["label_smoothing"] = float(label_smoothing)
        if ignore_index is not None:
            kwargs["ignore_index"] = int(ignore_index)
        return nn.CrossEntropyLoss(weight=w, **kwargs)

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
        # tolerate either "cls_out" (default) or "class_logits" (MultitaskUNet)
        v = self._pick(pred, "cls_out", "class_logits")
        if v is None:
            raise KeyError("No classification logits found under 'cls_out' or 'class_logits'.")
        return v

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
        # Weighted CE (uses config knobs if present)
        ls = self._get("label_smoothing", None)
        ig = self._get("ignore_index", None)
        return self._get_ce_criterion(label_smoothing=ls, ignore_index=ig)

    def _build_seg_loss(self) -> nn.Module:
        # Default seg loss: CE over index masks (compatible with ConfusionMatrix/DiceCE flows)
        ig = self._get("ignore_index", None)
        return nn.CrossEntropyLoss(ignore_index=int(ig) if ig is not None else -100)

    def get_loss_fn(self, task: str, cfg: Optional[dict] = None):
        """
        Return a loss_fn(pred_dict, target) for the given task:
          - classification: CE over logits [B,C] and labels [B]
          - segmentation:   CE over logits [B,C|1,H,W] (binary auto-expanded) and mask [B,H,W]
          - multitask:      alpha*cls + beta*seg (alpha/beta from cfg or default 1.0)
        """
        t = (task or "classification").lower()
        ce_cls = self._build_cls_loss()
        ce_seg = self._build_seg_loss()

        def cls_loss(pred: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
            logits = self.extract_logits(pred)
            y = self._get_cls_target(tgt)
            return ce_cls(logits, y)

        def seg_loss(pred: Dict[str, Any], tgt: Dict[str, Any]) -> torch.Tensor:
            logits = self._seg_logits_for_ce(self._get_seg_logits(pred))
            y = self._get_seg_target(tgt)
            return ce_seg(logits, y)

        if t in {"classification", "cls"}:
            return cls_loss
        if t in {"segmentation", "seg"}:
            return seg_loss

        # multitask
        alpha = float(self._cfg_get(self._cfg(cfg), "cls_weight", self._cfg_get(self._cfg(cfg), "alpha", 1.0)))
        beta = float(self._cfg_get(self._cfg(cfg), "seg_weight", self._cfg_get(self._cfg(cfg), "beta", 1.0)))
        return lambda pred, tgt: alpha * cls_loss(pred, tgt) + beta * seg_loss(pred, tgt)
