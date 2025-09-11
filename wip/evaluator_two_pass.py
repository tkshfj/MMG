# evaluator_two_pass.py
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, Any
import logging
import math
import numpy as np
import torch
from ignite.engine import Events, Engine
from ignite.metrics import ConfusionMatrix, DiceCoefficient, JaccardIndex
from protocols import CalibratorProtocol
from utils.safe import to_py, labels_to_1d_indices
from posprob import PosProbCfg, positive_score_from_logits
from metrics_utils import (
    extract_cls_logits_from_any,
    extract_seg_logits_from_any,
    add_confmat,
    promote_vec,
    get_mask_from_batch,
    seg_confmat_output_transform,
    make_std_cls_metrics_with_cal_thr
)

Loggable = Dict[str, Any]
TwoPassRunner = Callable[[int], Tuple[float, Dict[str, Any], Dict[str, Any]]]

logger = logging.getLogger(__name__)

SegEvalFn = Callable[[Any, Any], Dict[str, Any]]  # (model, val_loader) -> seg metrics dict
DiceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
IoUFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# local helpers
def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = labels.astype(np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, scores))
    except Exception:
        pass
    r = np.empty_like(scores, dtype=np.float64)
    order = np.argsort(scores)
    s_sorted = scores[order]
    ranks = np.empty_like(s_sorted, dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    r[order] = ranks
    sum_pos = float(r[y == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _default_classify_fn(model, x: torch.Tensor) -> torch.Tensor:
    for name in ("classify", "forward_classify", "predict_cls", "infer_cls"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn(x)
    return extract_cls_logits_from_any(model(x))


def as_bool(x, default: bool = True) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return default


def _pos_rate(scores: np.ndarray, thr: float) -> float:
    if scores.size == 0:
        return 0.0
    return float((scores >= float(thr)).mean())


def _rate_match_threshold(scores: np.ndarray, target_rate: float) -> float:
    """
    Monotone projection: choose t so that pos_rate(scores >= t) ~= target_rate.
    Using the (1 - target_rate)-quantile achieves that.
    """
    if scores.size == 0:
        return 0.5
    q = float(np.clip(1.0 - target_rate, 0.0, 1.0))
    return float(np.quantile(scores, q))


def _bounded_step(old: float, new: float, *, max_delta: float, lo: float, hi: float) -> float:
    step = np.clip(new - old, -float(max_delta), float(max_delta))
    return float(np.clip(old + step, float(lo), float(hi)))


class ThresholdBox:
    def __init__(self, value=0.5):
        self.value = float(value)


# main evaluator
class TwoPassEvaluator:
    """
    Mode-aware two-pass evaluator.
    - If has_cls/has_seg is None, the first validate() auto-detects capability from a batch.
    - Otherwise respects the flags we pass.
    """
    def __init__(
        self,
        calibrator: Optional[CalibratorProtocol] = None,
        trainer: Optional[Engine] = None,
        positive_index: int = 1,
        *,
        has_cls: Optional[bool] = None,
        has_seg: Optional[bool] = None,
        classify_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        seg_eval_fn: Optional[SegEvalFn] = None,
        dice_fn: Optional[DiceFn] = None,
        iou_fn: Optional[IoUFn] = None,
        rate_tol_override: float | None = None,
        seg_threshold: float = 0.5,
        num_classes: int = 2,
        cls_threshold: float = 0.5,
        health_tol: float = 0.35,
        health_need_k: int = 3,
        health_warmup: Optional[int] = None,
        enable_decision_health: bool = False,
        score_provider: PosProbCfg | None = None,
        cfg: dict | None = None,
    ):
        self.cfg = dict(cfg) if cfg is not None else {}
        self.calibrator = calibrator
        self._trainer = trainer
        self.pos_idx = int(positive_index)
        self._has_cls = has_cls
        self._has_seg = has_seg
        self.classify_fn = classify_fn or _default_classify_fn
        self.seg_eval_fn = seg_eval_fn
        self.dice_fn = dice_fn
        self.iou_fn = iou_fn
        self.rate_tol_override = rate_tol_override
        self.last_threshold: Optional[float] = None
        self.bad_count: int = 0
        self._capabilities_frozen = False
        self.seg_threshold = float(seg_threshold)
        self.num_classes = int(num_classes)
        self._std_engine: Optional[Engine] = None
        self._current_model = None

        self.enable_decision_health = enable_decision_health
        self.health_tol = float(health_tol)
        self.health_need_k = int(health_need_k)
        self.health_warmup = int(health_warmup) if health_warmup is not None else int(getattr(getattr(calibrator, "cfg", None), "warmup_epochs", 1))
        self._thr_box = ThresholdBox(value=float(cls_threshold))
        self._watchdog_active: bool = False   # suppress internal health logs when external watchdog is attached
        # Single source of truth for score extraction / binarization
        self._score_provider = score_provider or PosProbCfg(
            binary_single_logit=True, positive_index=self.pos_idx
        )
        # Threshold guardrails
        _ccfg = getattr(calibrator, "cfg", None)
        self.thr_min = float(getattr(_ccfg, "thr_min", 0.05))
        self.thr_max = float(getattr(_ccfg, "thr_max", 0.95))
        self.thr_posrate_min = float(getattr(_ccfg, "thr_posrate_min", 0.02))
        self.thr_posrate_max = float(getattr(_ccfg, "thr_posrate_max", 0.98))
        self.thr_warmup_epochs = int(getattr(_ccfg, "thr_warmup_epochs", max(5, int(getattr(_ccfg, "warmup_epochs", 1)))))
        self.cal_auc_floor = float(getattr(_ccfg, "cal_auc_floor", 0.50))
        self.thr_min_tp = int(getattr(_ccfg, "thr_min_tp", 0))
        self.thr_min_tn = int(getattr(_ccfg, "thr_min_tn", 0))
        # smoothing & hysteresis knobs (gentle defaults)
        self.cal_ema_beta = float(getattr(_ccfg, "cal_ema_beta", 0.30))
        self.thr_hysteresis = float(getattr(_ccfg, "thr_hysteresis", 0.02))

    # central threshold sanitizer
    def _clamp_thr01(self, value, *, default: float = 0.5, ctx: str = "thr") -> float:
        """
        Cast to float, fall back to `default` if cast fails or non-finite, and clamp to [0, 1].
        Logs once if clamped or bad.
        """
        try:
            f = float(value)
        except Exception:
            logger.warning("bad %s: %r (non-castable); using default=%.3f", ctx, value, default)
            return float(default)
        if not math.isfinite(f):
            logger.warning("bad %s: %r (non-finite); using default=%.3f", ctx, f, default)
            return float(default)
        if f < 0.0 or f > 1.0:
            logger.info("%s out of [0,1]: %.6f -> clamped", ctx, f)
        return float(min(1.0, max(0.0, f)))

    def _resolve_rate_tolerance(self) -> float:
        if getattr(self, "rate_tol_override", None) is not None:
            return float(self.rate_tol_override)
        cfg = getattr(self.calibrator, "cfg", None)
        # try multiple names to be sweep-friendly
        for k in ("rate_tolerance", "cal_rate_tolerance"):
            if cfg is not None and hasattr(cfg, k):
                return float(getattr(cfg, k))
        return 0.10

    def _resolve_max_delta(self) -> float:
        # single source of truth for max step
        if hasattr(self, "cal_max_delta"):
            return float(self.cal_max_delta)
        cfg = getattr(self.calibrator, "cfg", None)
        return float(getattr(cfg, "cal_max_delta", 0.10))

    # Low-spread clamp (always-on pre-step)
    def _low_spread_posrate_clamp(self, probs: np.ndarray, t_seed: float) -> float:
        """
        If scores have (very) low spread and the current pos-rate (under t_seed)
        sits outside the allowed corridor, softly move the threshold toward the
        corridor using a quantile projection, bounded by cal_max_delta and [thr_min, thr_max].
        Returns the (possibly) adjusted seed threshold.
        """
        if probs.size == 0:
            return float(t_seed)

        std = float(np.std(probs))
        q10 = float(np.quantile(probs, 0.10))
        q90 = float(np.quantile(probs, 0.90))
        iqr = q90 - q10
        pos_rate_seed = float((probs >= float(t_seed)).mean())

        cfg = getattr(self.calibrator, "cfg", None)
        cal_min_std = float(getattr(cfg, "cal_min_std", 1e-4))
        cal_min_iqr = float(getattr(cfg, "cal_min_iqr", 1e-3))

        if std >= cal_min_std or iqr >= cal_min_iqr:
            # Not low-spread → leave seed as-is
            return float(t_seed)

        # Low-spread; if rate sits outside corridor, project toward the nearest bound
        if pos_rate_seed < self.thr_posrate_min or pos_rate_seed > self.thr_posrate_max:
            target = self.thr_posrate_min if pos_rate_seed < self.thr_posrate_min else self.thr_posrate_max
            t_dm = _rate_match_threshold(probs, float(target))
            t_adj = _bounded_step(
                float(t_seed), float(t_dm),
                max_delta=self._resolve_max_delta(),
                lo=float(self.thr_min), hi=float(self.thr_max)
            )
            logger.info(
                "[cal] low-spread clamp %.4f → %.4f (pos_rate %.3f → [%0.2f,%0.2f]) std=%.4g iqr=%.4g",
                float(t_seed), float(t_adj), pos_rate_seed, self.thr_posrate_min, self.thr_posrate_max, std, iqr
            )
            return float(t_adj)

        # Low-spread but already inside corridor → keep seed
        return float(t_seed)

    # Public guards
    @property
    def has_cls(self) -> bool:
        return bool(self._has_cls)

    @property
    def has_seg(self) -> bool:
        return bool(self._has_seg)

    def _compute_seg_metrics(self, model, val_loader) -> Dict:
        if self.seg_eval_fn is None and (self.dice_fn is None or self.iou_fn is None):
            return {}

        if self.seg_eval_fn is not None:
            raw = dict(self.seg_eval_fn(model, val_loader))  # whatever shim returns
            flat: Dict[str, Any] = {}
            # Dice / IoU / Loss
            if "dice" in raw:
                promote_vec(flat, "dice", raw["dice"])
            if "iou" in raw:
                promote_vec(flat, "iou", raw["iou"])
            if "loss" in raw:
                flat["loss"] = float(to_py(raw["loss"]))
            # Confusion matrix (any reasonable key)
            for k in ("seg_confmat", "confmat", "seg_cm"):
                if k in raw:
                    add_confmat(flat, "seg_confmat", raw[k])
                    break
            # Keep any other simple scalars (auto-sanitized)
            for k, v in raw.items():
                if k in flat:
                    continue
                pv = to_py(v)
                if isinstance(pv, (float, int, str)) or pv is None:
                    flat[k] = pv
            return flat

        # Fallback: naive per-batch dice/iou
        dice_scores, iou_scores = [], []
        device = getattr(model, "device", next(model.parameters()).device)
        with torch.inference_mode():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)

                # robust mask lookup (supports nested label dict)
                y_mask = get_mask_from_batch(batch)
                if y_mask is None:
                    continue
                y_mask = y_mask.to(device)

                # logits from model
                out = model.segment(x) if hasattr(model, "segment") else model(x)
                seg_logits = out if (torch.is_tensor(out) and out.ndim >= 3) else extract_seg_logits_from_any(out)

                # handle binary vs multi-class correctly
                if seg_logits.shape[1] == 1:  # binary
                    prob = torch.sigmoid(seg_logits[:, 0])  # [B,H,W]
                    pred = (prob >= self.seg_threshold).long()  # indices {0,1}
                else:  # multiclass
                    pred = torch.softmax(seg_logits, dim=1).argmax(1).long()  # [B,H,W]

                # Ensure masks have (B,H,W)
                if y_mask.ndim == 4 and y_mask.size(1) == 1:
                    y_mask = y_mask[:, 0]
                y_mask = y_mask.long()

                dice_scores.append(self.dice_fn(pred, y_mask))
                iou_scores.append(self.iou_fn(pred, y_mask))

        return {
            "dice": float(torch.stack(dice_scores).mean().item()) if dice_scores else 0.0,
            "iou": float(torch.stack(iou_scores).mean().item()) if iou_scores else 0.0,
        }

    @staticmethod
    @torch.no_grad()
    def _compute_loss(model, val_loader, loss_fn, prepare_batch, device) -> float:
        model.eval()
        total, count = 0.0, 0
        pin = bool(getattr(val_loader, "pin_memory", False))
        for batch in val_loader:
            x, targets = prepare_batch(batch, device, pin)
            out = model(x)
            loss = loss_fn(out, targets)
            loss_f = float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss)
            total += loss_f
            count += 1
        return total / max(1, count)

    # main entrypoint
    @torch.inference_mode()
    def validate(self, epoch: int, model, val_loader, base_rate: Optional[float] = None) -> Tuple[float, Dict, Dict]:
        # Make val_loader discoverable by decision-health wiring
        self.val_loader = val_loader

        def _ingest_cls_metric_map(dst: Dict[str, Any], raw: Dict[str, Any]) -> None:
            """
            Copy classification metrics from Ignite to a flat Python dict.
            - 'prec'/'recall' may be scalar or per-class vectors -> promote_vec expands as needed.
            - Scalars (acc/auc/pos_rate/gt_pos_rate) are cast to float safely.
            - Confusion matrix is expanded into list plus integer cells.
            """
            # Scalars we expect to be scalar-like
            for k in ("acc", "auc", "pos_rate", "gt_pos_rate"):
                if k in raw:
                    v = to_py(raw[k])
                    if isinstance(v, (int, float)):
                        dst[k] = float(v)
            # Vector-or-scalar metrics -> promote to per-index keys and mean
            # Old name (“prec”/“recall”) or new name (“precision_pos”/“recall_pos”)
            if "prec" in raw:  # yields 'prec', 'prec_0', 'prec_1', ...
                promote_vec(dst, "prec", raw["prec"])
            elif "precision_pos" in raw:
                v = to_py(raw["precision_pos"])
                if isinstance(v, (int, float)):
                    dst["prec"] = float(v)
            if "recall" in raw:  # yields 'recall', 'recall_0', 'recall_1', ...
                promote_vec(dst, "recall", raw["recall"])
            elif "recall_pos" in raw:
                v = to_py(raw["recall_pos"])
                if isinstance(v, (int, float)):
                    dst["recall"] = float(v)
            # Optional: also expose macro scores if present
            if "precision_macro" in raw and "prec" not in dst:
                vm = to_py(raw["precision_macro"])
                if isinstance(vm, (int, float)):
                    dst["prec_macro"] = float(vm)
            if "recall_macro" in raw and "recall" not in dst:
                vm = to_py(raw["recall_macro"])
                if isinstance(vm, (int, float)):
                    dst["recall_macro"] = float(vm)
            # Confusion matrix -> matrix list + per-cell ints
            if "cls_confmat" in raw:
                add_confmat(dst, "cls_confmat", raw["cls_confmat"])

        device = getattr(model, "device", next(model.parameters()).device)
        # device = _safe_model_device(model)
        model.eval()
        self._current_model = model

        # One-time auto-detect if flags not forced
        if not self._capabilities_frozen and (self._has_cls is None or self._has_seg is None):
            self._autodetect_capabilities(model, val_loader, device)
            self._capabilities_frozen = True
            logger.info("[TwoPass] capabilities: has_cls=%s has_seg=%s", self._has_cls, self._has_seg)

        cls_metrics: Dict[str, Any] = {}
        seg_metrics: Dict[str, Any] = {}

        # Seed threshold (trust only previously accepted → box; never seed from calibrator)
        if self.last_threshold is not None:
            t_seed = float(self.last_threshold)
        else:
            try:
                t_seed = float(self._thr_box.value)
            except Exception:
                t_seed = 0.5
        t_seed = self._clamp_thr01(t_seed, default=0.5, ctx="init_thr")

        # Classification
        if self.has_cls:
            # A) collect LOGITS + labels
            logits_np, labels = self._gather_logits_labels(model, val_loader, device)

            # derive base rate if not provided
            if base_rate is None and labels.size > 0:
                base_rate = float((labels == 1).mean())

            # B) logits -> probs once (PosProbCfg = single SoT)
            #    Support either 1-logit (BCE) or C-logits (CE); keep on CPU numpy
            if logits_np.ndim == 1 or (logits_np.ndim == 2 and logits_np.shape[-1] == 1):
                # 1-logit case
                t = torch.from_numpy(logits_np.reshape(-1))
            else:
                # multi-logit case [N, C]
                t = torch.from_numpy(logits_np)
            probs = self._score_provider.logits_to_pos_prob(t).detach().reshape(-1).clamp_(0, 1).cpu().numpy()
            # Always-on low-spread clamp (pre-calibrator); does not early-return
            t_seed = self._low_spread_posrate_clamp(probs, float(t_seed))

            # C) Candidate from calibrator (now pass PROBS, unchanged API)
            t_cand = t_seed
            if self.calibrator is not None:
                try:
                    picked = self.calibrator.pick(epoch, probs, labels, base_rate)
                    t_new = picked[0] if isinstance(picked, (tuple, list)) else picked
                    t_cand = self._clamp_thr01(t_new, default=t_seed, ctx="calibrator_thr")
                except Exception:
                    logger.warning("calibrator.pick failed; keeping seed threshold", exc_info=True)

            # D) Smooth + guardrails + acceptance (operate on PROBS)
            prev = float(self.last_threshold if self.last_threshold is not None else t_seed)
            # 1) clamp candidate into global bounds
            t_cand = float(min(self.thr_max, max(self.thr_min, t_cand)))
            # 2) hysteresis: if candidate's pos-rate is comfortably inside corridor, hold position
            pr_cand = _pos_rate(probs, t_cand)
            if (self.thr_posrate_min + self.thr_hysteresis) <= pr_cand <= (self.thr_posrate_max - self.thr_hysteresis):
                t_cand = prev
            # 3) EMA smoothing and max-delta bounding (gentle movement)
            t_smooth = prev + self.cal_ema_beta * (t_cand - prev)
            t_smooth = _bounded_step(prev, t_smooth, max_delta=self._resolve_max_delta(), lo=self.thr_min, hi=self.thr_max)

            # 4) acceptance checks AFTER smoothing
            def _accept_smoothed(t_try: float) -> bool:
                if epoch < int(self.thr_warmup_epochs) or probs.size == 0:
                    return False
                t_try = float(min(self.thr_max, max(self.thr_min, t_try)))
                # pos-rate corridor
                pr = _pos_rate(probs, t_try)
                if not (self.thr_posrate_min <= pr <= self.thr_posrate_max):
                    return False
                # auc floor on the *scores* distribution (model separability)
                auc_val = _safe_auc(labels, probs)
                if not (isinstance(auc_val, float) and math.isfinite(auc_val) and auc_val >= self.cal_auc_floor):
                    return False
                # enforce BOTH TN and TP mins
                if self.thr_min_tp > 0 or self.thr_min_tn > 0:
                    pred = (probs >= t_try).astype(np.int64)
                    tp = int(((labels == 1) & (pred == 1)).sum())
                    tn = int(((labels == 0) & (pred == 0)).sum())
                    if tp < self.thr_min_tp or tn < self.thr_min_tn:
                        return False
                # movement guard vs last accepted
                max_delta = self._resolve_max_delta()
                if self.last_threshold is not None and abs(t_try - float(self.last_threshold)) > max_delta:
                    return False
                return True
            if _accept_smoothed(t_smooth):
                t_final = float(t_smooth)
            else:
                # reject update → keep previous (stable)
                t_final = float(prev)

            # E) optional decision-health rate matching (still on PROBS)
            if self.enable_decision_health and base_rate is not None and epoch >= int(self.health_warmup):
                rate_tol = self._resolve_rate_tolerance()
                pr_now = _pos_rate(probs, t_final)
                if abs(pr_now - float(base_rate)) > rate_tol:
                    t_dm = _rate_match_threshold(probs, float(base_rate))
                    t_final = _bounded_step(
                        float(t_final), float(t_dm),
                        max_delta=self._resolve_max_delta(),
                        lo=float(self.thr_min), hi=float(self.thr_max)
                    )

            # F) publish once and run the standard engine to emit metrics
            self.last_threshold = float(t_final)
            self._thr_box.value = self._clamp_thr01(self.last_threshold, default=0.5, ctx="thr_box")
            self._ensure_std_engine()

            pin = bool(getattr(val_loader, "pin_memory", False))
            setattr(self._std_engine.state, "device", device)
            setattr(self._std_engine.state, "non_blocking", pin)
            self._std_engine.run(val_loader)
            raw = dict(self._std_engine.state.metrics or {})
            _ingest_cls_metric_map(cls_metrics, raw)
            cls_metrics["threshold"] = float(self._thr_box.value)
            cls_metrics["cal_thr"] = float(self._thr_box.value)

            if base_rate is not None:
                cls_metrics["base_rate"] = float(base_rate)

            if self.enable_decision_health and base_rate is not None and epoch >= int(self.health_warmup):
                rate_tol = self._resolve_rate_tolerance()
                pos_rate = float(cls_metrics.get("pos_rate", 0.0))
                delta = abs(pos_rate - base_rate)
                self.bad_count = self.bad_count + 1 if (delta > rate_tol) else 0
                if not getattr(self, "_watchdog_active", False):
                    logger.info("[decision health] pos=%.4f base=%.4f Δ=%.4f tol=%.4f bad_now=%d",
                                pos_rate, base_rate, delta, rate_tol, self.bad_count)

        # Segmentation
        if self.has_seg:
            # If the Ignite engine has seg metrics attached, read them; otherwise keep fallback
            if self._std_engine is not None and "seg_confmat" in (self._std_engine.state.metrics or {}):
                raw = dict(self._std_engine.state.metrics or {})
                # dice / iou may be scalars or per-class vectors; promote_vec handles both
                if "dice" in raw:
                    promote_vec(seg_metrics, "dice", raw["dice"])
                if "iou" in raw:
                    promote_vec(seg_metrics, "iou", raw["iou"])
                if "seg_confmat" in raw:
                    add_confmat(seg_metrics, "seg_confmat", raw["seg_confmat"])
                # keep loss if present
                if "loss" in raw:
                    v = to_py(raw["loss"])
                    if isinstance(v, (int, float)):
                        seg_metrics["loss"] = float(v)
            else:
                seg_metrics = self._compute_seg_metrics(model, val_loader)

        # Combined scores (linear mix; keep behavior)
        if self.has_cls and self.has_seg and "auc" in cls_metrics and "dice" in seg_metrics:
            dice_mean = float(seg_metrics["dice"])
            w = getattr(getattr(model, "cfg", None), "multi_weight", 0.65)
            cls_metrics["multi"] = (1.0 - w) * float(cls_metrics["auc"]) + w * dice_mean
            # Optional per-class multi with same AUC (binary symmetry)
            if "dice_0" in seg_metrics:
                cls_metrics["multi_0"] = (1.0 - w) * float(cls_metrics["auc"]) + w * float(seg_metrics["dice_0"])
            if "dice_1" in seg_metrics:
                cls_metrics["multi_1"] = (1.0 - w) * float(cls_metrics["auc"]) + w * float(seg_metrics["dice_1"])

        # Return the accepted threshold (or current box value) with metrics
        return float(self._thr_box.value), cls_metrics, seg_metrics

    # capability detection
    def _autodetect_capabilities(self, model, val_loader, device) -> None:
        # Peek one batch (val_loader must be re-iterable)
        it = iter(val_loader)
        try:
            batch = next(it)
        except StopIteration:
            self._has_cls, self._has_seg = False, False
            return
        has_label = ("label" in batch) or ("y" in batch)
        has_mask = ("mask" in batch) or (isinstance(batch.get("label"), dict) and "mask" in batch["label"])

        # Try classification extraction
        can_cls = False
        if has_label:
            try:
                x = batch["image"].to(device, non_blocking=True)
                _ = self.classify_fn(model, x)
                can_cls = True
            except Exception:
                can_cls = False

        # Seg path available if seg_eval_fn exists OR we can extract seg logits + have a mask + dice/iou
        can_seg = False
        if self.seg_eval_fn is not None:
            can_seg = True
        elif has_mask and (self.dice_fn is not None) and (self.iou_fn is not None):
            try:
                out = model(batch["image"].to(device, non_blocking=True))
                _ = extract_seg_logits_from_any(out)
                can_seg = True
            except Exception:
                can_seg = False

        # Respect user-provided flags if set; otherwise use detection
        self._has_cls = can_cls if self._has_cls is None else bool(self._has_cls)
        self._has_seg = can_seg if self._has_seg is None else bool(self._has_seg)

    # internals
    def _gather_logits_labels(self, model, val_loader, device) -> Tuple[np.ndarray, np.ndarray]:
        all_logits, all_labels = [], []
        for batch in val_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = labels_to_1d_indices(batch.get("label", batch.get("y"))).cpu().numpy()

            out = self.classify_fn(model, x)
            logits = extract_cls_logits_from_any(out)
            if not torch.is_tensor(logits):
                logits = torch.as_tensor(logits)

            # Keep raw logits; shape -> [B] or [B, C] ok
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y)

        if not all_logits:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels).astype(np.int64)

        # sanitize NaNs/Infs in logits (rare, but defensive)
        if not np.isfinite(logits).all():
            logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return logits.astype(np.float32), labels

    # Ignite engine wiring
    def _ensure_std_engine(self) -> None:
        """
        Build a single Ignite evaluator that:
        - emits a dict with cls logits/labels and seg logits/masks
        - has classification metrics wired through:
                * AUC (raw logits OT)
                * Acc/Prec/Recall/ConfMat/pos_rate/gt_pos_rate (decision OT via self._thr_box)
        - has segmentation metrics wired through ConfusionMatrix -> Dice/IoU
        """
        if self._std_engine is not None:
            return

        def _step(engine, batch):
            model = self._current_model
            device = getattr(self._std_engine.state, "device", next(model.parameters()).device)
            nb = bool(getattr(self._std_engine.state, "non_blocking", False))

            # inputs → device
            x = batch["image"]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            x = x.float().to(device, non_blocking=nb)

            # optional targets from batch
            y_any = batch.get("label", batch.get("y"))
            y_mask = get_mask_from_batch(batch)

            with torch.inference_mode():
                raw = model(x)

            out_map: Dict[str, Any] = {}

            # classification logits (if present)
            try:
                cls_logits = extract_cls_logits_from_any(raw)
                out_map["cls_out"] = cls_logits
                out_map["logits"] = cls_logits
                out_map["y_pred"] = cls_logits  # Ignite CM wants y_pred/y
                # optional debug:
                out_map["prob_pos"] = positive_score_from_logits(cls_logits, positive_index=self.pos_idx)
            except Exception:
                pass

            # segmentation logits (if present)
            try:
                seg_logits = extract_seg_logits_from_any(raw)
                out_map["seg_out"] = seg_logits
                out_map["seg_logits"] = seg_logits
            except Exception:
                pass

            # labels/masks -> SAME device as predictions/evaluator
            label_dict: Dict[str, Any] = {}
            if y_any is not None:
                y = labels_to_1d_indices(y_any).to(device, non_blocking=nb)
                out_map["y"] = y
                label_dict["label"] = y
            if y_mask is not None:
                m = torch.as_tensor(y_mask).long().to(device, non_blocking=nb)
                out_map["mask"] = m
                label_dict["mask"] = m
            if label_dict:
                out_map["label"] = label_dict

            return out_map

        self._std_engine = Engine(_step)

        # Classification (AUC from raw logits; thresholded metrics via thr_box)
        if self.has_cls:

            def _thr_getter() -> float:
                # Always read the most recent threshold set in validate()
                return float(self._thr_box.value)

            cls_metrics = make_std_cls_metrics_with_cal_thr(
                num_classes=self.num_classes,
                decision="threshold",
                positive_index=self.pos_idx,
                thr_getter=_thr_getter,
                score_provider=self._score_provider,
            )

            for name, m in cls_metrics.items():
                m.attach(self._std_engine, name)

        # Decision health on the STANDARD pass (uncalibrated)
        # Only if a trainer was provided and classification metrics exist.
        if self._trainer is not None:
            try:
                from decision_health import attach_decision_health

                warmup = int(self.health_warmup)
                if self.enable_decision_health:
                    attach_decision_health(
                        trainer=self._trainer,
                        evaluator=self._std_engine,
                        val_loader=getattr(self, "val_loader", None),
                        num_classes=int(self.num_classes),
                        positive_index=int(self.pos_idx),
                        tol=float(self.health_tol),
                        need_k=int(self.health_need_k),
                        warmup=int(warmup),
                        metric_key="cls_confmat",
                        inject_metrics=True,
                        terminate=bool(self.cfg.get("decision_health_terminate", False)),
                        min_samples=int(self.cfg.get("decision_health_min_samples", 200)),
                    )
                    # External watchdog is active; suppress internal duplicate logs
                    self._watchdog_active = True
            except Exception as e:
                logger.warning("attach_decision_health(std) failed: %s", e)

        # Segmentation (ConfMat -> Dice/IoU)
        if self.has_seg:
            seg_cm = ConfusionMatrix(
                num_classes=max(2, self.num_classes),
                output_transform=lambda o: seg_confmat_output_transform(o, threshold=self.seg_threshold),
            )
            seg_cm.attach(self._std_engine, "seg_confmat")
            DiceCoefficient(cm=seg_cm).attach(self._std_engine, "dice")
            JaccardIndex(cm=seg_cm).attach(self._std_engine, "iou")


def _to_float(v: Any) -> Any:
    try:
        if hasattr(v, "item") and callable(v.item):
            v = v.item()
    except Exception:
        pass
    return float(v) if isinstance(v, (int, float)) else v


def attach_two_pass_validation(
    *,
    trainer: Engine,
    run_two_pass: TwoPassRunner,             # closure that calls your two_pass.validate(...)
    cal_warmup_epochs: int = 1,
    disable_std_logging: bool = True,        # turn off any existing 1-pass epoch logger
    wandb_prefix: str = "val/",
    log_fn: Callable[[Loggable], None] | None = None,  # e.g., wandb.log
) -> None:
    """
    Attach a warmup-guarded, idempotent two-pass validation hook.

    - Metrics MUST already be attached to engines with bare names.
    - This function only orchestrates the second pass and logs a single
      epoch-level payload with 'val/' prefix (and 'trainer/epoch' step).
    """
    st = getattr(trainer, "state", None)
    if st and getattr(st, "_two_pass_attached", False):
        return  # idempotent

    # Optionally remove an existing 1-pass logging hook to prevent double logs
    if disable_std_logging and st and hasattr(st, "_std_val_hook") and st._std_val_hook:
        try:
            trainer.remove_event_handler(st._std_val_hook, Events.EPOCH_COMPLETED)
        except Exception:
            pass
        st._std_val_hook = None

    # Define the epoch-completed hook
    @trainer.on(Events.EPOCH_COMPLETED)
    def _two_pass_hook(engine: Engine) -> None:
        epoch = int(getattr(engine.state, "epoch", 0) or 0)
        if epoch < int(cal_warmup_epochs):
            return

        try:
            thr, cls_m, seg_m = run_two_pass(epoch)  # user-supplied runner
        except Exception:
            # Never crash training due to two-pass
            logger.debug("two-pass evaluation raised; skipping this epoch", exc_info=True)
            return

        # Build a flat payload with the 'val/' prefix
        payload: Loggable = {"trainer/epoch": epoch, f"{wandb_prefix}cal_thr": _to_float(thr)}
        for k, v in (cls_m or {}).items():
            payload[f"{wandb_prefix}{k}"] = _to_float(v)
        for k, v in (seg_m or {}).items():
            key = f"{wandb_prefix}{k}"
            if key not in payload:
                payload[key] = _to_float(v)

        if log_fn is not None and len(payload) > 1:
            try:
                log_fn(payload)  # e.g., wandb.log(payload)
            except Exception:
                logger.debug("two-pass logging failed; continuing.", exc_info=True)

        # Optional: expose results to trainer.state.metrics for schedulers/handlers
        engine.state.metrics = (engine.state.metrics or {})
        engine.state.metrics.update({**(cls_m or {}), **(seg_m or {}), "cal_thr": _to_float(thr)})

    if st is not None:
        st._two_pass_attached = True
        st._two_pass_hook = _two_pass_hook


# build a flat, prefixed payload
def make_val_log_payload(
    epoch: int,
    cls_metrics: Dict[str, Any],
    seg_metrics: Dict[str, Any],
    *,
    prefix: str = "val/"
) -> Dict[str, Any]:
    """
    Build a single flat dict for logging, with epoch added and keys prefixed.
    No side effects; caller decides how/when to log and with which step.
    """
    payload: Dict[str, Any] = {"trainer/epoch": int(epoch)}
    for k, v in cls_metrics.items():
        payload[f"{prefix}{k}"] = v
    for k, v in seg_metrics.items():
        # don't overwrite same key if present on cls side
        if f"{prefix}{k}" not in payload:
            payload[f"{prefix}{k}"] = v
    return payload


def make_two_pass_evaluator(
    *,
    calibrator: Optional[CalibratorProtocol] = None,
    task: str,
    trainer: Optional[Engine] = None,
    positive_index: int = 1,
    dice_fn: Optional[DiceFn] = None,
    iou_fn: Optional[IoUFn] = None,
    cls_decision: str = "threshold",
    cls_threshold: float = 0.5,
    num_classes: int = 2,
    multitask: bool = False,
    seg_num_classes: Optional[int] = None,
    loss_fn=None,
    enable_decision_health: bool = True,
    score_provider: PosProbCfg | None = None,
    cfg: dict | None = None,
    **_
) -> TwoPassEvaluator:
    """
    Build the right flavor for:
      - "multitask" (cls + seg)
      - "classification"
      - "segmentation"

    No classify_key or seg_eval_fn required:
      • Classification uses the evaluator's _default_classify_fn (classify() if present,
        else robust extraction from model(x) outputs).
      • Segmentation tries to import eval.seg_eval.model_eval_seg automatically; if not
        found, it will use dice_fn/iou_fn if provided, else return empty seg metrics.
    """
    t = (task or "multitask").lower()
    has_cls = t != "segmentation"
    has_seg = t != "classification"
    # Try to auto-wire a segmentation shim if user has one in eval/seg_eval.py
    auto_seg_eval = None
    if has_seg:
        for mod, attr in (("eval.seg_eval", "model_eval_seg"), ("seg_eval", "model_eval_seg")):
            try:
                _m = __import__(mod, fromlist=[attr])
                fn = getattr(_m, attr, None)
                if callable(fn):
                    auto_seg_eval = fn
                    break
            except Exception:
                pass

    return TwoPassEvaluator(
        calibrator=calibrator,
        trainer=trainer,
        positive_index=positive_index,
        has_cls=has_cls,
        has_seg=has_seg,
        classify_fn=None,                 # uses _default_classify_fn internally
        seg_eval_fn=auto_seg_eval,        # auto shim if available
        dice_fn=dice_fn if (auto_seg_eval is None and has_seg) else None,
        iou_fn=iou_fn if (auto_seg_eval is None and has_seg) else None,
        seg_threshold=float(getattr(getattr(calibrator, "cfg", None), "seg_threshold", 0.5)),
        num_classes=int(num_classes),
        cls_threshold=float(cls_threshold),
        health_tol=float(getattr(getattr(calibrator, "cfg", None), "decision_health_tol", 0.35)),
        health_need_k=int(getattr(getattr(calibrator, "cfg", None), "decision_health_need_k", 3)),
        enable_decision_health=bool(enable_decision_health),
        score_provider=score_provider,
        cfg=cfg or {},
    )


__all__ = ["attach_two_pass_validation", "make_two_pass_evaluator", "TwoPassEvaluator"]

# # Classification
# if self.has_cls:
#     # Gather scores/labels for calibration
#     scores, labels = self._gather_scores_labels(model, val_loader, device)

#     # derive base rate if not provided
#     if base_rate is None and labels.size > 0:
#         base_rate = float((labels == 1).mean())

#     # Candidate threshold from calibrator (or stick with seed)
#     t_cand = t_seed
#     if self.calibrator is not None:
#         try:
#             picked = self.calibrator.pick(epoch, scores, labels, base_rate)
#             t_new = picked[0] if isinstance(picked, (tuple, list)) else picked
#             t_cand = self._clamp_thr01(t_new, default=t_seed, ctx="calibrator_thr")
#         except Exception:
#             logger.warning("calibrator.pick failed; keeping seed threshold", exc_info=True)

#     # Guardrails & acceptance: only adopt candidate if sane/stable
#     def _accept(t_try: float) -> bool:
#         # warmup & data presence
#         if epoch < int(self.thr_warmup_epochs) or scores.size == 0:
#             return False
#         # clamp to global bounds
#         t_try = float(min(self.thr_max, max(self.thr_min, t_try)))
#         # basic sanity on pos_rate envelope (absolute bounds)
#         pr = _pos_rate(scores, t_try)
#         if not (self.thr_posrate_min <= pr <= self.thr_posrate_max):
#             return False
#         # auc floor
#         auc_val = _safe_auc(labels, scores)
#         if not (isinstance(auc_val, float) and math.isfinite(auc_val) and auc_val >= self.cal_auc_floor):
#             return False
#         # optional count constraints
#         if self.thr_min_tp > 0 or self.thr_min_tn > 0:
#             pred = (scores >= t_try).astype(np.int64)
#             tp = int(((labels == 1) & (pred == 1)).sum())
#             tn = int(((labels == 0) & (pred == 0)).sum())
#             if tp < self.thr_min_tp or tn < self.thr_min_tn:
#                 return False
#         # max jump (delta) vs last accepted
#         max_delta = self._resolve_max_delta()
#         if self.last_threshold is not None and abs(t_try - float(self.last_threshold)) > max_delta:
#             return False

#         return True

#     # Finalize exactly once, then write the box once
#     accepted = _accept(t_cand)
#     t_final = float(min(self.thr_max, max(self.thr_min, t_cand))) if accepted else float(t_seed)

#     if self.enable_decision_health and base_rate is not None and epoch >= int(self.health_warmup):
#         rate_tol = self._resolve_rate_tolerance()
#         pr_now = _pos_rate(scores, t_final)
#         if abs(pr_now - float(base_rate)) > rate_tol:
#             t_dm = _rate_match_threshold(scores, float(base_rate))
#             t_final = _bounded_step(
#                 float(t_final), float(t_dm),
#                 max_delta=self._resolve_max_delta(),
#                 lo=float(self.thr_min), hi=float(self.thr_max)
#             )
#         delta = abs(pr_now - float(base_rate))
#         if delta > float(rate_tol):
#             # Project to rate-matched threshold, then bound by max-delta and global bounds.
#             t_dm = _rate_match_threshold(scores, float(base_rate))
#             max_delta = float(getattr(getattr(self.calibrator, "cfg", None), "cal_max_delta", 0.10))
#             t_final = _bounded_step(
#                 float(t_final), float(t_dm),
#                 max_delta=max_delta, lo=float(self.thr_min), hi=float(self.thr_max)
#             )

#     # remember accepted threshold and publish it once
#     self.last_threshold = float(t_final)
#     # Wire the accepted threshold to the metric transforms
#     self._thr_box.value = self._clamp_thr01(self.last_threshold, default=0.5, ctx="thr_box")
#     # Inject threshold into the evaluator used for metrics
#     self._ensure_std_engine()

#     # Allow the step to know device & pinning
#     pin = bool(getattr(val_loader, "pin_memory", False))
#     setattr(self._std_engine.state, "device", device)
#     setattr(self._std_engine.state, "non_blocking", pin)
#     # Run once to populate metrics
#     self._std_engine.run(val_loader)
#     raw = dict(self._std_engine.state.metrics or {})
#     # Ingest metrics (scalars, vectors, confmat)
#     _ingest_cls_metric_map(cls_metrics, raw)
#     # record the threshold we used this epoch (post-clamp)
#     cls_metrics["threshold"] = float(self._thr_box.value)
#     cls_metrics["cal_thr"] = float(self._thr_box.value)

#     # (optional) a concise status line for debugging / audits
#     if base_rate is not None:
#         cls_metrics["base_rate"] = float(base_rate)

#     # Decision-health logging (gated)
#     if self.enable_decision_health and base_rate is not None and epoch >= int(self.health_warmup):
#         rate_tol = self._resolve_rate_tolerance()
#         pos_rate = float(cls_metrics.get("pos_rate", 0.0))
#         delta = abs(pos_rate - base_rate)
#         self.bad_count = self.bad_count + 1 if (delta > rate_tol) else 0
#         if not getattr(self, "_watchdog_active", False):
#             logger.info("[decision health] pos=%.4f base=%.4f Δ=%.4f tol=%.4f bad_now=%d",
#                         pos_rate, base_rate, delta, rate_tol, self.bad_count)

# # low-spread guard
# # If scores are almost constant, project into the pos-rate envelope and return quickly.
# if probs.size:
#     std = float(np.std(probs))
#     q10 = float(np.quantile(probs, 0.10))
#     q90 = float(np.quantile(probs, 0.90))
#     iqr = q90 - q10
#     pos_rate_seed = float((probs >= t_seed).mean())
#     # defaults chosen to be gentle; you can read from calibrator.cfg if present
#     cal_min_std = float(getattr(getattr(self.calibrator, "cfg", None), "cal_min_std", 1e-4))
#     cal_min_iqr = float(getattr(getattr(self.calibrator, "cfg", None), "cal_min_iqr", 1e-3))
#     if std < cal_min_std and iqr < cal_min_iqr:
#         if pos_rate_seed < self.thr_posrate_min or pos_rate_seed > self.thr_posrate_max:
#             target = self.thr_posrate_min if pos_rate_seed < self.thr_posrate_min else self.thr_posrate_max
#             t_dm = _rate_match_threshold(probs, float(target))
#             t_final = _bounded_step(
#                 float(t_seed), float(t_dm),
#                 max_delta=self._resolve_max_delta(),
#                 lo=float(self.thr_min), hi=float(self.thr_max)
#             )
#             # publish and continue to metrics emit
#             self.last_threshold = float(t_final)
#             self._thr_box.value = self._clamp_thr01(self.last_threshold, default=0.5, ctx="thr_box")
#             self._ensure_std_engine()
#             pin = bool(getattr(val_loader, "pin_memory", False))
#             setattr(self._std_engine.state, "device", device)
#             setattr(self._std_engine.state, "non_blocking", pin)
#             self._std_engine.run(val_loader)
#             raw = dict(self._std_engine.state.metrics or {})
#             _ingest_cls_metric_map(cls_metrics, raw)
#             cls_metrics["threshold"] = float(self._thr_box.value)
#             cls_metrics["cal_thr"] = float(self._thr_box.value)
#             if base_rate is not None:
#                 cls_metrics["base_rate"] = float(base_rate)
#             # (decision health logging unchanged below will still run)
#             # proceed to the segmentation block / combined metric as usual
#         else:
#             # already within envelope; keep seed
#             self.last_threshold = float(t_seed)
#             self._thr_box.value = float(t_seed)
#             self._ensure_std_engine()
#             pin = bool(getattr(val_loader, "pin_memory", False))
#             setattr(self._std_engine.state, "device", device)
#             setattr(self._std_engine.state, "non_blocking", pin)
#             self._std_engine.run(val_loader)
#             raw = dict(self._std_engine.state.metrics or {})
#             _ingest_cls_metric_map(cls_metrics, raw)
#             cls_metrics["threshold"] = float(self._thr_box.value)
#             cls_metrics["cal_thr"] = float(self._thr_box.value)
#             if base_rate is not None:
#                 cls_metrics["base_rate"] = float(base_rate)
#             # proceed as usual

# def _gather_scores_labels(self, model, val_loader, device) -> Tuple[np.ndarray, np.ndarray]:
#     all_scores, all_labels = [], []
#     for batch in val_loader:
#         x = batch["image"].to(device, non_blocking=True)
#         y = labels_to_1d_indices(batch.get("label", batch.get("y"))).cpu().numpy()

#         logits = self.classify_fn(model, x)
#         logits = extract_cls_logits_from_any(logits)
#         if not torch.is_tensor(logits):
#             logits = torch.as_tensor(logits)
#         # unified: BCE(1-logit) or CE(2+ logits)
#         s = positive_score_from_logits(logits, positive_index=self.pos_idx).view(-1)

#         all_scores.append(s.detach().cpu().numpy())
#         all_labels.append(y)

#     scores = np.concatenate(all_scores).astype(np.float32) if all_scores else np.zeros(0, dtype=np.float32)
#     labels = np.concatenate(all_labels).astype(np.int64) if all_labels else np.zeros(0, dtype=np.int64)
#     if scores.size and not np.isfinite(scores).all():
#         scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
#     return scores, labels

# # D) Guardrails & acceptance — reuse existing checks, but with 'probs'
# def _accept(t_try: float) -> bool:
#     if epoch < int(self.thr_warmup_epochs) or probs.size == 0:
#         return False
#     t_try = float(min(self.thr_max, max(self.thr_min, t_try)))
#     pr = _pos_rate(probs, t_try)
#     if not (self.thr_posrate_min <= pr <= self.thr_posrate_max):
#         return False
#     auc_val = _safe_auc(labels, probs)
#     if not (isinstance(auc_val, float) and math.isfinite(auc_val) and auc_val >= self.cal_auc_floor):
#         return False
#     if self.thr_min_tp > 0 or self.thr_min_tn > 0:
#         pred = (probs >= t_try).astype(np.int64)
#         tp = int(((labels == 1) & (pred == 1)).sum())
#         tn = int(((labels == 0) & (pred == 0)).sum())
#         if tp < self.thr_min_tp or tn < self.thr_min_tn:
#             return False
#     max_delta = self._resolve_max_delta()
#     if self.last_threshold is not None and abs(t_try - float(self.last_threshold)) > max_delta:
#         return False
#     return True

# accepted = _accept(t_cand)
# t_final = float(min(self.thr_max, max(self.thr_min, t_cand))) if accepted else float(t_seed)
