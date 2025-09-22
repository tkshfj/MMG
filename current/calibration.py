# calibration.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

# Implements the API expected by evaluator_two_pass.TwoPassEvaluator
#   - has attributes: .cfg, .t_prev
#   - has method: pick(epoch, scores, labels, base_rate=None, auc=None) -> float
from protocols import CalibratorProtocol


@dataclass
class ThresholdCalibratorConfig:
    method: str = "youden"                    # {"youden","f1","rate_match"}
    ema_beta: float = 0.1
    init_threshold: float = 0.5
    max_delta: float = 0.10                   # clamp per-epoch movement
    q_bounds: Tuple[float, float] = (0.10, 0.90)   # search window quantiles
    rate_tolerance: float = 0.15              # +/- band around base rate
    warmup_epochs: int = 1                    # don’t calibrate before this epoch
    min_tp: int = 0                           # optional safety floor on TP count
    auc_floor: float = 0.0                    # optional: skip updates if AUC < floor
    bootstraps: int = 0                       # optional: bagging for stability

    # Extras read elsewhere (TwoPass evaluator accesses these if present)
    seg_threshold: float = 0.5
    decision_health_tol: float = 0.35
    decision_health_need_k: int = 3

    # Alias used by the evaluator: expose "rate_tol" in addition to "rate_tolerance"
    @property
    def rate_tol(self) -> float:
        return float(self.rate_tolerance)


class ThresholdCalibrator(CalibratorProtocol):
    """
    Stateful, epoch-by-epoch decision threshold calibrator.

    Use:
        cal = ThresholdCalibrator(cfg)
        t = cal.pick(epoch, scores, labels, base_rate=None, auc=None)
    """
    def __init__(self, cfg: ThresholdCalibratorConfig):
        self.cfg = cfg
        self._t = float(np.clip(cfg.init_threshold, 0.0, 1.0))

    # protocol-visible state
    @property
    def t_prev(self) -> float:
        return float(self._t)

    @t_prev.setter
    def t_prev(self, v: float) -> None:
        self._t = float(np.clip(v, 0.0, 1.0))

    # Convenience alias kept for external callers that used .threshold before
    @property
    def threshold(self) -> float:
        return float(self._t)

    # public API expected by TwoPassEvaluator
    def pick(
        self,
        epoch: int,
        scores: np.ndarray,   # positive-class probabilities in [0,1]
        labels: np.ndarray,
        base_rate: Optional[float] = None,
        auc: Optional[float] = None,   # optional; if not provided, no AUC gate is applied
    ) -> float:
        """
        Compute and update the decision threshold from positive-class probabilities,
        then apply per-epoch delta clamp and EMA smoothing. Returns the smoothed threshold.
        `scores` must be probabilities; logits must be converted by the caller.
        """
        # Warmup and AUC gating
        if epoch < int(self.cfg.warmup_epochs):
            return self._t
        if auc is not None and float(auc) < float(self.cfg.auc_floor):
            return self._t

        # s = np.asarray(scores, dtype=np.float64)
        # y = np.asarray(labels, dtype=np.int64)
        # if s.size == 0:
        # sanitize inputs and enforce probability range
        probs = np.asarray(scores, dtype=np.float64).reshape(-1)
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if probs.size == 0:
            return self._t

        # # sanitize degenerate / non-finite scores
        # if not np.isfinite(s).all():
        #     s = np.nan_to_num(s, nan=0.5, posinf=1.0, neginf=0.0)

        if base_rate is None:
            base_rate = float((y == 1).mean()) if y.size else 0.5
        base_rate = float(np.clip(base_rate, 0.0, 1.0))

        # compute raw threshold (optionally via bootstraps)
        # if int(self.cfg.bootstraps) > 0 and s.size > 1:
        if int(self.cfg.bootstraps) > 0 and probs.size > 1:
            ts = []
            # n = s.shape[0]
            n = probs.shape[0]
            for _ in range(int(self.cfg.bootstraps)):
                idx = np.random.randint(0, n, size=n)
                # ts.append(self._pick_threshold(s[idx], y[idx], base_rate))
                ts.append(self._pick_threshold(probs[idx], y[idx], base_rate))
            raw_t = float(np.median(ts))
        else:
            # raw_t = float(self._pick_threshold(s, y, base_rate))
            raw_t = float(self._pick_threshold(probs, y, base_rate))

        # clamp jump and EMA smooth
        prev = float(self._t)
        clipped = float(np.clip(raw_t, prev - self.cfg.max_delta, prev + self.cfg.max_delta))
        new_t = (1.0 - self.cfg.ema_beta) * prev + self.cfg.ema_beta * clipped
        self._t = float(np.clip(new_t, 0.0, 1.0))
        return self._t

    # internals
    def _pick_threshold(self, probs: np.ndarray, labels: np.ndarray, base_rate: float) -> float:
        """
        Choose a raw (pre-smoothed) threshold according to the configured method,
        constrained to the quantile window and the predicted-positive rate band.
        """
        qlo, qhi = np.quantile(probs, [self.cfg.q_bounds[0], self.cfg.q_bounds[1]])
        cand = np.unique(probs)
        cand = cand[(cand >= qlo) & (cand <= qhi)]
        if cand.size == 0:
            return self._t

        rate_tol = float(self.cfg.rate_tolerance)  # maintain current config name
        lo, hi = max(0.0, base_rate - rate_tol), min(1.0, base_rate + rate_tol)
        stride = max(1, cand.size // 512)

        def _tp_at(t):
            pred = (probs >= t)
            return int((pred & (labels == 1)).sum())

        method = str(self.cfg.method).lower()

        if method in {"rate_match", "rate", "quantile"}:
            # search within band for the closest predicted-positive rate
            best_t, best_gap = None, 1e9
            for t in cand[::stride]:
                p = float((probs >= t).mean())
                gap = 0.0 if (lo <= p <= hi) else min(abs(p - lo), abs(p - hi))
                if gap < best_gap:
                    best_gap, best_t = gap, float(t)
            t = float(best_t) if best_t is not None else float(np.quantile(probs, 1.0 - base_rate))

        elif method in {"youden", "j"}:
            P = max(int((labels == 1).sum()), 1)
            N = max(int((labels == 0).sum()), 1)
            best_j, best_t = -1e9, None
            for t in cand[::stride]:
                pred = (probs >= t)
                p = float(pred.mean())
                if not (lo <= p <= hi):
                    continue
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                j = (tp / P) - (fp / N)
                if j > best_j:
                    best_j, best_t = j, float(t)
            t = float(best_t) if best_t is not None else float(np.quantile(probs, 1.0 - base_rate))

        else:  # "f1" (default)
            best_f1, best_t = -1e9, None
            for t in cand[::stride]:
                pred = (probs >= t)
                p = float(pred.mean())
                if not (lo <= p <= hi):
                    continue
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                fn = int(((~pred) & (labels == 1)).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = (2.0 * prec * rec) / max(prec + rec, 1e-12)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            t = float(best_t) if best_t is not None else float(np.quantile(probs, 1.0 - base_rate))

        # enforce a minimum TP floor if requested (move left to increase positives)
        if int(self.cfg.min_tp) > 0 and _tp_at(t) < int(self.cfg.min_tp):
            qs = np.linspace(self.cfg.q_bounds[0], self.cfg.q_bounds[1], 100)
            for q in qs[::-1]:
                cand_t = float(np.quantile(probs, q))
                if _tp_at(cand_t) >= int(self.cfg.min_tp):
                    t = cand_t
                    break

        # final clip to the quantile window
        return float(np.clip(t, qlo, qhi))


def build_calibrator(cfg: dict) -> ThresholdCalibrator:
    q = cfg.get("cal_q_bounds", (0.10, 0.90))
    if not (isinstance(q, (list, tuple)) and len(q) == 2):
        q = (0.10, 0.90)
    c = ThresholdCalibratorConfig(
        method=str(cfg.get("calibration_method", "rate_match")),
        ema_beta=float(cfg.get("cal_ema_beta", 0.3)),
        init_threshold=float(cfg.get("cal_init_threshold", cfg.get("cls_threshold", 0.5))),
        max_delta=float(cfg.get("cal_max_delta", 0.10)),
        q_bounds=(float(q[0]), float(q[1])),
        rate_tolerance=float(cfg.get("cal_rate_tolerance", 0.10)),
        warmup_epochs=int(cfg.get("cal_warmup_epochs", 1)),
        min_tp=int(cfg.get("cal_min_tp", 0)),
        auc_floor=float(cfg.get("cal_auc_floor", 0.0)),
        bootstraps=int(cfg.get("cal_bootstraps", 0)),
        seg_threshold=float(cfg.get("seg_threshold", 0.5)),
        decision_health_tol=float(cfg.get("decision_health_tol", 0.35)),
        decision_health_need_k=int(cfg.get("decision_health_need_k", 3)),
    )
    return ThresholdCalibrator(c)
