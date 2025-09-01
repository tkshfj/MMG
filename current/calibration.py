# calibration.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class ThresholdCalibratorConfig:
    method: str = "youden"            # {"youden","f1","rate_match"}
    ema_beta: float = 0.1
    init_threshold: float = 0.5
    max_delta: float = 0.10           # clamp per-epoch movement
    q_bounds: Tuple[float, float] = (0.10, 0.90)   # search window quantiles
    rate_tolerance: float = 0.15      # +/- band around base rate
    warmup_epochs: int = 1            # don’t calibrate before this epoch
    min_tp: int = 0                   # optional safety floor on TP count
    auc_floor: float = 0.0            # optional: skip updates if AUC < floor
    bootstraps: int = 0               # optional: bagging for stability


class ThresholdCalibrator:
    """
    Stateless computation + stateful smoothing for a per-epoch decision threshold.
    Call `fit(probs, labels, epoch, auc)` once per epoch to get a new smoothed threshold.
    """
    def __init__(self, cfg: ThresholdCalibratorConfig):
        self.cfg = cfg
        self._t = float(cfg.init_threshold)

    @property
    def threshold(self) -> float:
        return float(self._t)

    # ---- public API ----
    def fit(self, probs: np.ndarray, labels: np.ndarray, *, epoch: int, auc: Optional[float] = None) -> float:
        """
        Compute and update the threshold given probability scores & labels for the epoch.
        Returns the smoothed threshold to use this epoch.
        """
        # warm-up / gate by AUC if requested
        if epoch < int(self.cfg.warmup_epochs):
            return self._t
        if auc is not None and float(auc) < float(self.cfg.auc_floor):
            return self._t

        # compute raw threshold (possibly with bootstraps)
        if int(self.cfg.bootstraps) > 0:
            ts = []
            n = probs.shape[0]
            for _ in range(int(self.cfg.bootstraps)):
                idx = np.random.randint(0, n, size=n)
                ts.append(self._pick_threshold(probs[idx], labels[idx]))
            raw_t = float(np.median(ts))
        else:
            raw_t = self._pick_threshold(probs, labels)

        # clamp jump and EMA smooth
        prev = float(self._t)
        clipped = np.clip(raw_t, prev - self.cfg.max_delta, prev + self.cfg.max_delta)
        new_t = (1.0 - self.cfg.ema_beta) * prev + self.cfg.ema_beta * float(clipped)
        self._t = float(np.clip(new_t, 0.0, 1.0))
        return self._t

    # ---- internals ----
    def _pick_threshold(self, probs: np.ndarray, labels: np.ndarray) -> float:
        probs = probs.astype(np.float64, copy=False)
        labels = labels.astype(np.int32, copy=False)
        if probs.size == 0:
            return self._t

        qlo, qhi = np.quantile(probs, [self.cfg.q_bounds[0], self.cfg.q_bounds[1]])
        cand = np.unique(probs)
        cand = cand[(cand >= qlo) & (cand <= qhi)]
        if cand.size == 0:
            return self._t

        base = float(labels.mean()) if labels.size else 0.5
        lo, hi = max(0.0, base - self.cfg.rate_tolerance), min(1.0, base + self.cfg.rate_tolerance)
        stride = max(1, cand.size // 512)

        # optional TP floor utility
        def _tp_at(t):
            pred = (probs >= t)
            return int((pred & (labels == 1)).sum())

        if self.cfg.method.lower() in ("rate_match", "rate", "quantile"):
            t = float(np.quantile(probs, 1.0 - base))
        elif self.cfg.method.lower() in ("youden", "j"):
            P = max(int((labels == 1).sum()), 1)
            N = max(int((labels == 0).sum()), 1)
            best_j, best_t = -1.0, None
            for t in cand[::stride]:
                pred = (probs >= t)
                prate = float(pred.mean())
                if not (lo <= prate <= hi):
                    continue
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                j = (tp / P) - (fp / N)
                if j > best_j:
                    best_j, best_t = j, float(t)
            t = float(best_t) if best_t is not None else float(np.quantile(probs, 1.0 - base))
        else:  # F1 default
            best_f1, best_t = -1.0, None
            b2 = 1.0
            for t in cand[::stride]:
                pred = (probs >= t)
                prate = float(pred.mean())
                if not (lo <= prate <= hi):
                    continue
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                fn = int(((~pred) & (labels == 1)).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = (1 + b2) * prec * rec / max(b2 * prec + rec, 1e-12)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            t = float(best_t) if best_t is not None else float(np.quantile(probs, 1.0 - base))

        # enforce min_tp if requested (lower the threshold until satisfied)
        if int(self.cfg.min_tp) > 0 and _tp_at(t) < int(self.cfg.min_tp):
            # move t leftward across quantiles to increase positives
            qs = np.linspace(self.cfg.q_bounds[0], self.cfg.q_bounds[1], 100)
            for q in qs[::-1]:
                cand_t = float(np.quantile(probs, q))
                if _tp_at(cand_t) >= int(self.cfg.min_tp):
                    t = cand_t
                    break

        # clip to the search window
        return float(np.clip(t, qlo, qhi))
