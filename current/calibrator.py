from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


@dataclass
class CalConfig:
    # Selection rule
    method: str = "youden"              # "youden" | "f1" | "rate_match"
    q_bounds: Tuple[float, float] = (0.10, 0.90)  # threshold search window (quantiles)
    min_tp: int = 10                    # guard for F1 selection
    bootstraps: int = 0                 # >0 → bootstrap pick() and take median

    # Stability
    warmup_epochs: int = 2              # fixed threshold during warmup
    init_threshold: float = 0.5         # warmup & fallback starting point
    ema_beta: float = 0.2               # EMA smoothing across epochs
    max_delta: float = 0.10             # trust-region per-epoch change clamp

    # Rate health
    rate_tol: float = 0.10              # |pos_rate - base_rate| tolerance band

    # Weak-score handling
    auc_floor: float = 0.52             # if AUC < auc_floor → fallback path
    fallback: str = "rate_match"        # "rate_match" | "keep_last"


class Calibrator:
    """
    Robust per-epoch threshold picker for binary decisions on noisy scores.
    Implements warmup, AUC-floor fallback, rate guard, smoothing, and trust region.
    """

    def __init__(self, cfg: CalConfig):
        self.cfg = cfg
        self.t_prev: float = float(cfg.init_threshold)

    # public API
    def pick(self, epoch: int, scores: np.ndarray, labels: np.ndarray, base_rate: float) -> float:
        scores = self._sanitize_scores(scores)
        labels = labels.astype(np.int64, copy=False)

        # Warmup: keep fixed threshold to avoid early thrash
        if epoch < self.cfg.warmup_epochs:
            return self._smooth(self.cfg.init_threshold)

        # AUC health check → choose selection strategy
        auc = self._safe_auc(labels, scores)
        weak = (np.isnan(auc) or auc < self.cfg.auc_floor)

        if weak:
            if self.cfg.fallback == "rate_match":
                raw = self._threshold_for_rate(scores, base_rate)
            else:  # "keep_last"
                raw = self.t_prev
        else:
            method = (self.cfg.method or "youden").lower()
            if method == "youden":
                raw = self._bootstrap(self._youden, scores, labels)
            elif method == "f1":
                raw = self._bootstrap(self._f1_with_guard, scores, labels)
            elif method == "rate_match":
                raw = self._threshold_for_rate(scores, base_rate)
            else:
                raw = self.cfg.init_threshold  # safe default

        # Rate guard: clamp predicted positive rate to [base ± tol]
        raw = self._apply_rate_guard(scores, raw, base_rate)

        # Smooth & trust region
        return self._smooth(self._trust_region(raw))

    # selection rules
    def _youden(self, scores: np.ndarray, labels: np.ndarray) -> float:
        lo_q, hi_q = self._clamped_qbounds()
        lo, hi = np.quantile(scores, [lo_q, hi_q])
        grid = np.unique(np.clip(scores, lo, hi))
        if grid.size > 512:
            grid = np.linspace(lo, hi, 512)

        order = np.argsort(scores, kind="mergesort")
        s_sorted = scores[order]
        y_sorted = labels[order]

        pos = (y_sorted == 1).astype(np.int64)
        neg = 1 - pos
        P = int(pos.sum())
        N = int(neg.sum())
        if P == 0 or N == 0:
            return self.t_prev

        cum_pos_rev = np.cumsum(pos[::-1])[::-1]
        cum_neg_rev = np.cumsum(neg[::-1])[::-1]

        idx = np.searchsorted(s_sorted, grid, side="left")
        idx = np.clip(idx, 0, len(s_sorted) - 1)

        TPR = cum_pos_rev[idx] / P
        FPR = cum_neg_rev[idx] / N
        youden = TPR - FPR
        return float(grid[int(np.argmax(youden))])

    def _f1_with_guard(self, scores: np.ndarray, labels: np.ndarray) -> float:
        lo_q, hi_q = self._clamped_qbounds()
        lo, hi = np.quantile(scores, [lo_q, hi_q])
        grid = np.linspace(lo, hi, 512)

        y = labels.astype(np.int64, copy=False)
        P = int((y == 1).sum())
        best_f1, best_t = -1.0, self.t_prev

        for t in grid:
            pred = (scores >= t)
            tp = int(((y == 1) & pred).sum())
            if tp < self.cfg.min_tp or P == 0:
                continue
            fp = int(((y == 0) & pred).sum())
            fn = int(((y == 1) & ~pred).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            denom = prec + rec
            f1 = 0.0 if denom == 0.0 else (2.0 * prec * rec / denom)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        return float(best_t)

    def _threshold_for_rate(self, scores: np.ndarray, desired_rate: float) -> float:
        """Inverse-CDF threshold: choose t so that P(score >= t) ≈ desired_rate."""
        q = float(np.clip(1.0 - desired_rate, 0.0, 1.0))
        return float(np.quantile(scores, q))

    def _bootstrap(
        self,
        pick_fn: Callable[[np.ndarray, np.ndarray], float],
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        if self.cfg.bootstraps <= 0:
            return float(pick_fn(scores, labels))
        n = len(scores)
        rng = np.random.default_rng(1234)
        Ts = [float(pick_fn(scores[rng.integers(0, n, n)], labels[rng.integers(0, n, n)]))
              for _ in range(self.cfg.bootstraps)]
        return float(np.median(Ts))

    # guards & smoothing
    def _apply_rate_guard(self, scores: np.ndarray, t_raw: float, base_rate: float) -> float:
        """Clamp predicted positive rate into [base ± tol] using inverse-CDF projection."""
        rate = float((scores >= t_raw).mean())
        lo = max(0.0, base_rate - self.cfg.rate_tol)
        hi = min(1.0, base_rate + self.cfg.rate_tol)

        if rate < lo:
            return self._threshold_for_rate(scores, lo)
        if rate > hi:
            return self._threshold_for_rate(scores, hi)
        return t_raw

    def _smooth(self, t_new: float) -> float:
        t = self.cfg.ema_beta * float(t_new) + (1.0 - self.cfg.ema_beta) * float(self.t_prev)
        self.t_prev = float(t)
        return self.t_prev

    def _trust_region(self, t_new: float) -> float:
        lo = float(self.t_prev) - self.cfg.max_delta
        hi = float(self.t_prev) + self.cfg.max_delta
        return float(np.clip(float(t_new), lo, hi))

    # misc helpers
    def _clamped_qbounds(self) -> Tuple[float, float]:
        lo, hi = self.cfg.q_bounds
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, 0.0, 1.0))
        if hi <= lo:
            hi = min(1.0, lo + 1e-3)
        return lo, hi

    @staticmethod
    def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
        if np.isfinite(scores).all():
            return scores.astype(np.float32, copy=False)
        return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)

    @staticmethod
    def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
        y = y_true.astype(np.int64, copy=False)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y, scores))
        except Exception:
            pass

        # Tie-safe Mann–Whitney AUC
        order = np.argsort(scores)
        s_sorted = scores[order]
        ranks = np.empty_like(s_sorted, dtype=np.float64)

        i = 0
        while i < len(s_sorted):
            j = i
            while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
                j += 1
            ranks[i:j + 1] = (i + j + 2) / 2.0  # 1-based average rank for ties
            i = j + 1

        r = np.empty_like(ranks)
        r[order] = ranks
        sum_pos = float(r[y == 1].sum())
        return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
