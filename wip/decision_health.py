# decision_health.py (refactored)
import logging
import numpy as np
import torch
from ignite.engine import Events

logger = logging.getLogger(__name__)


def attach_decision_health(
    trainer,
    evaluator,
    *,
    val_loader=None,
    num_classes: int,
    positive_index: int = 1,
    tol: float = 0.35,          # |pos_rate - gt_rate| tolerance
    need_k: int = 3,            # consecutive "bad" epochs to stop
    warmup: int = None,         # ← if None, will use trainer.state.thr_warmup_epochs or default 5
    metric_key: str = "cls_confmat",
    inject_metrics: bool = False,
    terminate: bool = True,
    min_samples: int = 200,     # skip logging if too few val samples captured
):
    """
    Refactored decision-health:
      - Runs AFTER each training epoch (trainer hook), not on evaluator hook.
      - Delays until warmup epochs have passed (uses trainer.state.thr_warmup_epochs if warmup is None).
      - Captures RAW LOGITS + targets by wrapping evaluator._process_function for a single val pass.
      - Computes pos_rate, gt_rate, delta, and separability stats: std/IQR of logits, per-class means (mu0/mu1).
    """
    # Idempotency: don't attach twice
    if getattr(trainer, "_dh_attached", False):
        return
    setattr(trainer, "_dh_attached", True)

    state = {"bad_streak": 0}

    @trainer.on(Events.EPOCH_COMPLETED)
    def _decision_health(_):
        # Resolve warmup
        epoch = int(getattr(getattr(trainer, "state", None), "epoch", 0))
        eff_warmup = (
            int(warmup)
            if warmup is not None
            else int(getattr(trainer.state, "thr_warmup_epochs", 5))
        )
        if epoch < eff_warmup:
            return

        # 1) Try to read confusion matrix from evaluator metrics (fast path)
        metrics = getattr(evaluator.state, "metrics", {}) or {}
        cm = None
        for k in (metric_key, f"val_{metric_key}", f"val/{metric_key}"):
            if k in metrics:
                cm = metrics[k]
                break

        # 2) Capture logits + labels in a single eval pass (for separability + pos_rate at current threshold)
        _vl = (
            val_loader or getattr(trainer.state, "val_loader", None) or getattr(evaluator, "data_loader", None) or getattr(getattr(evaluator.state, "dataloader", None), "loader", None)  # noqa E501
        )
        if _vl is None:
            logger.debug("[decision health] no val_loader available; skipping this epoch")
            return

        captured_logits, captured_y = [], []
        pf = evaluator._process_function

        def _wrap(engine_, batch):
            out = pf(engine_, batch)
            try:
                y_pred, y_true = out
            except Exception:
                # If evaluator returns just y_pred or a dict, adapt here
                y_pred, y_true = out, {}
            # Extract logits/labels (compatible with model output/target)
            lg = None
            if isinstance(y_pred, dict) and "cls_logits" in y_pred:
                lg = y_pred["cls_logits"]
            elif torch.is_tensor(y_pred):
                lg = y_pred
            if lg is not None:
                lg = lg.detach().float()
                if lg.ndim == 2 and lg.shape[-1] == 1:
                    lg = lg.squeeze(-1)
                elif lg.ndim == 2 and lg.shape[-1] == 2:
                    # collapse two-logit to single via pos-neg if binary
                    pos = int(positive_index)
                    neg = 1 - pos
                    lg = lg[:, pos] - lg[:, neg]
                captured_logits.append(lg.cpu().numpy().reshape(-1))

            lab = None
            if isinstance(y_true, dict):
                lab = y_true.get("label", y_true.get("y", None))
            elif torch.is_tensor(y_true):
                lab = y_true
            if lab is not None:
                lab = torch.as_tensor(lab).long().view(-1)
                captured_y.append(lab.cpu().numpy().reshape(-1))

            return out

        evaluator._process_function = _wrap
        try:
            # evaluator.run(val_loader)
            evaluator.run(_vl)
        finally:
            evaluator._process_function = pf

        if not captured_logits or not captured_y:
            return

        logits = np.concatenate(captured_logits)
        y = np.concatenate(captured_y)
        if y.size < min_samples:
            return

        # Separability stats on raw logits
        std_all = float(np.std(logits))
        q1, q3 = np.quantile(logits, [0.25, 0.75])
        iqr_all = float(q3 - q1)
        mu0 = float(np.mean(logits[y == 0])) if (y == 0).any() else float("nan")
        mu1 = float(np.mean(logits[y == 1])) if (y == 1).any() else float("nan")

        # Pos/GT rates and delta using confusion matrix if available; else derive from probs at current threshold
        if cm is not None:
            cm_t = torch.as_tensor(cm, dtype=torch.float32).detach().cpu()
            if cm_t.ndim == 2 and cm_t.shape[0] == num_classes and cm_t.shape[1] == num_classes:
                total = float(cm_t.sum().item())
                if total > 0:
                    pi = int(positive_index)
                    pred_pos = float(cm_t[:, pi].sum().item())
                    gt_pos = float(cm_t[pi, :].sum().item())
                    pos_rate = pred_pos / total
                    gt_rate = gt_pos / total
                else:
                    pos_rate = gt_rate = 0.0
            else:
                cm = None  # fallback to threshold-based computation below

        if cm is None:
            # derive from current threshold
            thr = float(getattr(trainer.state, "threshold", getattr(trainer.state, "cls_threshold", 0.5)))
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= thr).astype(np.int64)
            pos_rate = float(preds.mean())
            gt_rate = float(y.mean())

        delta = abs(pos_rate - gt_rate)
        bad = int(delta > float(tol))

        if inject_metrics:
            # attach to evaluator metrics namespace (common pattern)
            metrics = getattr(evaluator.state, "metrics", None)
            if isinstance(metrics, dict):
                metrics["val_pos_rate"] = pos_rate
                metrics["val_gt_pos_rate"] = gt_rate
                metrics["val_pos_rate_delta"] = delta
                metrics["val_logits_std"] = std_all
                metrics["val_logits_iqr"] = iqr_all
                metrics["val_logits_mu0"] = mu0
                metrics["val_logits_mu1"] = mu1

        prev = state["bad_streak"]
        state["bad_streak"] = (prev + 1) if bad else 0

        logger.info(
            "[decision health] pos=%.4f gt=%.4f Δ=%.4f | thr=%.3f std=%.4g iqr=%.4g mu0=%.4g mu1=%.4g | bad_prev=%d bad_now=%d",
            pos_rate,
            gt_rate,
            delta,
            float(getattr(trainer.state, "threshold", getattr(trainer.state, "cls_threshold", 0.5))),
            std_all,
            iqr_all,
            mu0,
            mu1,
            prev,
            state["bad_streak"],
        )

        if terminate and state["bad_streak"] >= int(need_k):
            logger.warning("[decision health] persistent decision collapse (k=%d). Terminating.", state["bad_streak"])
            trainer.terminate()

# import logging
# import torch
# from ignite.engine import Events

# logger = logging.getLogger(__name__)


# def attach_decision_health(
#     trainer, evaluator, *,
#     num_classes: int,
#     positive_index: int = 1,
#     tol: float = 0.35,           # |pos_rate - gt_rate| tolerance
#     need_k: int = 3,             # consecutive "bad" epochs to stop
#     warmup: int = 2,             # ignore bad epochs before this
#     metric_key: str = "cls_confmat",
#     inject_metrics: bool = False,
#     terminate: bool = True,
# ):
#     # Idempotency: don't attach twice
#     if getattr(trainer, "_dh_attached", False):
#         return
#     setattr(trainer, "_dh_attached", True)

#     state = {"bad_streak": 0}

#     @evaluator.on(Events.EPOCH_COMPLETED)
#     def _decision_health(engine):
#         metrics = getattr(engine.state, "metrics", {}) or {}
#         cm = None
#         # Safe lookup without boolean ops on tensors
#         for k in (metric_key, f"val_{metric_key}", f"val/{metric_key}"):
#             if k in metrics:
#                 cm = metrics[k]
#                 break
#         if cm is None:
#             return

#         cm_t = torch.as_tensor(cm, dtype=torch.float32).detach().cpu()
#         if cm_t.ndim != 2 or cm_t.shape[0] != num_classes or cm_t.shape[1] != num_classes:
#             return

#         total = float(cm_t.sum().item())
#         if total <= 0:
#             return

#         pi = int(positive_index)
#         # predicted positives are the *column* sum for the positive class
#         pred_pos = float(cm_t[:, pi].sum().item())
#         # ground-truth positives are the *row* sum for the positive class
#         gt_pos = float(cm_t[pi, :].sum().item())

#         pos_rate = pred_pos / total
#         gt_rate = gt_pos / total
#         delta = abs(pos_rate - gt_rate)
#         bad = int(delta > float(tol))

#         if inject_metrics:
#             # names are *val_* when we attach to the std val engine
#             metrics["val_pos_rate"] = pos_rate
#             metrics["val_gt_pos_rate"] = gt_rate
#             metrics["val_pos_rate_delta"] = delta

#         prev = state["bad_streak"]
#         state["bad_streak"] = (prev + 1) if bad else 0

#         epoch = int(getattr(getattr(trainer, "state", None), "epoch", 0))
#         # Warmup gate avoids noisy first epochs and de-duplicates logs (trainer-only)
#         if epoch >= int(warmup):
#             logger.info("[decision health] pos=%.4f gt=%.4f Δ=%.4f bad_prev=%d bad_now=%d",
#                         pos_rate, gt_rate, delta, prev, state["bad_streak"])

#             if terminate and state["bad_streak"] >= int(need_k):
#                 logger.warning("[decision health] persistent decision collapse (k=%d). Terminating.", state["bad_streak"])
#                 trainer.terminate()
