# decision_health.py
import logging
import torch
from ignite.engine import Events

logger = logging.getLogger(__name__)


def attach_decision_health(
    evaluator, trainer, *,
    num_classes: int,
    positive_index: int = 1,
    tol: float = 0.35,           # |pos_rate - gt_rate| tolerance
    need_k: int = 3,             # consecutive "bad" epochs to stop
    warmup: int = 2,             # ignore bad epochs before this
    metric_key: str = "cls_confmat",
    inject_metrics: bool = True,
    terminate: bool = True,
):
    state = {"bad_streak": 0}

    @evaluator.on(Events.EPOCH_COMPLETED)
    def _decision_health(engine):
        metrics = getattr(engine.state, "metrics", {}) or {}
        cm = None
        # Safe lookup without boolean ops on tensors
        for k in (metric_key, f"val_{metric_key}", f"val/{metric_key}"):
            if k in metrics:
                cm = metrics[k]
                break
        if cm is None:
            return

        cm_t = torch.as_tensor(cm, dtype=torch.float32).detach().cpu()
        if cm_t.ndim != 2 or cm_t.shape[0] != num_classes or cm_t.shape[1] != num_classes:
            return

        total = float(cm_t.sum().item())
        if total <= 0:
            return

        pi = int(positive_index)
        pred_pos = float(cm_t[:, pi].sum().item())
        gt_pos = float(cm_t[pi, :].sum().item())

        pos_rate = pred_pos / total
        gt_rate = gt_pos / total
        delta = abs(pos_rate - gt_rate)
        bad = int(delta > float(tol))

        if inject_metrics:
            # names are *val_* when we attach to the std val engine
            metrics["val_pos_rate"] = pos_rate
            metrics["val_gt_pos_rate"] = gt_rate

        prev = state["bad_streak"]
        state["bad_streak"] = (prev + 1) if bad else 0

        epoch = int(getattr(getattr(trainer, "state", None), "epoch", 0))

        if epoch >= int(warmup):
            logger.info("[decision health] pos=%.4f gt=%.4f Δ=%.4f bad_prev=%d bad_now=%d",
                        pos_rate, gt_rate, delta, prev, state["bad_streak"])

        if terminate and epoch >= int(warmup) and state["bad_streak"] >= int(need_k):
            logger.warning("[decision health] persistent decision collapse (k=%d). Terminating.", state["bad_streak"])
            trainer.terminate()
