# decision_health.py
import logging
import torch
from ignite.engine import Events

logger = logging.getLogger(__name__)


def attach_decision_health(
    evaluator, trainer, *,
    num_classes: int,
    positive_index: int = 1,
    tol: float = 0.35,    # |pos_rate - gt_rate| tolerance
    need_k: int = 3,      # consecutive "bad" epochs to stop
    warmup: int = 2,      # ignore bad epochs before this
    inject_metrics: bool = True,
):
    state = {"bad_streak": 0}

    @evaluator.on(Events.EPOCH_COMPLETED)
    def _decision_health(engine):
        cm = engine.state.metrics.get("val_cls_confmat")
        if cm is None:
            return

        cm_t = cm.detach().cpu().to(torch.float32) if torch.is_tensor(cm) else torch.as_tensor(cm, dtype=torch.float32)
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
            engine.state.metrics["val_pos_rate"] = pos_rate
            engine.state.metrics["val_gt_pos_rate"] = gt_rate

        prev = state["bad_streak"]
        state["bad_streak"] = (prev + 1) if bad else 0

        logger.info("[decision health] pos=%.4f gt=%.4f Δ=%.4f bad_prev=%d bad_now=%d",
                    pos_rate, gt_rate, delta, prev, state["bad_streak"])

        cur_epoch = int(getattr(trainer.state, "epoch", 0))
        if cur_epoch >= int(warmup) and state["bad_streak"] >= int(need_k):
            # FIXED: correct formatting order/types
            logger.warning("[decision health] persistent decision collapse (k=%d). %s", state["bad_streak"], "Terminating training to save time.")
            trainer.terminate()
