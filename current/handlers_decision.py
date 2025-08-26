# handlers_decision.py
import torch
from ignite.engine import Events


def _to_pos_probs(logits):
    # works for 2-logit softmax or single-logit BCE
    if logits.ndim == 2 and logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    # assume shape (N,) or (N,1) for single-logit
    return torch.sigmoid(logits.squeeze(-1))


def youden_threshold(p, y):
    # p, y on CPU
    # sort by score desc
    sort_idx = torch.argsort(p, descending=True)
    p_sorted = p[sort_idx]
    y_sorted = y[sort_idx].float()

    P = y.sum().item()
    N = (y.numel() - P)
    if P == 0 or N == 0:
        return 0.5  # degenerate, fall back

    # cumulative positives/negatives as we sweep threshold downward
    cum_pos = torch.cumsum(y_sorted, dim=0)                # TP at each cut
    cum_neg = torch.cumsum(1.0 - y_sorted, dim=0)          # FP at each cut

    TPR = cum_pos / P                                      # sensitivity
    FPR = cum_neg / N
    J = TPR - FPR                                          # Youden’s J
    k = int(torch.argmax(J).item())

    # threshold is the score at index k (between p_k and next)
    thr = float(p_sorted[k].item())
    return thr


def match_rate_threshold(p, target_rate):
    # choose threshold so predicted positive fraction ≈ target_rate
    q = 1.0 - float(target_rate)
    q = min(max(q, 0.0), 1.0)
    return float(torch.quantile(p, q).item())


def attach_calibrated_decision(evaluator, method="youden", use_rate_match_warmup_epochs=2):
    """
    Adds a pass that computes a calibrated threshold from the current
    validation epoch and writes calibrated metrics into evaluator.state.metrics:
      - 'val/thr_cal'
      - 'val/pos_rate_cal'
      - 'val/acc_cal', 'val/prec_cal_0', 'val/prec_cal_1',
        'val/recall_cal_0', 'val/recall_cal_1'
    """

    @evaluator.on(Events.STARTED)
    def _start_epoch(_):
        evaluator.state._calib_p = []
        evaluator.state._calib_y = []

    @evaluator.on(Events.ITERATION_COMPLETED)
    def _gather_preds(engine):
        out = engine.state.output
        # Accept either mapping {'y_pred','y'} or tuple (logits, labels)
        if isinstance(out, dict):
            logits, y = out["y_pred"], out["y"]
        else:
            logits, y = out
        p = _to_pos_probs(logits).detach().float().cpu()
        evaluator.state._calib_p.append(p)
        evaluator.state._calib_y.append(y.detach().long().cpu())

    @evaluator.on(Events.COMPLETED)
    def _compute_and_log(engine):
        p = torch.cat(evaluator.state._calib_p, dim=0)
        y = torch.cat(evaluator.state._calib_y, dim=0)

        base_rate = float(y.float().mean().item())

        # pick a threshold
        if engine.state.epoch <= use_rate_match_warmup_epochs:
            thr = match_rate_threshold(p, target_rate=base_rate)
        else:
            thr = youden_threshold(p, y) if method == "youden" else match_rate_threshold(p, base_rate)

        yhat = (p >= thr).long()
        pos_rate = float(yhat.float().mean().item())

        # confusion + metrics
        tp = int(((yhat == 1) & (y == 1)).sum().item())
        tn = int(((yhat == 0) & (y == 0)).sum().item())
        fp = int(((yhat == 1) & (y == 0)).sum().item())
        fn = int(((yhat == 0) & (y == 1)).sum().item())
        acc = (tp + tn) / max(len(y), 1)

        prec0 = tn / max(tn + fn, 1)  # precision for class 0 (pred=0 treated as "positive" for class 0)
        prec1 = tp / max(tp + fp, 1)
        rec0 = tn / max(tn + fp, 1)  # recall for class 0
        rec1 = tp / max(tp + fn, 1)

        # stash alongside  existing metrics —  logger will pick these up
        m = engine.state.metrics
        m["val/thr_cal"] = thr
        m["val/pos_rate_cal"] = pos_rate
        m["val/acc_cal"] = acc
        m["val/prec_cal_0"] = prec0
        m["val/prec_cal_1"] = prec1
        m["val/recall_cal_0"] = rec0
        m["val/recall_cal_1"] = rec1
