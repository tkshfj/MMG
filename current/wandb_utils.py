# wandb_utils.py
import logging
from typing import Dict, Any, Literal, Optional, Sequence, Mapping
import numpy as np
import torch
from ignite.engine import Engine
from utils.safe import to_float_scalar

logger = logging.getLogger(__name__)


def make_wandb_logger(
    *,
    prefix: str = "train",
    source: Literal["output", "metrics"] = "output",
    trainer: Optional[Engine] = None,
    evaluator: Optional[Engine] = None,
    metric_names: Optional[Sequence[str]] = None,
    step_by: Literal["global", "epoch", "omit"] = "global",
    compute_multi: bool = True,
    debug: bool = False,
):
    ns = prefix.rstrip("/")
    pfx = (ns + "/") if ns else ""

    def _denamespace_any(key: str) -> str:
        """
        Remove exactly one leading 'val_'/'val/' or 'train_'/'train/' if present,
        regardless of our current target prefix.
        """
        for root in ("val", "train"):
            if key.startswith(root + "/"):
                return key[len(root) + 1:]
            if key.startswith(root + "_"):
                return key[len(root) + 1:]
        return key

    def _flatten(name: str, val: Any, out: Dict[str, float]):
        # scalars
        if isinstance(val, (int, float, np.floating, np.integer)):
            out[name] = float(val)
            return
        # torch tensors
        if torch.is_tensor(val):
            if val.numel() == 0:
                return
            if val.ndim == 0:
                out[name] = float(val.detach().cpu().item())
                return
            if val.ndim == 1:
                arr = val.detach().cpu().numpy()
                for i, x in enumerate(arr):
                    out[f"{name}/{i}"] = float(x)
                mean_val = float(arr.mean())
                out[name] = mean_val
                alias = name.replace("/", "_")
                if alias != name:
                    out[alias] = mean_val
                return
            f = to_float_scalar(val, strict=False)
            if f is not None:
                out[name] = f
            return
        # numpy arrays
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return
            if val.ndim == 0:
                out[name] = float(val.item())
                return
            if val.ndim == 1:
                for i, x in enumerate(val.tolist()):
                    out[f"{name}/{i}"] = float(x)
                mean_val = float(np.mean(val))
                out[name] = mean_val
                alias = name.replace("/", "_")
                if alias != name:
                    out[alias] = mean_val
                return
        # mappings
        if isinstance(val, Mapping):
            for k, v in val.items():
                _flatten(f"{name}/{k}", v, out)
                return
        # sequences
        if isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                _flatten(f"{name}/{i}", v, out)
                return

    def _pick_data_dict(engine: Engine) -> Dict[str, Any]:
        return dict((engine.state.output if source == "output" else engine.state.metrics) or {})

    def _handler(engine: Engine):
        epoch = int(getattr(evaluator.state if evaluator is not None else engine.state,
                            "trainer_epoch", getattr(engine.state, "epoch", 0)))
        global_step = int(getattr(trainer.state if trainer is not None else engine.state,
                                  "iteration", getattr(engine.state, "iteration", 0)))

        # 1) get dict, 2) strip any pre-existing train/val namespace, 3) (optionally) synthesize
        raw = _pick_data_dict(engine)
        data = {_denamespace_any(str(k)): v for k, v in raw.items()}

        if compute_multi and source == "metrics":
            try:
                import wandb
                mw = float(getattr(wandb.config, "multi_weight", 0.65))
            except Exception:
                mw = 0.65
            dice = data.get("dice")
            auc = data.get("auc")
            if ("multi" not in data) and (dice is not None) and (auc is not None):
                try:
                    data["multi"] = mw * float(to_float_scalar(dice, strict=False) or 0.0) + (1.0 - mw) * float(to_float_scalar(auc, strict=False) or 0.0)
                except Exception:
                    pass

        if metric_names:
            keep = set(metric_names) | {k.replace("_", "/") for k in metric_names} | {k.replace("/", "_") for k in metric_names}
            data = {k: v for k, v in data.items() if k in keep}

        flat: Dict[str, float] = {}
        for k, v in data.items():
            _flatten(k, v, flat)

        # Add our prefix exactly once
        payload = {(pfx + k if ns else k): v for k, v in flat.items()}
        payload["epoch"] = epoch
        payload["global_step"] = global_step

        step = global_step if step_by == "global" else (epoch if step_by == "epoch" else None)
        try:
            import wandb
            if debug:
                logger.info("wandb log [%s] step_by=%s step=%s keys=%d", prefix, step_by, step, len(payload))
            wandb.log(payload) if step is None else wandb.log(payload, step=step)
        except Exception as e:
            logger.warning("wandb.log failed: %s", e)

    return _handler
