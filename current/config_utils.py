# config_utils.py
from __future__ import annotations
from typing import Any, Dict, Mapping  # Tuple
# from types import SimpleNamespace
import copy
import yaml
from ast import literal_eval


# tiny dot-access dict
class AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


# helpers
def _flatten_nested_config(nested: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Flattens the new nested config.yaml into flat keys that the code expects.
    Only pulls the known sections/keys we care about; ignores extras.
    """
    out: Dict[str, Any] = {}

    # experiment
    exp = nested.get("experiment", {})
    out["architecture"] = exp.get("architecture")
    out["task"] = exp.get("task")
    out["run_cap"] = exp.get("run_cap")

    # execution
    exe = nested.get("execution", {})
    out["epochs"] = exe.get("epochs")
    out["num_workers"] = exe.get("num_workers")
    out["debug"] = exe.get("debug")
    out["debug_transforms"] = exe.get("debug_transforms")
    out["batch_size"] = exe.get("batch_size")

    # data
    data = nested.get("data", {})
    out["in_channels"] = data.get("in_channels")
    out["input_shape"] = data.get("input_shape")
    out["num_classes"] = data.get("num_classes")

    # metrics
    metrics = nested.get("metrics", {})
    out["multi_weight"] = metrics.get("multi_weight")

    # optim
    optim = nested.get("optim", {})
    out["optimizer"] = optim.get("optimizer")
    out["lr"] = optim.get("lr")
    out["weight_decay"] = optim.get("weight_decay")

    # regularization
    reg = nested.get("regularization", {})
    out["dropout_rate"] = reg.get("dropout_rate")

    # scheduler
    sch = nested.get("scheduler", {})
    out["lr_strategy"] = sch.get("lr_strategy")
    # cosine
    out["T_max"] = sch.get("T_max")
    out["eta_min"] = sch.get("eta_min")
    # onecycle
    out["max_lr"] = sch.get("max_lr")
    out["pct_start"] = sch.get("pct_start")
    out["div_factor"] = sch.get("div_factor")
    out["final_div_factor"] = sch.get("final_div_factor")
    # plateau
    out["monitor"] = sch.get("monitor")
    out["monitor_mode"] = sch.get("monitor_mode")
    out["patience"] = sch.get("patience")
    out["factor"] = sch.get("factor")
    # warmup + cosine
    out["warmup_epochs"] = sch.get("warmup_epochs")
    out["warmup_start_factor"] = sch.get("warmup_start_factor")

    # loss
    loss = nested.get("loss", {})
    out["alpha"] = loss.get("alpha")
    out["beta"] = loss.get("beta")

    # early stop
    es = nested.get("early_stop", None)
    if es is not None:
        out["early_stop"] = es

    # any other top-level keys
    if "metadata_csv" in nested:
        out["metadata_csv"] = nested["metadata_csv"]

    # prune None values; sweep overrides will fill missing
    return {k: v for k, v in out.items() if v is not None}


def _to_tuple(val, typ=float):
    if isinstance(val, str):
        s = val.strip("()[] ").replace(" ", "")
        if not s:
            return tuple()
        return tuple(typ(x) for x in s.split(","))
    if isinstance(val, (list, tuple)):
        return tuple(typ(x) for x in val)
    return val


def _coerce_types(cfg: Dict[str, Any]) -> None:
    # ints
    for k in ("epochs", "num_workers", "batch_size", "num_classes",
              "T_max", "patience", "warmup_epochs"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])

    # floats
    for k in ("lr", "weight_decay", "dropout_rate", "eta_min", "max_lr",
              "pct_start", "div_factor", "final_div_factor",
              "factor", "warmup_start_factor", "alpha", "beta", "multi_weight"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])

    # tuples
    if "input_shape" in cfg and cfg["input_shape"] is not None:
        cfg["input_shape"] = _to_tuple(cfg["input_shape"], int)

    # strings lowercased where appropriate
    for k in ("optimizer", "lr_strategy", "monitor_mode", "architecture", "task"):
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = cfg[k].lower() if k != "architecture" else cfg[k]

    def _parse_seq(val, cast):
        if val is None:
            return None
        if isinstance(val, str):
            try:
                val = literal_eval(val)  # turns "[2000, 600]" -> [2000, 600]
            except Exception:
                return None
        if isinstance(val, (list, tuple)):
            return tuple(cast(x) for x in val)
        return None

    cfg["class_counts"] = _parse_seq(cfg.get("class_counts"), int) or cfg.get("class_counts")
    cfg["class_weights"] = _parse_seq(cfg.get("class_weights"), float) or cfg.get("class_weights")


def _apply_defaults(cfg: Dict[str, Any]) -> None:
    # sensible defaults
    defaults = {
        "architecture": "multitask_unet",
        "task": "multitask",
        "run_cap": 20,

        "epochs": 40,
        "num_workers": 8,
        "debug": False,
        "debug_transforms": False,

        "in_channels": 1,
        "input_shape": (256, 256),
        "num_classes": 2,

        "multi_weight": 0.65,

        "optimizer": "adamw",
        "lr": 2e-4,
        "weight_decay": 5e-5,

        "dropout_rate": 0.20,

        "lr_strategy": "cosine",
        "T_max": None,  # if None, code can fall back to epochs
        "eta_min": 0.0,

        "max_lr": None,
        "pct_start": 0.3,
        "div_factor": 25.0,
        "final_div_factor": 1e4,

        "monitor": "val_loss",
        "monitor_mode": "min",
        "patience": 3,
        "factor": 0.5,

        "warmup_epochs": 3,
        "warmup_start_factor": 0.1,

        "alpha": 1.0,
        "beta": 1.0,

        "metadata_csv": "../data/processed/cbis_ddsm_metadata_full.csv",

        "early_stop": {
            "patience": 10,
            "min_delta": 0.001,
            "metric": "val_auc",
            "mode": "max",
        },
    }
    # inject defaults where missing
    for k, v in defaults.items():
        if k not in cfg or cfg[k] is None:
            # deep-merge early_stop dict
            if k == "early_stop" and isinstance(v, dict):
                es = copy.deepcopy(v)
                user_es = cfg.get("early_stop") or {}
                es.update({kk: user_es.get(kk, vv) for kk, vv in v.items()})
                cfg["early_stop"] = es
            else:
                cfg[k] = v


def _normalize_back_compat(cfg: Dict[str, Any]) -> None:
    """
    Map old sweep names to the new canonical keys.
    - base_learning_rate * lr_multiplier -> lr
    - l2_reg -> weight_decay
    - dropout -> dropout_rate
    """
    if "lr" not in cfg or cfg.get("lr") is None:
        base = cfg.get("base_learning_rate")
        mult = cfg.get("lr_multiplier")
        if base is not None and mult is not None:
            try:
                cfg["lr"] = float(base) * float(mult)
            except Exception:
                pass
        elif base is not None:
            cfg["lr"] = float(base)

    if "weight_decay" not in cfg and "l2_reg" in cfg:
        try:
            cfg["weight_decay"] = float(cfg["l2_reg"])
        except Exception:
            pass

    if "dropout_rate" not in cfg and "dropout" in cfg:
        try:
            cfg["dropout_rate"] = float(cfg["dropout"])
        except Exception:
            pass


def _validate_required(cfg: Dict[str, Any]) -> None:
    missing = [k for k in ("batch_size", "epochs") if cfg.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # basic scheduler sanity
    strat = (cfg.get("lr_strategy") or "none").lower()
    if strat == "onecycle":
        if cfg.get("max_lr") is None:
            raise ValueError("OneCycle requires `max_lr` (sweep or config).")
    if strat == "plateau":
        if cfg.get("monitor") is None:
            raise ValueError("Plateau scheduler requires `monitor` (e.g., 'val_loss' or 'val_auc').")


def load_and_validate_config(
    wandb_config: Mapping[str, Any] | None,
    base_config_path: str | None = None,
) -> AttrDict:
    """
    Load base config (new nested YAML), overlay wandb sweep overrides (flat),
    normalize names, apply defaults, coerce types, and validate.
    Returns dot-accessible AttrDict so existing getattr(...) call sites keep working.
    """
    cfg: Dict[str, Any] = {}

    # Load base config.yaml
    if base_config_path:
        with open(base_config_path, "r") as f:
            nested = yaml.safe_load(f) or {}
        cfg.update(_flatten_nested_config(nested))

    # Overlay wandb config (sweep) — takes precedence
    if wandb_config is not None:
        cfg.update(dict(wandb_config))

    # Back-compat mapping (old → new names)
    _normalize_back_compat(cfg)

    # Coerce types & apply defaults
    _coerce_types(cfg)
    _apply_defaults(cfg)

    # Early-stop sub-dict coercion
    if "early_stop" in cfg and isinstance(cfg["early_stop"], Mapping):
        es = dict(cfg["early_stop"])
        es["patience"] = int(es.get("patience", 10))
        es["min_delta"] = float(es.get("min_delta", 0.001))
        es["metric"] = str(es.get("metric", "val_auc"))
        es["mode"] = str(es.get("mode", "max"))
        cfg["early_stop"] = AttrDict(es)

    # Final validation
    _validate_required(cfg)

    # Return dot-access config (preserves dict semantics too)
    return AttrDict(cfg)


def print_config(config):
    print("Experiment Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
