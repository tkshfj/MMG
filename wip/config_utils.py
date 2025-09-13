# config_utils.py
from __future__ import annotations
from typing import Any, Dict, Mapping
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
    # allow nested thresholds if provided
    out["cls_decision"] = metrics.get("cls_decision")
    out["cls_threshold"] = metrics.get("cls_threshold")
    out["seg_threshold"] = metrics.get("seg_threshold")
    out["positive_index"] = metrics.get("positive_index")

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
              "T_max", "patience", "warmup_epochs", "positive_index"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])

    # floats
    for k in ("lr", "weight_decay", "dropout_rate", "eta_min", "max_lr",
              "pct_start", "div_factor", "final_div_factor",
              "factor", "warmup_start_factor", "alpha", "beta",
              "multi_weight", "cls_threshold", "seg_threshold"):
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])

    # tuples
    if "input_shape" in cfg and cfg["input_shape"] is not None:
        cfg["input_shape"] = _to_tuple(cfg["input_shape"], int)

    # strings lowercased where appropriate
    for k in ("optimizer", "lr_strategy", "monitor_mode", "architecture", "task", "cls_decision"):
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

        # multitask aggregation
        "multi_weight": 0.65,

        # optim
        "optimizer": "adamw",
        "lr": 2e-4,
        "weight_decay": 5e-5,

        # regularization
        "dropout_rate": 0.20,

        # scheduler
        "lr_strategy": "cosine",
        "T_max": None,  # if None, code can fall back to epochs
        "eta_min": 0.0,

        "max_lr": None,
        "pct_start": 0.3,
        "div_factor": 25.0,
        "final_div_factor": 1e4,

        "monitor": "loss",
        "monitor_mode": "min",
        "patience": 3,
        "factor": 0.5,

        "warmup_epochs": 3,
        "warmup_start_factor": 0.1,

        # loss mixing
        "alpha": 1.0,
        "beta": 1.0,

        # IO
        "metadata_csv": "../data/processed/cbis_ddsm_metadata_full.csv",

        # early stop
        "early_stop": {
            "patience": 10,
            "min_delta": 0.001,
            "metric": "val_auc",
            "mode": "max",
        },

        # classification/seg decisions (thresholdable defaults)
        "cls_decision": "threshold",
        "positive_index": 1,
        "cls_threshold": 0.5,
        "seg_threshold": 0.5,
    }
    # inject defaults where missing
    for k, v in defaults.items():
        if k not in cfg or cfg[k] is None:
            if k == "early_stop" and isinstance(v, dict):
                es = copy.deepcopy(v)
                user_es = cfg.get("early_stop") or {}
                es.update({kk: user_es.get(kk, vv) for kk, vv in v.items()})
                cfg["early_stop"] = es
            else:
                cfg[k] = v


def normalize_lr_config(cfg: dict) -> dict:
    # alias lr_scheduler -> lr_strategy
    lr_strat = str(cfg.get("lr_strategy", cfg.get("lr_scheduler", "none"))).lower()
    cfg["lr_strategy"] = lr_strat

    # normalize metric key once (slashes → underscores)
    if "plateau_metric" in cfg and cfg["plateau_metric"]:
        cfg["plateau_metric"] = str(cfg["plateau_metric"]).replace("/", "_")

    # prune dead keys if scheduler is off
    if lr_strat in ("none", "", "off"):
        for k in ("plateau_metric", "plateau_mode", "patience", "factor",
                  "eta_min", "warmup_epochs", "warmup_start_factor", "T_max"):
            cfg.pop(k, None)
    return cfg


def _normalize_back_compat(cfg: Dict[str, Any]) -> None:
    """
    Map old sweep names to the new canonical keys.
    - base_learning_rate * lr_multiplier -> lr
    - l2_reg -> weight_decay
    - dropout -> dropout_rate
    - lr_scheduler -> lr_strategy
    - plateau_metric -> monitor
    - plateau_mode -> monitor_mode
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

    # aliases for scheduler/monitor keys
    if "lr_strategy" not in cfg and "lr_scheduler" in cfg:
        cfg["lr_strategy"] = cfg["lr_scheduler"]
    if "monitor" not in cfg and "plateau_metric" in cfg:
        cfg["monitor"] = cfg["plateau_metric"]
    if "monitor_mode" not in cfg and "plateau_mode" in cfg:
        cfg["monitor_mode"] = cfg["plateau_mode"]


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
            raise ValueError("Plateau scheduler requires `monitor` (e.g., 'loss' or 'val_auc').")


def _normalize_cls_loss_name(v: str | None) -> str:
    """Map common aliases to {'bce','ce','auto'}."""
    if not v:
        return "auto"
    s = str(v).strip().lower()
    # BCE-with-logits family
    if s in {"bcewithlogits", "bce_with_logits", "binary_cross_entropy_with_logits", "bce", "binary_bce"}:
        return "bce"
    # Cross-entropy family
    if s in {"ce", "crossentropy", "cross_entropy", "softmax_ce"}:
        return "ce"
    if s in {"auto", "default"}:
        return "auto"
    # Unknown → pass through (but lowercased) so we can still warn/fix
    return s


def _sync_head_and_loss(cfg: Dict[str, Any]) -> None:
    """
    Enforce a compatible (head_logits, cls_loss, binary flags) trio and keep posprob_mode=auto.

    Rules:
      - head_logits==1 → cls_loss='bce', binary_single_logit=True, binary_bce_from_two_logits=False
      - head_logits==2 → cls_loss='ce',  binary_single_logit=False
      - posprob_mode defaults to 'auto'
    """
    # 0) Normalize current values
    head_logits = cfg.get("head_logits", None)
    if head_logits is not None:
        try:
            head_logits = int(head_logits)
        except Exception:
            head_logits = None

    # Allow back-compat: infer head_logits from binary_single_logit when head_logits is absent
    if head_logits is None:
        bsl = cfg.get("binary_single_logit", None)
        if isinstance(bsl, bool):
            head_logits = 1 if bsl else 2
            cfg["head_logits"] = head_logits

    # Normalize/alias cls_loss
    cls_loss = _normalize_cls_loss_name(cfg.get("cls_loss"))

    # 1) Apply canonical combos if head_logits is known
    if head_logits == 1:
        if cls_loss not in {"bce", "auto"}:
            print("[WARN] head_logits=1 → switching cls_loss to 'bce'.")
        cfg["cls_loss"] = "bce"
        cfg["binary_single_logit"] = True
        cfg["binary_bce_from_two_logits"] = False  # never needed when head has a single logit

    elif head_logits == 2:
        if cls_loss not in {"ce", "auto"}:
            print("[WARN] head_logits=2 → switching cls_loss to 'ce'.")
        cfg["cls_loss"] = "ce"
        cfg["binary_single_logit"] = False
        # leave binary_bce_from_two_logits as-is (rarely used; keep explicit opt-in)

    else:
        # If still unknown, assume single-logit default (backward compatible)
        cfg.setdefault("head_logits", 1)
        if cls_loss == "auto":
            cfg["cls_loss"] = "bce"
        cfg.setdefault("binary_single_logit", True)
        cfg.setdefault("binary_bce_from_two_logits", False)

    # 2) Probability mapping should be automatic (keeps evaluator/ppcfg consistent)
    # cfg.setdefault("posprob_mode", "auto")

    # 3) Guard against contradictory user flags
    #    If user explicitly forced an incompatible pair, prefer head_logits and warn.
    if cfg["head_logits"] == 1 and _normalize_cls_loss_name(cfg.get("cls_loss")) != "bce":
        print("[WARN] Forcing cls_loss='bce' to match head_logits=1.")
        cfg["cls_loss"] = "bce"
    if cfg["head_logits"] == 2 and _normalize_cls_loss_name(cfg.get("cls_loss")) != "ce":
        print("[WARN] Forcing cls_loss='ce' to match head_logits=2.")
        cfg["cls_loss"] = "ce"


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

    # Back-compat/aliases first
    _normalize_back_compat(cfg)

    # Explicit aliases (belt-and-suspenders)
    cfg.setdefault("lr_strategy", cfg.get("lr_scheduler", "none"))
    cfg.setdefault("monitor", cfg.get("plateau_metric", cfg.get("monitor")))
    cfg.setdefault("monitor_mode", cfg.get("plateau_mode", cfg.get("monitor_mode")))

    # Coerce types & apply defaults
    _coerce_types(cfg)
    _apply_defaults(cfg)
    _sync_head_and_loss(cfg)

    # calibration defaults & coercion (kept simple; can expand if nested YAML exists)
    cal = cfg.setdefault("calibration", {})
    cal.setdefault("enabled", True)
    cal.setdefault("rate_tol", 0.10)
    cal["enabled"] = bool(cal["enabled"])
    try:
        cal["rate_tol"] = float(cal["rate_tol"])
    except Exception:
        raise ValueError("calibration.rate_tol must be a float")

    # dot-access for nested calibration
    if isinstance(cal, Mapping):
        cfg["calibration"] = AttrDict(dict(cal))

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
