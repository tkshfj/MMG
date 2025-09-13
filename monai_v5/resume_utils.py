# resume_utils.py
import os
import re
import glob
import torch
from typing import Dict, Optional


def _collect_ckpt_paths(dirname: str, prefix: str) -> Dict[str, Dict[int, str]]:
    """
    Returns mapping: { key -> {epoch -> path} } for keys in
    {"model","optimizer","lr_scheduler","trainer"}.
    """
    pat = re.compile(rf"^{re.escape(prefix)}_(model|optimizer|lr_scheduler|trainer)_(\d+)\.pt$")
    out: Dict[str, Dict[int, str]] = {"model": {}, "optimizer": {}, "lr_scheduler": {}, "trainer": {}}
    for p in glob.glob(os.path.join(dirname, f"{prefix}_*.pt")):
        fn = os.path.basename(p)
        m = pat.match(fn)
        if not m:
            continue
        key, ep = m.group(1), int(m.group(2))
        out[key][ep] = p
    return out


def find_latest_checkpoint_set(dirname: str, prefix: str) -> Optional[Dict[str, str]]:
    """
    Finds the largest epoch that exists for at least the model file, then
    returns the set of available paths for that epoch.
    """
    by_key = _collect_ckpt_paths(dirname, prefix)
    if not by_key["model"]:
        return None
    latest_ep = max(by_key["model"])
    result = {}
    for k in ("model", "optimizer", "lr_scheduler", "trainer"):
        if latest_ep in by_key[k]:
            result[k] = by_key[k][latest_ep]
    return result


def restore_training_state(
    dirname: str,
    prefix: str,
    *,
    model,
    optimizer=None,
    scheduler=None,
    trainer=None,
    map_location="cpu",
) -> Optional[int]:
    """
    Loads state_dicts if files for latest epoch exist.
    Returns the restored epoch (int) or None if nothing found.
    """
    ckpts = find_latest_checkpoint_set(dirname, prefix)
    if not ckpts:
        return None

    def _load(obj, path):
        state = torch.load(path, map_location=map_location)
        obj.load_state_dict(state)

    _load(model, ckpts["model"])
    if optimizer is not None and "optimizer" in ckpts:
        _load(optimizer, ckpts["optimizer"])
    if scheduler is not None and "lr_scheduler" in ckpts:
        _load(scheduler, ckpts["lr_scheduler"])
    if trainer is not None and "trainer" in ckpts:
        _load(trainer, ckpts["trainer"])

    # parse epoch number from any file we used
    ep = int(re.search(r"_(\d+)\.pt$", list(ckpts.values())[0]).group(1))
    return ep
