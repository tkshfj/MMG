# data_utils.py
from __future__ import annotations
import logging
import os
import ast
import pandas as pd
from typing import Optional, Sequence
import torch
from monai.data import CacheDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_transforms import make_train_transforms, make_val_transforms

logger = logging.getLogger(__name__)


_BINARY_ALIASES_POS = {"pos", "positive", "malignant", "cancer", "abnormal", "lesion", "true", "yes", "1"}
_BINARY_ALIASES_NEG = {"neg", "negative", "benign", "normal", "healthy", "false", "no", "0"}


def normalize_binary_labels_inplace(
        df: pd.DataFrame, *, label_col: str = "label", label_map: dict | None = None, invert: bool = False, verbose: bool = True) -> dict:
    """
    Ensures df[label_col] ∈ {0,1}. Returns the mapping used.
    If label_map is None, infer it from the unique values.
    If invert=True, flips 0<->1 after mapping.
    """
    if label_col not in df.columns:
        raise KeyError(f"normalize_binary_labels_inplace: column '{label_col}' not found.")

    # Convert to a simpler python type first (avoid pandas NA weirdness)
    raw = df[label_col].astype("object").tolist()
    uniq = set(raw)

    def _infer_binary_mapping(unique_vals) -> dict:
        """
        Infer a mapping old_value -> {0,1}.
        Handles common patterns: {0,1}, {-1,1}, {1,2}, {0,255}, strings like 'benign'/'malignant', 'yes'/'no', etc.
        Raises if >2 unique labels.
        """
        if len(unique_vals) > 2:
            raise ValueError(f"Expected binary labels but found {len(unique_vals)} unique values: {unique_vals}")

        vals = sorted(unique_vals)
        svals = {str(v).strip().lower() for v in vals}

        # Already 0/1
        if set(vals) == {0, 1}:
            return {0: 0, 1: 1}
        # Common numeric encodings
        if set(vals) == {-1, 1}:
            return {-1: 0, 1: 1}
        if set(vals) == {1, 2}:
            return {1: 0, 2: 1}
        if set(vals) == {0, 255}:
            return {0: 0, 255: 1}

        # String-ish encodings
        if any(v in _BINARY_ALIASES_POS for v in svals) or any(v in _BINARY_ALIASES_NEG for v in svals):
            pos = next((v for v in vals if str(v).strip().lower() in _BINARY_ALIASES_POS), None)
            neg = next((v for v in vals if str(v).strip().lower() in _BINARY_ALIASES_NEG), None)
            if pos is not None and neg is not None:
                return {neg: 0, pos: 1}

        # Fallback: map smaller to 0, larger to 1
        if len(vals) == 2:
            return {vals[0]: 0, vals[1]: 1}

        raise ValueError(f"Could not infer binary mapping from values: {vals}")

    mapping = label_map or _infer_binary_mapping(uniq)
    df[label_col] = [mapping.get(v, v) for v in raw]
    # Ensure ints
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    # Invert if requested
    if invert:
        df[label_col] = 1 - df[label_col]

    if verbose:
        base = float(df[label_col].mean()) if len(df) else 0.0
        logger.info(f"[labels] normalized to 0/1 with mapping={mapping}, invert={invert}, base_pos_rate={base:.4f}")

    # Return the *final* mapping to {0,1} (including invert info)
    final_map = {k: (1 - v if invert else v) for k, v in mapping.items()}
    return final_map


def make_weighted_sampler_binary(
    targets: Sequence[int] | torch.Tensor,
    *,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Binary case (labels in {0,1}): per-sample weights = neg_weight for y=0, pos_weight for y=1.
    """
    y = torch.as_tensor(list(targets) if not torch.is_tensor(targets) else targets).long().view(-1)
    w = torch.where(y == 1, torch.tensor(float(pos_weight)), torch.tensor(float(neg_weight)))
    return WeightedRandomSampler(weights=w.double(), num_samples=(len(y) if num_samples is None else int(num_samples)), replacement=True)


def make_weighted_sampler_multiclass(
    targets: Sequence[int] | torch.Tensor,
    *,
    class_weights: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Multi-class: per-sample weight = class_weights[y].
    If class_weights is None, defaults to inverse-frequency (mean-normalized).
    """
    y = torch.as_tensor(list(targets) if not torch.is_tensor(targets) else targets).long().view(-1)
    C = int(num_classes) if num_classes is not None else int(y.max().item() + 1)
    if class_weights is None:
        counts = torch.bincount(y, minlength=C).clamp_min(1)
        inv = 1.0 / counts
        class_weights = inv * (inv.numel() / inv.sum())
    per_sample_w = class_weights.gather(0, y).double()
    return WeightedRandomSampler(weights=per_sample_w, num_samples=(len(y) if num_samples is None else int(num_samples)), replacement=True)


def _sanity_check_batch(batch, task, seg_target: str = "indices"):
    x = batch["image"]
    assert x.ndim == 4 and x.shape[1] == 1, f"image must be [N,1,H,W], got {tuple(x.shape)}"

    if task in ("segmentation", "multitask") and "mask" in batch:
        m = batch["mask"]
        if seg_target == "indices":
            assert m.ndim == 3, f"mask must be [N,H,W] for indices, got {tuple(m.shape)}"
            assert m.dtype in (torch.int64, torch.long), f"mask dtype must be long, got {m.dtype}"
        else:
            assert m.ndim == 4 and m.shape[1] >= 1, f"mask must be [N,C,H,W] for channels, got {tuple(m.shape)}"
            assert m.dtype.is_floating_point, f"mask should be float for channels, got {m.dtype}"


def filter_dataframe(df, require_mask=True, require_image=True, verbose=True):

    def _check_image_exists(row):
        """Return True if the image file exists."""
        img_path = row.get('image_path')
        if img_path and os.path.exists(img_path):
            return True
        fallback_path = os.path.join("../data/", row.get('full_mammo_path', ''))
        if fallback_path and os.path.exists(fallback_path):
            return True
        return False

    def _check_mask_exists(row):
        """Return True if at least one mask file exists."""
        # Try mask_paths
        mask_paths = row.get('mask_paths')
        paths = []
        if isinstance(mask_paths, str) and mask_paths.strip().startswith('['):
            try:
                paths = ast.literal_eval(mask_paths)
            except Exception:
                paths = [mask_paths]
        elif isinstance(mask_paths, list):
            paths = mask_paths
        elif isinstance(mask_paths, str):
            paths = [mask_paths]
        # Try roi_mask_path
        roi_path = row.get('roi_mask_path')
        if roi_path:
            paths.append(roi_path)
        # Check if any exists
        return any(os.path.exists(p) for p in paths if p)

    mask = pd.Series(True, index=df.index)
    if require_image:
        mask &= df.apply(_check_image_exists, axis=1)
    if require_mask:
        mask &= df.apply(_check_mask_exists, axis=1)
    filtered = df[mask].reset_index(drop=True)
    return filtered


def print_batch_debug(batch):
    print("[DEBUG] Batch keys:", batch.keys())
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape {tuple(v.shape)}, dtype {v.dtype}, sample: {v.flatten()[0:10].tolist()}")
        else:
            print(f"  {k}: {type(v)}")


# Turn a split DataFrame into MONAI items
def _row_to_item(row, *, task: str) -> dict:
    item = {"image": row.get("image_path") or os.path.join("../data/", row.get("full_mammo_path", ""))}
    if task in ("classification", "multitask") and "label" in row and row["label"] != "":
        item["label"] = int(row["label"])
    if task in ("segmentation", "multitask"):
        # prefer explicit list in CSV, else roi_mask_path
        mask_paths = row.get("mask_paths")
        if isinstance(mask_paths, str) and mask_paths.strip().startswith("["):
            import ast
            try:
                mask_paths = ast.literal_eval(mask_paths)
            except Exception:
                pass
        if isinstance(mask_paths, list) and len(mask_paths) > 0:
            item["mask"] = mask_paths[0] if len(mask_paths) == 1 else mask_paths  # LoadDICOMd via Lambdad handles both
        else:
            roi = row.get("roi_mask_path")
            if roi:
                item["mask"] = roi
    return item


def _df_to_items(df: pd.DataFrame, *, task: str) -> list[dict]:
    items = []
    for _, row in df.iterrows():
        it = _row_to_item(row, task=task)
        # keep only valid items (must have image path)
        if it.get("image") and os.path.exists(it["image"]):
            if task in ("segmentation", "multitask"):
                # ensure masks exist for seg tasks
                m = it.get("mask", None)
                if m is None:
                    continue
                if isinstance(m, list):
                    m = [p for p in m if os.path.exists(p)]
                    if not m:
                        continue
                    it["mask"] = m
                elif isinstance(m, str) and not os.path.exists(m):
                    continue
            items.append(it)
    return items


def build_dataloaders(
    metadata_csv,
    input_shape=(256, 256),
    batch_size=8,
    task="multitask",
    split=(0.7, 0.15, 0.15),
    num_workers=32,
    debug=False,
    pin_memory=False,
    multiprocessing_context=None,
    seg_target: str = "indices",
    num_classes: int = 2,
    sampling: str = "none",
    pos_weight: float | None = None,
    neg_weight: float = 1.0,
    class_weights: list[float] | None = None,
    class_counts: list[float] | None = None,
    seed: int = 42,
    invert_labels: bool = False,
    label_map: dict | None = None,
):
    df = pd.read_csv(metadata_csv)

    require_mask = (task in ['segmentation', 'multitask'])
    df = filter_dataframe(df, require_mask=require_mask, require_image=True, verbose=True)

    # normalize labels
    if task in ['classification', 'multitask'] and 'label' in df.columns:
        normalize_binary_labels_inplace(
            df, label_col="label", label_map=label_map, invert=bool(invert_labels), verbose=True
        )

    # split (same stratified logic)
    train_frac, val_frac, test_frac = split
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5
    temp_frac = val_frac + test_frac
    train_df, temp_df = train_test_split(
        df, test_size=temp_frac, random_state=seed,
        stratify=df['label'] if 'label' in df.columns else None
    )
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative_val_frac, random_state=seed,
        stratify=temp_df['label'] if 'label' in temp_df.columns else None
    )

    # dataframe -> items
    train_items = _df_to_items(train_df, task=task)
    val_items = _df_to_items(val_df, task=task)
    test_items = _df_to_items(test_df, task=task)

    for name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        if len(items) == 0:
            raise RuntimeError(f"{name}_items is empty after filtering. Check CSV paths/filters.")

    # transforms live in data_transforms.py
    train_tf = make_train_transforms(input_shape=input_shape, task=task, seg_target=seg_target, num_classes=num_classes)
    val_tf = make_val_transforms(input_shape=input_shape, task=task, seg_target=seg_target, num_classes=num_classes)

    # CacheDataset (cache_rate=0 means “IdentityDataset” with the same API)
    train_ds = CacheDataset(data=train_items, transform=train_tf, cache_rate=0.0)
    val_ds = CacheDataset(data=val_items, transform=val_tf, cache_rate=0.0)
    test_ds = CacheDataset(data=test_items, transform=val_tf, cache_rate=0.0)

    # loader kwargs
    common_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=bool(pin_memory))
    if num_workers and num_workers > 0:
        common_kwargs.update(persistent_workers=True, prefetch_factor=2)

    if isinstance(multiprocessing_context, str):
        import multiprocessing as mp
        multiprocessing_context = mp.get_context(multiprocessing_context)
    if multiprocessing_context is not None and num_workers and num_workers > 0:
        common_kwargs["multiprocessing_context"] = multiprocessing_context

    g = torch.Generator().manual_seed(int(seed))

    # weighted sampler
    train_sampler = None
    if sampling.lower() == "weighted" and task in ("classification", "multitask"):
        # y_train = torch.as_tensor([it["label"] for it in train_items], dtype=torch.long)
        lbls = [it["label"] for it in train_items if "label" in it]
        if not lbls:
            logger.warning("sampling='weighted' requested but no labels found in train_items; using shuffle.")
        else:
            y_train = torch.as_tensor(lbls, dtype=torch.long)
            if int(num_classes) == 2:
                n_pos = int((y_train == 1).sum())
                n_neg = int((y_train == 0).sum())
                default_pos_w = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
                pw = float(default_pos_w if pos_weight is None else pos_weight)
                train_sampler = make_weighted_sampler_binary(y_train, pos_weight=pw, neg_weight=float(neg_weight))
            else:
                cw = torch.as_tensor(class_weights, dtype=torch.float) if class_weights is not None else None
                train_sampler = make_weighted_sampler_multiclass(y_train, class_weights=cw, num_classes=int(num_classes))

    # construct loaders (default collate is fine now)
    def _make_loader(ds, *, shuffle: bool, drop_last: bool = False, sampler=None):
        kwargs = {**common_kwargs, "shuffle": (shuffle if sampler is None else False), "drop_last": drop_last}
        if sampler is not None:
            kwargs["sampler"] = sampler
        try:
            return DataLoader(ds, generator=g, **kwargs)
        except TypeError as e:  # fallback for older torch
            msg = str(e)
            for bad_key in ("multiprocessing_context", "prefetch_factor", "persistent_workers", "generator"):
                if bad_key in kwargs and f"unexpected keyword argument '{bad_key}'" in msg:
                    kwargs.pop(bad_key, None)
            return DataLoader(ds, **kwargs)

    train_loader = _make_loader(train_ds, shuffle=True, drop_last=True, sampler=train_sampler)
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    if debug:
        batch = next(iter(val_loader))
        _sanity_check_batch(batch, task, seg_target=seg_target)
        print_batch_debug(batch)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    return train_loader, val_loader, test_loader
