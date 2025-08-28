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
            raise RuntimeError(f"{name}_items is empty after filtering. Check your CSV paths/filters.")

    # transforms live in data_transforms.py
    train_tf = make_train_transforms(input_shape=input_shape, task=task, seg_target=seg_target, num_classes=num_classes)
    val_tf = make_val_transforms(input_shape=input_shape, task=task, seg_target=seg_target, num_classes=num_classes)

    # CacheDataset (cache_rate=0 means “IdentityDataset” with the same API)
    train_ds = CacheDataset(data=train_items, transform=train_tf, cache_rate=0.0)
    val_ds = CacheDataset(data=val_items, transform=val_tf, cache_rate=0.0)
    test_ds = CacheDataset(data=test_items, transform=val_tf, cache_rate=0.0)

    # loader kwargs (unchanged)
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

# from __future__ import annotations
# import logging
# import os
# import ast
# import glob
# import numpy as np
# import pandas as pd
# import pydicom
# import torch
# from typing import Any, Optional, Sequence, Literal, Tuple
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# # from monai.transforms import Compose, EnsureTyped, AsDiscreted, RandFlipd, RandRotate90d, ScaleIntensityd, ToTensord, SqueezeDimd, Lambdad
# from monai.transforms import (
#     Compose, EnsureTyped, AsDiscreted, RandFlipd, RandRotate90d,
#     NormalizeIntensityd, ToTensord, SqueezeDimd, Lambdad, EnsureChannelFirstd
# )

# logging.basicConfig(
#     level=logging.INFO,  # Change to DEBUG or WARNING as appropriate
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )
# logger = logging.getLogger(__name__)


# EPSILON = 1e-8  # Small value to avoid division by zero
# Label normalization helpers

# # HxW -> 1xHxW; no-op if channel already present
# def add_channel_if_missing(x):
#     t = torch.as_tensor(x)
#     return t.unsqueeze(0) if t.ndim == 2 else t


# def extract_targets(
#     obj: Any,
#     *,
#     label_col: str = "label",
#     allow_iterate: bool = False,
# ) -> torch.Tensor:
#     """
#     Returns labels as a 1D int64 tensor from either:
#       - a pandas DataFrame (column `label_col`)
#       - a dataset object exposing .targets/.labels or .df[label_col]
#       - optionally iterates the dataset to collect labels if allow_iterate=True

#     Raises RuntimeError if labels cannot be found without iteration.
#     """
#     # Case 1: DataFrame
#     if isinstance(obj, pd.DataFrame):
#         if label_col not in obj.columns:
#             raise RuntimeError(f"extract_targets: DataFrame missing column '{label_col}'.")
#         return torch.as_tensor(obj[label_col].to_numpy()).long().view(-1)

#     # Case 2: dataset-style containers
#     for attr in ("targets", "labels"):
#         if hasattr(obj, attr):
#             return torch.as_tensor(getattr(obj, attr)).long().view(-1)

#     if hasattr(obj, "df") and isinstance(obj.df, pd.DataFrame) and label_col in obj.df.columns:
#         return torch.as_tensor(obj.df[label_col].to_numpy()).long().view(-1)

#     for attr in ("y", "labels_", "label_ids"):
#         if hasattr(obj, attr):
#             return torch.as_tensor(getattr(obj, attr)).long().view(-1)

#     # Optional: iterate dataset to extract labels (last resort)
#     if allow_iterate and hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
#         ys = []
#         for i in range(len(obj)):
#             sample = obj[i]
#             if isinstance(sample, dict) and label_col in sample:
#                 ys.append(int(sample[label_col]))
#             elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
#                 ys.append(int(sample[1]))  # assume (image, label, ...)
#         if ys:
#             return torch.as_tensor(ys).long().view(-1)

#     raise RuntimeError("extract_targets: could not find labels on dataset "
#                        "(expected DataFrame column, .targets/.labels, .df['label'], .y/.labels_ "
#                        "or enable allow_iterate=True).")


# # def extract_targets(dataset) -> torch.Tensor:
# #     """
# #     Try common places for labels without iterating the dataset.
# #     Falls back to dataset.df['label'] or dataset.labels if present.
# #     As a last resort, raises to avoid O(N) scan in the hot path.
# #     """
# #     # TorchVision-style
# #     if hasattr(dataset, "targets"):
# #         return torch.as_tensor(getattr(dataset, "targets")).long().view(-1)
# #     if hasattr(dataset, "labels"):
# #         return torch.as_tensor(getattr(dataset, "labels")).long().view(-1)
# #     # Pandas-backed custom datasets often expose df
# #     if hasattr(dataset, "df") and hasattr(dataset.df, "label"):
# #         return torch.as_tensor(dataset.df["label"].to_numpy()).long().view(-1)
# #     # MONAI / custom dict datasets: sometimes store y in .y or .labels_
# #     for attr in ("y", "labels_", "label_ids"):
# #         if hasattr(dataset, attr):
# #             return torch.as_tensor(getattr(dataset, attr)).long().view(-1)
# #     raise RuntimeError("extract_targets: could not find labels on dataset "
# #                        "(expected .targets / .labels / .df['label'] / .y / .labels_)")

# class MammoSegmentationDataset(Dataset):
#     """ MONAI/PyTorch Dataset for multitask learning. """
#     def __init__(self, df, root="../data/", input_shape=(256, 256), task='segmentation', transform=None, label_map: dict | None = None):
#         self.df = df
#         self.root = root
#         self.input_shape = input_shape
#         self.task = task
#         self.transform = transform
#         self.label_map = label_map or None

#     def resolve_image_path(self, row):
#         # Prefer image_path if it exists and file exists
#         img_path = row.get('image_path', None)
#         if img_path and os.path.exists(img_path):
#             return img_path
#         # Fallback: try constructing from root and full_mammo_path (should fail if UID-based)
#         fallback_path = os.path.join(self.root, row.get('full_mammo_path', ''))
#         if os.path.exists(fallback_path):
#             return fallback_path
#         # Last resort: search for a DICOM file in the corresponding folder using metadata
#         folder = os.path.dirname(img_path) if img_path else self.root
#         matches = glob.glob(os.path.join(folder, "*.dcm"))
#         if matches:
#             print(f"[WARNING] Used fallback for image: {matches[0]}")
#             return matches[0]
#         # raise FileNotFoundError(f"No image file found for {row}")
#         return None  # Instead of raising

#     def resolve_mask_paths(self, row):
#         # mask_paths can be a list or string representation of a list
#         mask_paths = row.get('mask_paths')
#         if isinstance(mask_paths, str):
#             try:
#                 mask_paths = ast.literal_eval(mask_paths)
#             except Exception:
#                 mask_paths = [mask_paths]
#         if isinstance(mask_paths, list):
#             existing = [p for p in mask_paths if os.path.exists(p)]
#             if existing:
#                 return existing
#         # Fallback: Try roi_mask_path column (not recommended if UIDs)
#         roi_mask = row.get('roi_mask_path')
#         if roi_mask and os.path.exists(roi_mask):
#             return [roi_mask]
#         # Try searching for any DICOM in the expected folder
#         if mask_paths and len(mask_paths) > 0:
#             folder = os.path.dirname(mask_paths[0])
#             matches = glob.glob(os.path.join(folder, "*.dcm"))
#             if matches:
#                 print(f"[WARNING] Used fallback for mask: {matches}")
#                 return matches
#         # raise FileNotFoundError(f"No mask file found for {row}")
#         logger.error(f"No mask file found for {row}")
#         return []

#     def load_dicom(self, path):
#         try:
#             dcm = pydicom.dcmread(path)
#             img = dcm.pixel_array.astype(np.float32)
#             # Rescale if provided
#             slope = float(getattr(dcm, "RescaleSlope", 1.0) or 1.0)
#             intercept = float(getattr(dcm, "RescaleIntercept", 0.0) or 0.0)
#             img = img * slope + intercept
#             # Invert if MONOCHROME1
#             if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
#                 img = img.max() - img
#             # img = (img - img.min()) / (img.max() - img.min() + EPSILON)
#         except Exception as e:
#             logger.warning(f"[DICOM ERROR] Failed to load {path}: {e}")
#             img = np.zeros(self.input_shape, dtype=np.float32)
#         return img

#     def load_and_merge_masks(self, mask_paths, shape):
#         import cv2
#         mask = np.zeros(shape, dtype=np.float32)
#         for mpath in mask_paths:
#             m = self.load_dicom(mpath)
#             if m.shape != mask.shape:
#                 m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
#             mask = np.maximum(mask, m)
#         mask = (mask > 0.5).astype(np.float32)
#         return mask

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         import cv2
#         row = self.df.iloc[idx]
#         # DEBUG print row
#         # print(f"[DEBUG __getitem__] idx={idx}, row={row.to_dict()}")
#         # print("[DEBUG __getitem__] keys:", row.keys())

#         # Image path
#         img_path = self.resolve_image_path(row)
#         if not img_path:
#             # logger.warning(f"Skipping sample idx={idx} (no image path): {row.to_dict()}")
#             raise IndexError(f"No valid image for idx={idx}")
#         img = self.load_dicom(img_path)
#         img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

#         if img.ndim == 2:
#             img = np.expand_dims(img, axis=0)          # [1, H, W]
#         image_tensor = torch.from_numpy(img).float()   # float32
#         # image_tensor = torch.from_numpy(img).float()  # HxW

#         # Prepare sample dict with image only
#         sample = {"image": image_tensor}

#         # classification / multitask label
#         if self.task in ['classification', 'multitask']:
#             label_value = row.get('label')
#             if label_value is not None and not (pd.isna(label_value) or label_value == ''):
#                 if self.label_map is not None and label_value in self.label_map:
#                     label_value = self.label_map[label_value]
#                 # ensure 0/1 int
#                 label_value = int(label_value)
#                 if label_value not in (0, 1):
#                     raise ValueError(f"Label must be binary 0/1 after normalization, got {label_value} for idx={idx}")
#                 sample["label"] = torch.tensor(label_value, dtype=torch.long)

#         # if self.task in ['classification', 'multitask']:
#         #     label_value = row.get('label')
#         #     if label_value is not None and not (pd.isna(label_value) or label_value == ''):
#         #         sample["label"] = torch.tensor(int(label_value), dtype=torch.long)

#         # segmentation / multitask mask
#         if self.task in ['segmentation', 'multitask']:
#             mask_paths = self.resolve_mask_paths(row)
#             if mask_paths:
#                 mask = self.load_and_merge_masks(mask_paths, img.shape[1:])
#                 mask_tensor = torch.from_numpy((mask > 0.5).astype(np.int64)).long()
#                 sample["mask"] = mask_tensor

#         # Apply transforms if present
#         if self.transform:
#             sample = self.transform(sample)
#             for k in list(sample.keys()):
#                 # Remove keys where the value is None or NaN
#                 if sample[k] is None:
#                     sample.pop(k)
#                 elif isinstance(sample[k], torch.Tensor) and sample[k].numel() == 0:
#                     sample.pop(k)

#         return sample


# def get_monai_transforms(
#     task: str = "multitask",
#     *,
#     seg_target: Literal["channels", "indices"] = "indices",
#     num_classes: int = 2,
#     binarize_threshold: float = 0.5,
#     aug_flip_prob: float = 0.5,
#     aug_rot90_prob: float = 0.5,
# ) -> Tuple[Compose, Compose]:
#     t = (task or "multitask").lower()
#     has_seg = t != "classification"
#     has_cls = t != "segmentation"

#     img_keys = ("image",)
#     mask_keys = ("mask",) if has_seg else tuple()
#     label_keys = ("label",) if has_cls else tuple()

#     # image pre
#     img_pre = [
#         EnsureTyped(keys=img_keys, data_type="tensor", allow_missing_keys=True),
#         EnsureChannelFirstd(keys=img_keys, channel_dim=0, allow_missing_keys=True),
#         NormalizeIntensityd(
#             keys=img_keys,
#             nonzero=True,
#             channel_wise=True,
#             allow_missing_keys=True,
#         ),
#         # (Optional) keep ToTensord if you rely on explicit dtype casting elsewhere
#         ToTensord(keys=img_keys, dtype=torch.float32, allow_missing_keys=True),

#         # EnsureTyped(keys=img_keys, data_type="tensor", allow_missing_keys=True),
#         # # EnsureChannelFirstd(keys=img_keys, channel_dim=0, allow_missing_keys=True),  # -> [C,H,W]
#         # ScaleIntensityd(keys=img_keys, allow_missing_keys=True),
#         # ToTensord(keys=img_keys, dtype=torch.float32, allow_missing_keys=True),
#     ]

#     # mask pre (keep channel so augs always see at least 2 spatial dims)
#     mask_pre = []
#     if mask_keys:
#         mask_pre = [
#             EnsureTyped(keys=mask_keys, data_type="tensor", allow_missing_keys=True),
#             # HxW -> 1xHxW (no-op if already has channel)
#             # EnsureChannelFirstd(keys=mask_keys, channel_dim=0, allow_missing_keys=True),
#             Lambdad(keys=mask_keys, func=add_channel_if_missing, allow_missing_keys=True),
#             ToTensord(keys=mask_keys, dtype=torch.float32, allow_missing_keys=True),  # work in float during augs
#         ]

#     # labels
#     label_pre = []
#     if label_keys:
#         label_pre = [
#             EnsureTyped(keys=label_keys, data_type="tensor", allow_missing_keys=True),
#             ToTensord(keys=label_keys, dtype=torch.long, allow_missing_keys=True),
#         ]

#     # augs (operate on image + mask while both have >=2 spatial dims)
#     spatial_keys = img_keys + mask_keys if has_seg else img_keys
#     aug = [
#         RandFlipd(keys=spatial_keys, prob=aug_flip_prob, spatial_axis=-2, allow_missing_keys=True),
#         RandRotate90d(keys=spatial_keys, prob=aug_rot90_prob, spatial_axes=(-2, -1), allow_missing_keys=True),
#     ]

#     # mask post (only AFTER augs)
#     mask_post_indices = []
#     mask_post_channels = []
#     if mask_keys:
#         if seg_target == "indices":
#             # Float 1xHxW -> discretize -> HxW long
#             if num_classes > 2:
#                 mask_post_indices = [
#                     AsDiscreted(keys=mask_keys, argmax=True, allow_missing_keys=True),
#                     SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True),  # 1xHxW -> HxW
#                     ToTensord(keys=mask_keys, dtype=torch.long, allow_missing_keys=True),
#                 ]
#             else:
#                 mask_post_indices = [
#                     AsDiscreted(keys=mask_keys, threshold=binarize_threshold, allow_missing_keys=True),
#                     SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True),
#                     ToTensord(keys=mask_keys, dtype=torch.long, allow_missing_keys=True),
#                 ]
#         else:
#             # channels target: keep channel dim and return float [B,1,H,W] (binary) or [B,C,H,W] (multiclass)
#             if num_classes > 2:
#                 mask_post_channels = [
#                     AsDiscreted(keys=mask_keys, to_onehot=num_classes, allow_missing_keys=True),
#                     ToTensord(keys=mask_keys, dtype=torch.float32, allow_missing_keys=True),
#                 ]
#             else:
#                 mask_post_channels = [
#                     AsDiscreted(keys=mask_keys, threshold=binarize_threshold, allow_missing_keys=True),
#                     ToTensord(keys=mask_keys, dtype=torch.float32, allow_missing_keys=True),
#                 ]

#     # compose
#     if seg_target == "indices":
#         train_t = Compose(img_pre + mask_pre + label_pre + aug + mask_post_indices)
#         val_t = Compose(img_pre + mask_pre + label_pre + mask_post_indices)  # no aug on val
#     else:
#         train_t = Compose(img_pre + mask_pre + label_pre + aug + mask_post_channels)
#         val_t = Compose(img_pre + mask_pre + label_pre + mask_post_channels)

#     return train_t, val_t


# def nested_dict_collate(batch):
#     """
#     Recursively collate a batch of dicts, handling nested dicts for multitask/segmentation/classification.
#     Works with {"image": tensor, "label": {"mask": tensor, "label": int/tensor}}.
#     """
#     import torch
#     elem = batch[0]

#     if isinstance(elem, torch.Tensor):
#         # Stack tensors
#         return torch.stack(batch, dim=0)
#     elif isinstance(elem, dict):
#         # Recursively collate dict fields
#         return {k: nested_dict_collate([d[k] for d in batch]) for k in elem}
#     elif isinstance(elem, (int, float)):
#         # Stack scalars
#         return torch.tensor(batch)
#     elif elem is None:
#         # Leave None
#         return None
#     elif isinstance(elem, list):
#         # Handle lists of uniform type
#         if all(isinstance(x, torch.Tensor) for x in elem):
#             return torch.stack(batch, dim=0)
#         elif all(isinstance(x, (int, float)) for x in elem):
#             return torch.tensor(batch)
#         elif all(isinstance(x, dict) for x in elem):
#             # Collate list of dicts to dict of lists
#             keys = elem[0].keys()
#             return {k: nested_dict_collate([d[k] for d in elem]) for k in keys}
#         elif all(isinstance(x, list) for x in elem):
#             # Nested list: just return as is
#             return batch
#         else:
#             raise TypeError(f"Unsupported list element type: {type(elem)}")
#     else:
#         raise TypeError(f"Unsupported batch element type: {type(elem)}")

# def build_dataloaders(
#     metadata_csv,
#     input_shape=(256, 256),
#     batch_size=8,
#     task="multitask",
#     split=(0.7, 0.15, 0.15),
#     num_workers=32,
#     debug=False,
#     pin_memory=False,
#     multiprocessing_context=None,
#     seg_target: str = "indices",
#     num_classes: int = 2,
#     sampling: str = "none",                 # "none" | "weighted"
#     pos_weight: float | None = None,        # binary
#     neg_weight: float = 1.0,                # binary
#     class_weights: list[float] | None = None,  # multiclass explicit
#     class_counts: list[float] | None = None,  # multiclass derive
#     seed: int = 42,                         # deterministic DataLoader generator
#     invert_labels: bool = False,
#     label_map: dict | None = None,
# ):
#     """ Builds train/val/test DataLoaders from a CSV split by given proportions."""
#     df = pd.read_csv(metadata_csv)

#     require_mask = (task in ['segmentation', 'multitask'])
#     require_image = True
#     df = filter_dataframe(df, require_mask=require_mask, require_image=require_image, verbose=True)

#     # Normalize labels to 0/1 and (optionally) invert
#     if task in ['classification', 'multitask'] and 'label' in df.columns:
#         used_map = normalize_binary_labels_inplace(
#             df, label_col="label", label_map=label_map, invert=bool(invert_labels), verbose=True
#         )
#     else:
#         used_map = None

#     # Split into train, val, test (by fractions)
#     train_frac, val_frac, test_frac = split
#     assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5, "Split fractions must sum to 1.0"
#     temp_frac = val_frac + test_frac
#     train_df, temp_df = train_test_split(
#         df,
#         test_size=temp_frac,
#         random_state=42,
#         stratify=df['label'] if 'label' in df.columns else None
#     )
#     relative_val_frac = val_frac / (val_frac + test_frac)
#     val_df, test_df = train_test_split(
#         temp_df,
#         test_size=1 - relative_val_frac,
#         random_state=42,
#         stratify=temp_df['label'] if 'label' in temp_df.columns else None
#     )

#     train_transforms, val_transforms = get_monai_transforms(task=task, seg_target="indices", num_classes=num_classes)
#     test_transforms = val_transforms

#     train_ds = MammoSegmentationDataset(train_df, input_shape=input_shape, task=task, transform=train_transforms, label_map=used_map)
#     val_ds = MammoSegmentationDataset(val_df, input_shape=input_shape, task=task, transform=val_transforms, label_map=used_map)
#     test_ds = MammoSegmentationDataset(test_df, input_shape=input_shape, task=task, transform=test_transforms, label_map=used_map)

#     # Common kwargs
#     common_kwargs = dict(
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=bool(pin_memory),
#     )
#     if num_workers and num_workers > 0:
#         common_kwargs.update(persistent_workers=True, prefetch_factor=2)

#     if isinstance(multiprocessing_context, str):
#         import multiprocessing as mp
#         multiprocessing_context = mp.get_context(multiprocessing_context)
#     if multiprocessing_context is not None and num_workers and num_workers > 0:
#         common_kwargs["multiprocessing_context"] = multiprocessing_context

#     # Deterministic generator
#     g = torch.Generator()
#     g.manual_seed(int(seed))

#     # optional WEIGHTED SAMPLER for training
#     train_sampler = None
#     if sampling.lower() == "weighted":
#         if "label" not in train_df.columns:
#             print("[WARN] sampling='weighted' requested but 'label' column not found; falling back to shuffle.")
#         else:
#             y_train = extract_targets(train_df, label_col="label")
#             if int(num_classes) == 2:
#                 # default pos_weight ~ n_neg / n_pos if not provided
#                 n_pos = int((y_train == 1).sum())
#                 n_neg = int((y_train == 0).sum())
#                 default_pos_w = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
#                 pw = float(default_pos_w if pos_weight is None else pos_weight)
#                 train_sampler = make_weighted_sampler_binary(y_train, pos_weight=pw, neg_weight=float(neg_weight))
#             else:
#                 cw = torch.as_tensor(class_weights, dtype=torch.float) if class_weights is not None else None
#                 train_sampler = make_weighted_sampler_multiclass(
#                     y_train, class_weights=cw, num_classes=int(num_classes), class_counts=class_counts
#                 )

#     # Helper to construct loaders with fallback for older torch
#     def _make_loader(ds, *, shuffle: bool, drop_last: bool = False, sampler=None):
#         kwargs = {**common_kwargs, "shuffle": (shuffle if sampler is None else False), "drop_last": drop_last}
#         if sampler is not None:
#             kwargs["sampler"] = sampler
#         # PyTorch compatibility fallback
#         try:
#             return DataLoader(ds, generator=g, **kwargs)
#         except TypeError as e:
#             msg = str(e)
#             for bad_key in ("multiprocessing_context", "prefetch_factor", "persistent_workers", "generator"):
#                 if bad_key in kwargs and f"unexpected keyword argument '{bad_key}'" in msg:
#                     kwargs.pop(bad_key, None)
#             return DataLoader(ds, **kwargs)

#     # Build loaders
#     train_loader = _make_loader(train_ds, shuffle=True, drop_last=True, sampler=train_sampler)
#     val_loader = _make_loader(val_ds, shuffle=False)
#     test_loader = _make_loader(test_ds, shuffle=False)

#     if debug:
#         batch = next(iter(val_loader))
#         _sanity_check_batch(batch, task, seg_target="indices")
#         print_batch_debug(batch)

#     return train_loader, val_loader, test_loader


# def _extract_labels_from_batch(batch: Any) -> torch.Tensor:
#     """
#     Supports common batch structures:
#       - (x, y) or (x, y, ...)
#       - {"label": y, ...} or {"y": y, ...}
#     Returns 1D long tensor of labels.
#     """
#     y = None
#     if isinstance(batch, (list, tuple)) and len(batch) >= 2:
#         y = batch[1]
#     elif isinstance(batch, dict):
#         if "label" in batch:
#             y = batch["label"]
#         elif "y" in batch:
#             y = batch["y"]
#     if y is None:
#         raise ValueError("Could not find labels in batch for class count estimation.")
#     y = y if torch.is_tensor(y) else torch.as_tensor(y)
#     return y.view(-1).long()


# def compute_class_counts_from_loader(loader, num_classes: Optional[int] = None) -> Sequence[int]:
#     """
#     Iterate once over loader to tally label frequencies.
#     If num_classes is None, infer from max label + 1.
#     """
#     counts = None
#     seen_max = -1
#     for batch in loader:
#         y = _extract_labels_from_batch(batch)
#         seen_max = max(seen_max, int(y.max().item()))
#         if num_classes is None:
#             k = seen_max + 1
#             if counts is None:
#                 counts = torch.zeros(k, dtype=torch.long)
#             elif k > counts.numel():
#                 counts = torch.nn.functional.pad(counts, (0, k - counts.numel()))
#         else:
#             if counts is None:
#                 counts = torch.zeros(num_classes, dtype=torch.long)
#             elif counts.numel() != num_classes:
#                 raise ValueError(f"num_classes={num_classes} but tally has {counts.numel()} slots")
#         binc = torch.bincount(y, minlength=counts.numel())
#         counts[:binc.numel()] += binc
#     if counts is None:
#         raise RuntimeError("No batches seen while computing class counts.")
#     return [int(x) for x in counts.tolist()]
