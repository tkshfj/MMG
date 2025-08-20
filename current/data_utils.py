# data_utils.py
import logging

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG or WARNING as appropriate
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

import os
import ast
import glob
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from typing import Any, Optional, Sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, ToTensord, Lambdad, RandFlipd, RandRotate90d

EPSILON = 1e-8  # Small value to avoid division by zero


# Move these functions to module level
def check_image_exists(row):
    """Return True if the image file exists."""
    img_path = row.get('image_path')
    if img_path and os.path.exists(img_path):
        return True
    fallback_path = os.path.join("../data/", row.get('full_mammo_path', ''))
    if fallback_path and os.path.exists(fallback_path):
        return True
    return False


def check_mask_exists(row):
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


def filter_dataframe(df, require_mask=True, require_image=True, verbose=True):
    mask = pd.Series(True, index=df.index)
    if require_image:
        mask &= df.apply(check_image_exists, axis=1)
    if require_mask:
        mask &= df.apply(check_mask_exists, axis=1)
    filtered = df[mask].reset_index(drop=True)

    # if verbose:
    #     print(f"[INFO] Filtered DataFrame: {len(filtered)} of {len(df)} samples remain after file existence check.")
    #     missing = df[~mask]
    #     if not missing.empty:
    #         print("[WARNING] Dropped samples (file not found):")
    #         print(missing[['image_path', 'mask_paths', 'roi_mask_path']].head())

    return filtered


def print_batch_debug(batch):
    print("[DEBUG] Batch keys:", batch.keys())
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape {tuple(v.shape)}, dtype {v.dtype}, sample: {v.flatten()[0:10].tolist()}")
        else:
            print(f"  {k}: {type(v)}")


class MammoSegmentationDataset(Dataset):
    """ MONAI/PyTorch Dataset for multitask learning. """
    def __init__(self, df, root="../data/", input_shape=(256, 256), task='segmentation', transform=None):
        self.df = df
        self.root = root
        self.input_shape = input_shape
        self.task = task
        self.transform = transform

    def resolve_image_path(self, row):
        # Prefer image_path if it exists and file exists
        img_path = row.get('image_path', None)
        if img_path and os.path.exists(img_path):
            return img_path
        # Fallback: try constructing from root and full_mammo_path (should fail if UID-based)
        fallback_path = os.path.join(self.root, row.get('full_mammo_path', ''))
        if os.path.exists(fallback_path):
            return fallback_path
        # Last resort: search for a DICOM file in the corresponding folder using metadata
        folder = os.path.dirname(img_path) if img_path else self.root
        matches = glob.glob(os.path.join(folder, "*.dcm"))
        if matches:
            print(f"[WARNING] Used fallback for image: {matches[0]}")
            return matches[0]
        # raise FileNotFoundError(f"No image file found for {row}")
        return None  # Instead of raising

    def resolve_mask_paths(self, row):
        # mask_paths can be a list or string representation of a list
        mask_paths = row.get('mask_paths')
        if isinstance(mask_paths, str):
            try:
                mask_paths = ast.literal_eval(mask_paths)
            except Exception:
                mask_paths = [mask_paths]
        if isinstance(mask_paths, list):
            existing = [p for p in mask_paths if os.path.exists(p)]
            if existing:
                return existing
        # Fallback: Try roi_mask_path column (not recommended if UIDs)
        roi_mask = row.get('roi_mask_path')
        if roi_mask and os.path.exists(roi_mask):
            return [roi_mask]
        # Try searching for any DICOM in the expected folder
        if mask_paths and len(mask_paths) > 0:
            folder = os.path.dirname(mask_paths[0])
            matches = glob.glob(os.path.join(folder, "*.dcm"))
            if matches:
                print(f"[WARNING] Used fallback for mask: {matches}")
                return matches
        # raise FileNotFoundError(f"No mask file found for {row}")
        logger.error(f"No mask file found for {row}")
        return []

    def load_dicom(self, path):
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            # Invert if MONOCHROME1
            if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                img = img.max() - img
            img = (img - img.min()) / (img.max() - img.min() + EPSILON)
        except Exception as e:
            logger.warning(f"[DICOM ERROR] Failed to load {path}: {e}")
            img = np.zeros(self.input_shape, dtype=np.float32)
        return img

    def load_and_merge_masks(self, mask_paths, shape):
        mask = np.zeros(shape, dtype=np.float32)
        for mpath in mask_paths:
            m = self.load_dicom(mpath)
            if m.shape != mask.shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, m)
        mask = (mask > 0.5).astype(np.float32)
        return mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # DEBUG print row
        # print(f"[DEBUG __getitem__] idx={idx}, row={row.to_dict()}")
        # print("[DEBUG __getitem__] keys:", row.keys())

        # Image path
        img_path = self.resolve_image_path(row)
        if not img_path:
            # logger.warning(f"Skipping sample idx={idx} (no image path): {row.to_dict()}")
            raise IndexError(f"No valid image for idx={idx}")
        img = self.load_dicom(img_path)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)          # [1, H, W]
        image_tensor = torch.from_numpy(img).float()   # float32

        # Prepare sample dict with image only
        sample = {"image": image_tensor}

        # classification / multitask label
        if self.task in ['classification', 'multitask']:
            label_value = row.get('label')
            if label_value is not None and not (pd.isna(label_value) or label_value == ''):
                sample["label"] = torch.tensor(int(label_value), dtype=torch.long)

        # segmentation / multitask mask
        if self.task in ['segmentation', 'multitask']:
            mask_paths = self.resolve_mask_paths(row)
            if mask_paths:
                mask = self.load_and_merge_masks(mask_paths, img.shape[1:])
                mask_tensor = torch.from_numpy((mask > 0.5).astype(np.int64)).long()
                sample["mask"] = mask_tensor

        # Apply transforms if present
        if self.transform:
            sample = self.transform(sample)
            for k in list(sample.keys()):
                # Remove keys where the value is None or NaN
                if sample[k] is None:
                    sample.pop(k)
                elif isinstance(sample[k], torch.Tensor) and sample[k].numel() == 0:
                    sample.pop(k)

        return sample


def to_long_nested_label(label):
    """ Convert only label['label'] to long/int if label is a dict with 'label' key. """
    """Convert nested label to a plain int (required by CrossEntropyLoss)."""
    if isinstance(label, dict):
        v = label.get("label", None)
        if v is None:
            return label
        return int(v.item()) if torch.is_tensor(v) else int(v)
    return int(label.item()) if torch.is_tensor(label) else int(label)


def _ensure_chw(x):
    """Return a tensor with shape [C,H,W]. If x is HxW, add C=1. If x is HxWxC, permute to CxHxW."""
    t = torch.as_tensor(x)
    if t.ndim == 2:                 # H, W
        t = t.unsqueeze(0)          # 1, H, W
    elif t.ndim == 3:
        # If first dim isn't a typical channel count but last is, assume HWC -> CHW
        if t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1)  # C, H, W
    # Optional strict check:
    assert t.ndim >= 3 and t.shape[-2] > 1 and t.shape[-1] > 1, f"Need [...,H,W], got {tuple(t.shape)}"
    return t


def _squeeze_ch1(x):
    import torch
    t = torch.as_tensor(x)
    return t.squeeze(0) if t.ndim == 3 and t.shape[0] == 1 else t  # [1,H,W] -> [H,W]


def get_monai_transforms(task: str = "segmentation", input_shape=(256, 256)):
    img_keys = ["image"]
    mask_keys = ["mask"] if task in ("segmentation", "multitask") else []
    spatial_keys = img_keys + mask_keys

    train_t = [
        ToTensord(keys=img_keys, dtype=torch.float32, allow_missing_keys=True),
        ToTensord(keys=mask_keys, dtype=torch.long, allow_missing_keys=True),
        Lambdad(keys=spatial_keys, func=_ensure_chw, allow_missing_keys=True),
        # (optional) ScaleIntensityRanged(...) if you need it
        RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=-2, allow_missing_keys=True),
        RandRotate90d(keys=spatial_keys, prob=0.5, spatial_axes=(-2, -1), allow_missing_keys=True),
        Lambdad(keys=mask_keys, func=_squeeze_ch1, allow_missing_keys=True),  # -> [H,W]
    ]

    val_t = [
        ToTensord(keys=img_keys, dtype=torch.float32, allow_missing_keys=True),
        ToTensord(keys=mask_keys, dtype=torch.long, allow_missing_keys=True),
        Lambdad(keys=spatial_keys, func=_ensure_chw, allow_missing_keys=True),
        Lambdad(keys=mask_keys, func=_squeeze_ch1, allow_missing_keys=True),
    ]

    if task in ("classification", "multitask"):
        label_tf = [
            Lambdad(keys="label", func=to_long_nested_label, allow_missing_keys=True),
            ToTensord(keys=["label"], dtype=torch.long, allow_missing_keys=True),
        ]
        train_t += label_tf
        val_t += label_tf

    return Compose(train_t), Compose(val_t)


def nested_dict_collate(batch):
    """
    Recursively collate a batch of dicts, handling nested dicts for multitask/segmentation/classification.
    Works with {"image": tensor, "label": {"mask": tensor, "label": int/tensor}}.
    """
    import torch
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        # Stack tensors
        return torch.stack(batch, dim=0)
    elif isinstance(elem, dict):
        # Recursively collate dict fields
        return {k: nested_dict_collate([d[k] for d in batch]) for k in elem}
    elif isinstance(elem, (int, float)):
        # Stack scalars
        return torch.tensor(batch)
    elif elem is None:
        # Leave None
        return None
    elif isinstance(elem, list):
        # Handle lists of uniform type
        if all(isinstance(x, torch.Tensor) for x in elem):
            return torch.stack(batch, dim=0)
        elif all(isinstance(x, (int, float)) for x in elem):
            return torch.tensor(batch)
        elif all(isinstance(x, dict) for x in elem):
            # Collate list of dicts to dict of lists
            keys = elem[0].keys()
            return {k: nested_dict_collate([d[k] for d in elem]) for k in keys}
        elif all(isinstance(x, list) for x in elem):
            # Nested list: just return as is
            return batch
        else:
            raise TypeError(f"Unsupported list element type: {type(elem)}")
    else:
        raise TypeError(f"Unsupported batch element type: {type(elem)}")


def _sanity_check_batch(batch, task):
    x = batch["image"]
    assert x.ndim == 4 and x.shape[1] == 1, f"image must be [N,1,H,W], got {tuple(x.shape)}"
    if task in ("segmentation", "multitask") and "mask" in batch:
        m = batch["mask"]
        assert m.ndim == 3, f"mask must be [N,H,W] integer, got {tuple(m.shape)}"
        assert m.dtype in (torch.int64, torch.long), f"mask dtype must be long, got {m.dtype}"
    if task in ("classification", "multitask") and "label" in batch:
        y = batch["label"]
        assert y.dtype in (torch.int64, torch.long), f"label dtype must be long, got {y.dtype}"


def build_dataloaders(
    metadata_csv,
    input_shape=(256, 256),
    batch_size=8,
    task="multitask",
    split=(0.7, 0.15, 0.15),
    num_workers=32,
    debug=False,
    pin_memory=False
):
    """ Builds train/val/test DataLoaders from a CSV split by given proportions."""
    df = pd.read_csv(metadata_csv)

    # print("[DEBUG] DataFrame head:")
    # print(df.head())

    # Filter for files that exist
    require_mask = (task in ['segmentation', 'multitask'])
    require_image = True
    df = filter_dataframe(df, require_mask=require_mask, require_image=require_image, verbose=True)

    # Split into train, val, test (by fractions)
    train_frac, val_frac, test_frac = split
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5, "Split fractions must sum to 1.0"
    # First split: Train vs (Val+Test)
    temp_frac = val_frac + test_frac
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_frac,
        random_state=42,
        stratify=df['label'] if 'label' in df.columns else None
    )
    # Second split: Val vs Test
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val_frac,
        random_state=42,
        stratify=temp_df['label'] if 'label' in temp_df.columns else None
    )

    train_transforms, val_transforms = get_monai_transforms(task=task, input_shape=input_shape)
    test_transforms = val_transforms
    # Create datasets
    train_ds = MammoSegmentationDataset(train_df, input_shape=input_shape, task=task, transform=train_transforms)
    val_ds = MammoSegmentationDataset(val_df, input_shape=input_shape, task=task, transform=val_transforms)
    test_ds = MammoSegmentationDataset(test_df, input_shape=input_shape, task=task, transform=test_transforms)
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), pin_memory=pin_memory)

    # Debug batch content for masks
    if debug:
        batch = next(iter(val_loader))
        _sanity_check_batch(batch, task)
        print_batch_debug(batch)

    return train_loader, val_loader, test_loader


def _extract_labels_from_batch(batch: Any) -> torch.Tensor:
    """
    Supports common batch structures:
      - (x, y) or (x, y, ...)
      - {"label": y, ...} or {"y": y, ...}
    Returns 1D long tensor of labels.
    """
    y = None
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        y = batch[1]
    elif isinstance(batch, dict):
        if "label" in batch:
            y = batch["label"]
        elif "y" in batch:
            y = batch["y"]
    if y is None:
        raise ValueError("Could not find labels in batch for class count estimation.")
    y = y if torch.is_tensor(y) else torch.as_tensor(y)
    return y.view(-1).long()


def compute_class_counts_from_loader(loader, num_classes: Optional[int] = None) -> Sequence[int]:
    """
    Iterate once over loader to tally label frequencies.
    If num_classes is None, infer from max label + 1.
    """
    counts = None
    seen_max = -1
    for batch in loader:
        y = _extract_labels_from_batch(batch)
        seen_max = max(seen_max, int(y.max().item()))
        if num_classes is None:
            k = seen_max + 1
            if counts is None:
                counts = torch.zeros(k, dtype=torch.long)
            elif k > counts.numel():
                counts = torch.nn.functional.pad(counts, (0, k - counts.numel()))
        else:
            if counts is None:
                counts = torch.zeros(num_classes, dtype=torch.long)
            elif counts.numel() != num_classes:
                raise ValueError(f"num_classes={num_classes} but tally has {counts.numel()} slots")
        binc = torch.bincount(y, minlength=counts.numel())
        counts[:binc.numel()] += binc
    if counts is None:
        raise RuntimeError("No batches seen while computing class counts.")
    return [int(x) for x in counts.tolist()]
