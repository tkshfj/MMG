# data_utils.py
import logging

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG or WARNING as appropriate
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

import os
import ast
# from ast import literal_eval
import glob
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, ScaleIntensityd, RandFlipd, RandRotate90d, ToTensord, Lambdad

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
    if verbose:
        print(f"[INFO] Filtered DataFrame: {len(filtered)} of {len(df)} samples remain after file existence check.")
        missing = df[~mask]
        if not missing.empty:
            print("[WARNING] Dropped samples (file not found):")
            print(missing[['image_path', 'mask_paths', 'roi_mask_path']].head())
    return filtered


def print_batch_debug(batch):
    print("Batch keys:", batch.keys())
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
        # Instead of raising:
        logger.error(f"No mask file found for {row}")
        return []

    def load_dicom(self, path):
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
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
            # Optionally, raise or return a dummy/empty sample
            raise IndexError(f"No valid image for idx={idx}")
            # raise ValueError(f"No image path found in row: {row.to_dict()}")
        img = self.load_dicom(img_path)
        # DEBUG
        # print(f"[DEBUG __getitem__] loaded img type: {type(img)}, shape: {getattr(img, 'shape', None)}")
        # if not isinstance(img, np.ndarray):
        #     print(f"[FATAL] __getitem__: img is not ndarray! idx={idx}, type={type(img)}, value={img}")
        #     raise TypeError(f"Image is not a numpy array: {type(img)}, value: {img}")
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = np.expand_dims(img, axis=0)  # [C, H, W]
        image_tensor = torch.as_tensor(img, dtype=torch.float32)

        # Prepare sample dict with image only
        sample = {"image": image_tensor}

        # Optional label
        if self.task in ['classification', 'multitask']:
            label_value = row.get('label')
            if label_value is not None and not (pd.isna(label_value) or label_value == ''):
                label_tensor = torch.tensor(int(label_value), dtype=torch.long)
                sample["label"] = label_tensor

        # Optional mask
        if self.task in ['segmentation', 'multitask']:
            mask_paths = self.resolve_mask_paths(row)
            if mask_paths:
                mask = self.load_and_merge_masks(mask_paths, img.shape[1:])
                mask = cv2.resize(mask, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, axis=0)
                mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
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

        # # Prepare labels
        # label_tensor = torch.tensor(int(row['label'])) if self.task in ['classification', 'multitask'] else None
        # # label_dict = {}
        # if self.task in ['segmentation', 'multitask']:
        #     mask_paths = self.resolve_mask_paths(row)
        #     if not mask_paths:
        #         raise IndexError(f"No valid mask for idx={idx}")
        #     mask = self.load_and_merge_masks(mask_paths, img.shape[1:])
        #     mask = cv2.resize(mask, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
        #     mask = np.expand_dims(mask, axis=0)
        #     mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
        # else:
        #     mask_tensor = None

        # # Always convert image to tensor
        # image_tensor = torch.as_tensor(img, dtype=torch.float32)

        # # Return a FLAT dict, not nested
        # sample = {"image": image_tensor}
        # if label_tensor is not None:
        #     sample["label"] = label_tensor
        # if mask_tensor is not None:
        #     sample["mask"] = mask_tensor

        # if self.transform:
        #     sample = self.transform(sample)
        #     # (Optionally ensure everything is a tensor)
        #     for k in sample:
        #         if not isinstance(sample[k], torch.Tensor):
        #             sample[k] = torch.as_tensor(sample[k])

        # return sample

        # if self.task in ['classification', 'multitask']:
        #     label_dict["label"] = int(row['label'])
        # if self.task in ['segmentation', 'multitask']:
        #     mask_value = row['roi_mask_path']
        #     if isinstance(mask_value, str):
        #         if mask_value.strip().startswith("[") and mask_value.strip().endswith("]"):
        #             mask_paths = literal_eval(mask_value)
        #         else:
        #             mask_paths = [mask_value]
        #     elif isinstance(mask_value, list):
        #         mask_paths = mask_value
        #     else:
        #         raise ValueError(f"Unexpected type for mask paths: {type(mask_value)} (value: {mask_value})")
        #     mask_paths = self.resolve_mask_paths(row)
        #     if not mask_paths:
        #         logger.warning(f"Skipping sample idx={idx} (no mask paths): {row.to_dict()}")
        #         # Optionally, raise or return a dummy/empty sample
        #         raise IndexError(f"No valid mask for idx={idx}")
        #     mask = self.load_and_merge_masks(mask_paths, img.shape[1:])
        #     mask = cv2.resize(mask, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
        #     mask = np.expand_dims(mask, axis=0)
        #     label_dict["mask"] = mask
        # sample = {"image": img, "label": label_dict if label_dict else None}

        # if self.transform:
        #     sample = self.transform(sample)
        #     # print(f"[DEBUG after transform] type: {type(sample)}, sample: {sample}")
        #     if not isinstance(sample, dict):
        #         raise TypeError(f"Transform returned a non-dict: {type(sample)}, value: {sample}")
        # if 'label_transforms' in sample:
        #     sample.pop('label_transforms')
        # # print(f"[DEBUG returning sample] type: {type(sample)}, sample keys: {list(sample.keys())}")
        # return sample


# Custom Transform for Nested Labels
def to_long_nested_label(label):
    """ Convert only label['label'] to long/int if label is a dict with 'label' key. """
    if isinstance(label, dict):
        if "label" in label and not isinstance(label["label"], int):
            # Convert to int if not already
            if hasattr(label["label"], "long"):
                label["label"] = label["label"].long()
            else:
                label["label"] = int(label["label"])
    return label


# MONAI transforms for segmentation and classification tasks
def get_monai_transforms(task="segmentation", input_shape=(256, 256)):
    keys = ["image", "mask"] if task in ["segmentation", "multitask"] else ["image"]
    # For multitask/classification, label might not always be present in some samples
    tensord_keys = keys + (["label"] if task in ["classification", "multitask"] else [])
    train_transforms = [
        ScaleIntensityd(keys=["image"]),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
        RandRotate90d(keys=keys, prob=0.5, allow_missing_keys=True),
        ToTensord(keys=tensord_keys, allow_missing_keys=True),
    ]
    val_transforms = [
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=tensord_keys, allow_missing_keys=True),
    ]
    if task in ["classification", "multitask"]:
        # Apply to nested dict
        train_transforms.append(Lambdad(keys="label", func=to_long_nested_label, allow_missing_keys=True))
        val_transforms.append(Lambdad(keys="label", func=to_long_nested_label, allow_missing_keys=True))
    return Compose(train_transforms), Compose(val_transforms)


def nested_dict_collate(batch):
    """
    Recursively collate a batch of dicts, handling nested dicts for multitask/segmentation/classification.
    Works with {"image": tensor, "label": {"mask": tensor, "label": int/tensor}}.
    """
    import torch
    elem = batch[0]
    # batch: list of samples (each is dict)
    # if isinstance(elem, torch.Tensor):
    #     for i, item in enumerate(batch):
    #         if not isinstance(item, torch.Tensor):
    #             print(f"[FATAL] Collate: Non-tensor in batch at idx {i}: type={type(item)}, value={item}")
    #     return torch.stack(batch, dim=0)
    # elif isinstance(elem, dict):
    #     # Print for debug
    #     # print("[nested_dict_collate] Collating dict with keys:", elem.keys())
    #     # for k in elem.keys():
    #     #     print(f"  - Key '{k}': type={type(elem[k])}, example={elem[k]}")
    #     # Remove metadata keys before recursing
    #     filtered = {k: v for k, v in elem.items() if not k.endswith("_transforms")}
    #     return {k: nested_dict_collate([d[k] for d in batch]) for k in filtered}
    #     # return {k: nested_dict_collate([d[k] for d in batch]) for k in elem}
    # elif isinstance(elem, (int, float)):
    #     return torch.tensor(batch)
    # elif elem is None:
    #     return None
    # elif isinstance(elem, list):
    #     if all(isinstance(x, torch.Tensor) for x in elem):
    #         return torch.stack(batch, dim=0)
    #     elif all(isinstance(x, (int, float)) for x in elem):
    #         return torch.tensor(batch)
    #     elif all(isinstance(x, dict) for x in elem):
    #         # If this happens at top-level, it's a bug.
    #         # Try to collapse it: assume all dicts have the same keys.
    #         keys = elem.keys()
    #         return {k: nested_dict_collate([d[k] for d in batch]) for k in keys}
    #     elif all(isinstance(x, list) for x in elem):
    #         print("[DEBUG] Collate produced a list! This should not happen:", type(elem), elem)
    #     else:
    #         raise TypeError(f"Unsupported type for collate: {type(elem)}")

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


def build_dataloaders(
    metadata_csv,
    input_shape=(256, 256),
    batch_size=8,
    task="multitask",
    split=(0.7, 0.15, 0.15),
    num_workers=32,
    debug=False
):
    """ Builds train/val/test DataLoaders from a CSV split by given proportions."""
    df = pd.read_csv(metadata_csv)
    print("[DEBUG] DataFrame head:")
    print(df.head())

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
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=nested_dict_collate)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), collate_fn=nested_dict_collate)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), collate_fn=nested_dict_collate)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2))

    # Debug batch content for masks
    if debug:
        batch = next(iter(val_loader))
        print_batch_debug(batch)

        # print("\n[DEBUG] Validating val_loader content for segmentation/multitask task:")
        # val_sample = next(iter(val_loader))
        # print("  Batch keys:", val_sample.keys())
        # # Print image shape
        # print("  Image shape:", val_sample["image"].shape)
        # # Print mask shape and sample values (if present)
        # if "mask" in val_sample:
        #     print("  Mask shape:", val_sample["mask"].shape)
        #     print("  Mask sample values:", val_sample["mask"][0, 0, :10, :10])
        # else:
        #     print("  [WARNING] 'mask' not in batch!")
        # # Print label info (if present)
        # if "label" in val_sample:
        #     print("  Label shape:", val_sample["label"].shape)
        #     print("  Label values:", val_sample["label"])
        # else:
        #     print("  [WARNING] 'label' not in batch!")

        # val_sample = next(iter(val_loader))
        # print("  Batch keys:", val_sample.keys())
        # label = val_sample.get("label")
        # if isinstance(label, dict):
        #     print("  Label dict keys:", label.keys())
        #     if "mask" in label:
        #         print("  Mask shape:", label['mask'].shape)
        #     else:
        #         print("  [WARNING] 'mask' not in label dict!")
        # else:
        #     print("  [WARNING] 'label' is not a dict or missing!")
        # print("  Image shape:", val_sample["image"].shape)
        # print("  Mask sample values:", label["mask"][0, 0, :10, :10])

    return train_loader, val_loader, test_loader
