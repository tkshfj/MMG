# data_utils_monai.py
from ast import literal_eval
import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, RandFlipd, RandRotate90d, ToTensord, ScaleIntensityd

EPSILON = 1e-8  # Small value to avoid division by zero


class MammoSegmentationDataset(Dataset):
    """ MONAI/PyTorch Dataset for mammogram segmentation or multitask learning."""
    def __init__(self, df, input_shape=(256, 256), task='segmentation', transform=None):
        self.df = df
        self.input_shape = input_shape
        self.task = task
        self.transform = transform

    def load_dicom(self, path):
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + EPSILON)
        except Exception as e:
            print(f"[DICOM ERROR] {path}: {e}")
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
        img = self.load_dicom(row['image_path'])
        img = cv2.resize(img, self.input_shape)
        img = np.expand_dims(img, axis=0)  # [C, H, W] for MONAI

        if self.task == 'segmentation':
            mask_paths = literal_eval(row['mask_paths']) if isinstance(row['mask_paths'], str) else row['mask_paths']
            mask = self.load_and_merge_masks(mask_paths, img.shape[1:])
            mask = cv2.resize(mask, self.input_shape, interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=0)
            sample = {"image": img, "mask": mask}
        elif self.task == 'multitask':
            mask_paths = literal_eval(row['mask_paths']) if isinstance(row['mask_paths'], str) else row['mask_paths']
            mask = self.load_and_merge_masks(mask_paths, img.shape[1:])
            mask = cv2.resize(mask, self.input_shape, interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=0)
            label = float(row['label'])
            sample = {"image": img, "mask": mask, "label": label}
        else:  # classification
            label = float(row['label'])
            sample = {"image": img, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_monai_transforms(task="segmentation", input_shape=(256, 256)):
    from monai.transforms import (
        Compose, RandFlipd, RandRotate90d, ScaleIntensityd, ToTensord
    )
    keys = ["image", "mask"] if task in ["segmentation", "multitask"] else ["image"]
    train_transforms = Compose([
        ScaleIntensityd(keys=["image"]),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=keys, prob=0.5),
        ToTensord(keys=keys + (["label"] if task == "multitask" else []))
    ])
    val_transforms = Compose([
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=keys + (["label"] if task == "multitask" else []))
    ])
    return train_transforms, val_transforms


def build_dataloaders(
    metadata_csv,
    input_shape=(256, 256),
    batch_size=8,
    task="segmentation",
    split=(0.7, 0.15, 0.15),
    num_workers=32
):
    """Builds train/val/test DataLoaders from a CSV split by given proportions."""
    # Load dataframe
    df = pd.read_csv(metadata_csv)

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

    # Decide transform keys
    if task in ["segmentation", "multitask"]:
        keys = ["image", "mask"]
    else:
        keys = ["image"]

    # Define transforms (train has augment, val/test don't)
    train_transforms = Compose([
        ScaleIntensityd(keys=["image"]),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandRotate90d(keys=keys, prob=0.5),
        ToTensord(keys=keys + (["label"] if task == "multitask" else []))
    ])
    val_transforms = Compose([
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=keys + (["label"] if task == "multitask" else []))
    ])
    test_transforms = val_transforms

    # Create datasets
    train_ds = MammoSegmentationDataset(train_df, input_shape=input_shape, task=task, transform=train_transforms)
    val_ds = MammoSegmentationDataset(val_df, input_shape=input_shape, task=task, transform=val_transforms)
    test_ds = MammoSegmentationDataset(test_df, input_shape=input_shape, task=task, transform=test_transforms)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2))

    return train_loader, val_loader, test_loader
