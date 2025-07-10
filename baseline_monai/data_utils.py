# data_utils.py
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import pydicom
import torch
from torch.utils.data import WeightedRandomSampler

# Global configuration
INPUT_SHAPE = (224, 224, 1)  # (512, 512, 1)
TARGET_SIZE = INPUT_SHAPE[:2]


# DICOM Loader
def load_dicom_image(path_tensor):
    """
    Loads and normalizes a DICOM image from a byte string path
    Converts image pixel values to float32, scales them to [0, 1], and handles exceptions by returning a zero image.
    """
    path = path_tensor.decode('utf-8')  # Decode byte string to UTF-8
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img -= np.min(img)
        img /= (np.max(img) + 1e-6)  # normalize to [0,1]
    except Exception as e:
        print(f"[DICOM ERROR] {path}: {e}")
        img = np.zeros(TARGET_SIZE, dtype=np.float32)
    return img


# TensorFlow Wrappers
def tf_load_dicom(path):
    """
    loads a single full mammogram DICOM image using load_dicom_image.
    Ensures the image has shape (H, W, 1), resizes it to the target size, and returns it as a TensorFlow tensor.
    """
    img = tf.numpy_function(func=load_dicom_image, inp=[path], Tout=tf.float32)
    img.set_shape([None, None])  # initially 2D
    img = tf.expand_dims(img, axis=-1)  # [H, W, 1]
    img.set_shape([None, None, 1])
    img = tf.image.resize(img, TARGET_SIZE)
    return img


def tf_load_multiple_dicom(paths):
    """
    Loads and combines multiple DICOM mask images (e.g., for multiple ROIs).
    Loads each mask, stacks them, and returns the pixel-wise union using tf.reduce_max.
    """
    def load_single(path):
        img = tf.numpy_function(load_dicom_image, [path], tf.float32)
        img.set_shape([None, None])
        img = tf.expand_dims(img, axis=-1)
        img.set_shape([None, None, 1])
        img = tf.image.resize(img, TARGET_SIZE)
        return img

    masks = tf.map_fn(
        load_single,
        paths,
        fn_output_signature=tf.TensorSpec(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1), dtype=tf.float32)
    )
    return tf.reduce_max(masks, axis=0)  # union of all masks


# Unified MTL Preprocessor for multitask learning (MTL)
def load_and_preprocess(image_path, mask_paths, label):
    """
    loads a single image, multiple mask images, and casts the label.
    Returns a tuple: (image, {"segmentation": mask, "classification": label}).
    """
    image = tf_load_dicom(image_path)
    mask = tf_load_multiple_dicom(mask_paths)
    label = tf.cast(label, tf.float32)
    return image, {"segmentation": mask, "classification": label}


def parse_record(record):
    """
    parses a dictionary record (with keys image_path, mask_paths, label) using load_and_preprocess.
    """
    image_path = record['image_path']
    mask_paths = record['mask_paths']
    label = record['label']
    image, target = load_and_preprocess(image_path, mask_paths, label)
    return image, target


# Builds a tf.data.Dataset from a metadata CSV file
def build_tf_dataset(
    metadata_csv: str,
    batch_size: int = 8,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Reads and parses the CSV, converts mask path strings to lists, ensures correct label type, and creates a TensorFlow dataset of records.
    Applies the multitask mapping (parse_record), shuffles, batches, and prefetches.
    """

    # Load metadata CSV
    df = pd.read_csv(metadata_csv)
    # Parse stringified list of mask_paths
    df['mask_paths'] = df['mask_paths'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    # Ensure label column is float32-compatible (e.g., 0.0, 1.0)
    df['label'] = df['label'].astype(np.float32)
    # Convert to list of dicts
    records = df[['image_path', 'mask_paths', 'label']].to_dict(orient='records')
    # Create dataset
    ds = tf.data.Dataset.from_generator(
        lambda: (r for r in records),
        output_signature={
            "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
            "mask_paths": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "label": tf.TensorSpec(shape=(), dtype=tf.float32),
        }
    )
    # Apply MTL-compatible mapping function
    ds = ds.map(lambda r: parse_record(r), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(records))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# MONAI/PyTorch Dataloader version for the same CSV/metadata pipeline
try:
    from monai.data import CacheDataset, DataLoader
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ResizeWithPadOrCropd,
        RandFlipd, RandRotate90d, RandZoomd, RandAffined, RandGaussianNoised, ToTensord, Compose
    )
    from sklearn.model_selection import train_test_split
except ImportError:
    # If MONAI not installed, this code block will be skipped (safe for TF-only environments)
    pass


def build_dataloaders(
    metadata_csv: str,
    batch_size: int = 8,
    shuffle: bool = True,
    input_shape = TARGET_SIZE,
    mode: str = "classification", # or "segmentation"
    num_workers: int = 4,
    rotation: float = 0.1,
    zoom: float = 0.1,
    random_state: int = 42,
):
    """
    Reads and parses the CSV, converts mask path strings to lists, ensures correct label type,
    and creates PyTorch DataLoaders of records for MONAI training.
    Applies augmentations similar to the TF pipeline.
    Returns train_loader, val_loader.
    """

    # Load metadata CSV
    df = pd.read_csv(metadata_csv)
    # Parse stringified list of mask_paths
    if 'mask_paths' in df.columns:
        df['mask_paths'] = df['mask_paths'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
        )
    # Ensure label column is float-compatible
    if 'label' in df.columns:
        df['label'] = df['label'].astype(float)

    # Convert to list of dicts (MTL compatible: image_path, mask_paths, label)
    records = df[['image_path', 'mask_paths', 'label']].to_dict(orient='records')

    # Choose input keys
    if mode == "classification":
        for r in records:
            r['image'] = r.pop('image_path')
            if 'mask_paths' in r:
                r.pop('mask_paths')
        keys = ["image"]
    elif mode == "segmentation":
        for r in records:
            r['image'] = r.pop('image_path')
            mask_paths = r.pop('mask_paths')
            r['mask'] = mask_paths[0] if mask_paths else None
        records = [r for r in records if r['mask'] is not None]
        keys = ["image", "mask"]
    else:
        raise ValueError("Unknown mode: choose 'classification' or 'segmentation'.")

    # Train/val split, stratified by label for class balance
    train_records, val_records = train_test_split(
        records,
        test_size=0.2,
        stratify=[r["label"] for r in records],
        random_state=random_state
    )

    # MONAI transforms (augments similar to Keras pipeline)
    train_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(keys=keys, spatial_size=input_shape),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandRotate90d(keys=keys, prob=0.5),
        RandZoomd(keys=keys, prob=0.2, min_zoom=1.0-zoom, max_zoom=1.0+zoom),
        RandAffined(keys=keys, prob=0.3, rotate_range=(0.2, 0.2), shear_range=(0.1, 0.1), scale_range=(0.1, 0.1)),
        RandGaussianNoised(keys=keys, prob=0.15, std=0.01),
        ToTensord(keys=keys),
    ])
    val_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(keys=keys, spatial_size=input_shape),
        ToTensord(keys=keys),
    ])

    train_ds = CacheDataset(data=train_records, transform=train_transforms, num_workers=num_workers)
    val_ds = CacheDataset(data=val_records, transform=val_transforms, num_workers=num_workers)

    # BEGIN RESAMPLING LOGIC
    # Compute class sample weights for the training set
    train_labels = np.array([r["label"] for r in train_records])
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(l)] for l in train_labels])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # Use sampler for train_loader (no shuffle when using sampler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    # END RESAMPLING LOGIC

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
