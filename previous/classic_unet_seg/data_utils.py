# Import necessary libraries
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pydicom
from ast import literal_eval
from sklearn.model_selection import train_test_split
import cv2
from typing import Tuple

# Global configuration
INPUT_SHAPE = (256, 256, 1)  # (512, 512, 1)
TARGET_SIZE = INPUT_SHAPE[:2]


def load_dicom(path):
    """Loads a DICOM file and rescales to [0,1]."""
    path = path.decode('utf-8') if isinstance(path, bytes) else path
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    except Exception as e:
        print(f"[DICOM ERROR] {path}: {e}")
        img = np.zeros(TARGET_SIZE, dtype=np.float32)
    return img


def load_and_merge_masks(mask_paths, shape):
    """Loads and merges multiple mask DICOMs into one binary mask."""
    mask = np.zeros(shape, dtype=np.float32)
    for mpath in mask_paths:
        m = load_dicom(mpath)
        if m.shape != mask.shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.maximum(mask, m)
    mask = (mask > 0.5).astype(np.float32)  # ensures masks are binarized (only 0 or 1 values)
    return mask


def parse_mask_paths(mask_str):
    """Parses a stringified Python list of mask paths."""
    if isinstance(mask_str, str):
        try:
            return literal_eval(mask_str)
        except Exception:
            return []
    return mask_str if isinstance(mask_str, list) else []


def preprocess(img, input_shape, augment):
    """Preprocesses and augments (image) for training/classification."""
    img = tf.cast(img, tf.float32)
    if augment:
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
    img = tf.image.resize(img, input_shape[:2])
    img = tf.ensure_shape(img, (input_shape[0], input_shape[1], 1))
    return img


def preprocess_pair(img, mask, input_shape, augment):
    """Preprocesses and augments (image, mask) pairs for segmentation."""
    img = tf.cast(img, tf.float32)
    mask = tf.cast(mask, tf.float32)
    if augment:
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
    img = tf.image.resize(img, input_shape[:2])
    mask = tf.image.resize(mask, input_shape[:2], method='nearest')
    img = tf.ensure_shape(img, (input_shape[0], input_shape[1], 1))
    mask = tf.ensure_shape(mask, (input_shape[0], input_shape[1], 1))
    return img, mask


def build_dataset(
    metadata_csv: str,
    input_shape=INPUT_SHAPE,
    batch_size: int = 8,
    task: str = 'classification',  # 'classification', 'segmentation', or 'multitask'
    shuffle: bool = True,
    augment: bool = True,
    split=(0.7, 0.15, 0.15)
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Creates TensorFlow datasets for:
      - Binary classification (task='classification')
      - Classic U-Net segmentation (task='segmentation')
      - Multitask U-Net (task='multitask')

    Returns:
        train_ds, val_ds, test_ds: tf.data.Dataset objects.
    """
    df = pd.read_csv(metadata_csv)
    # Parse masks
    if 'mask_paths' in df.columns:
        df['mask_paths'] = df['mask_paths'].apply(parse_mask_paths)
    else:
        df['mask_paths'] = [[] for _ in range(len(df))]
    # Labels
    if 'label' in df.columns:
        df['label'] = df['label'].astype(np.float32)
    else:
        df['label'] = 0.0

    # Filtering
    if task == 'classification':
        samples = [
            (row['image_path'], row['label'])
            for _, row in df.iterrows()
            if os.path.exists(row['image_path'])
        ]
    else:  # segmentation or multitask
        samples = [
            (row['image_path'], row['mask_paths'], row['label'])
            for _, row in df.iterrows()
            if os.path.exists(row['image_path']) and row['mask_paths']
        ]
    print(f"Loaded {len(samples)} samples for task: {task}")

    # Split indices
    idxs = np.arange(len(samples))
    train_idx, test_idx = train_test_split(idxs, test_size=split[2], random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=split[1] / (split[0] + split[1]), random_state=42)

    # Generators
    if task == 'classification':
        def generator(indices):
            for i in indices:
                img_path, label = samples[i]
                img = load_dicom(img_path)
                img = img[..., None]
                yield img, label

        def tf_preprocess(img, label):
            img = preprocess(img, input_shape, augment)
            return img, tf.cast(label, tf.float32)

        output_signature = (
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    elif task == 'segmentation':
        def generator(indices):
            for i in indices:
                img_path, mask_list, _ = samples[i]
                img = load_dicom(img_path)
                mask = load_and_merge_masks(mask_list, img.shape)
                img = img[..., None]
                mask = mask[..., None]
                yield img, mask

        def tf_preprocess(img, mask):
            img, mask = preprocess_pair(img, mask, input_shape, augment)
            return img, mask

        output_signature = (
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)
        )
    elif task == 'multitask':
        def generator(indices):
            for i in indices:
                img_path, mask_list, label = samples[i]
                img = load_dicom(img_path)
                mask = load_and_merge_masks(mask_list, img.shape)
                img = img[..., None]
                mask = mask[..., None]
                yield img, mask, label

        def tf_preprocess(img, mask, label):
            img, mask = preprocess_pair(img, mask, input_shape, augment)
            return img, {"segmentation": mask, "classification": tf.cast(label, tf.float32)}

        output_signature = (
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    def make_ds(indices, shuffle_ds=True):
        ds = tf.data.Dataset.from_generator(
            lambda: generator(indices),
            output_signature=output_signature
        )
        ds = ds.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()  # Only if RAM is sufficient
        if shuffle_ds:
            ds = ds.shuffle(buffer_size=256)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(train_idx, shuffle_ds=True)
    val_ds = make_ds(val_idx, shuffle_ds=False)
    test_ds = make_ds(test_idx, shuffle_ds=False)
    return train_ds, val_ds, test_ds

# Example usage:
# For binary classification (shallow CNN):
# train_ds, val_ds, test_ds = build_dataset("cbis_ddsm_metadata_full.csv", task='classification')
# For classic U-Net:
# train_ds, val_ds, test_ds = build_dataset("cbis_ddsm_metadata_full.csv", task='segmentation')
# For multitask U-Net:
# train_ds, val_ds, test_ds = build_dataset("cbis_ddsm_metadata_full.csv", task='multitask')
