# data_utils.py
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import pydicom

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
