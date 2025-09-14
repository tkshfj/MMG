# data_transforms.py
from __future__ import annotations
from typing import Tuple, List, Union
import numpy as np
import pydicom
import torch

from monai.transforms import (
    Compose, EnsureTyped, Lambdad,  # EnsureChannelFirstd,
    NormalizeIntensityd, RandFlipd, RandRotate90d,
    ScaleIntensityRangePercentilesd, ResizeD,
    AsDiscreted, SqueezeDimd, RandCropByPosNegLabeld,
)


# Top-level helpers (picklable)
def read_dicom(path: str) -> np.ndarray:
    """Load a single DICOM to float32 HxW with slope/intercept and MONOCHROME1 handled."""
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0) or 0.0)
    img = img * slope + intercept
    if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        img = img.max() - img
    return img  # HxW


def read_and_merge_masks(obj: Union[str, List[str]]) -> np.ndarray:
    """
    Accepts a single path or a list of paths. If list, resize each to the first mask's
    spatial size (nearest) and take elementwise max (logical OR for binary masks).
    Returns a single float32 HxW array.
    """
    import cv2  # local import to avoid hard dep at import time
    if isinstance(obj, str):
        return read_dicom(obj).astype(np.float32)

    assert isinstance(obj, list) and len(obj) > 0, "mask list must be non-empty"
    base = read_dicom(obj[0]).astype(np.float32)
    H, W = base.shape
    if len(obj) == 1:
        return base
    merged = base.copy()
    for p in obj[1:]:
        mi = read_dicom(p).astype(np.float32)
        if mi.shape != (H, W):
            mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, mi)
    return merged


def add_channel_if_2d(x):
    # import numpy as np, torch
    if torch.is_tensor(x):
        return x.unsqueeze(0) if x.ndim == 2 else x
    x = np.asarray(x)
    return x[None, ...] if x.ndim == 2 else x


def _pos_neg_from_ratio(ratio: float, num_samples: int) -> tuple[int, int]:
    """Convert desired positive ratio -> (pos_count, neg_count) with at least 1 of each."""
    r = max(0.0, min(1.0, float(ratio)))
    ns = max(2, int(num_samples))          # need >=2 to allow both pos and neg
    pos_k = int(round(r * ns))
    pos_k = max(1, min(pos_k, ns - 1))
    return pos_k, ns - pos_k


# Core builder
def _make_transforms(
    *,
    input_shape: Tuple[int, int],
    task: str,
    seg_target: str,
    num_classes: int,
    bin_thresh: float,
    augment: bool,
    crop_train: bool = False,
    lesion_class: int = 1,
    pos_ratio: float = 0.5,
    num_samples: int = 4,
) -> Compose:
    has_masks = task.lower() in {"segmentation", "multitask"}
    img_keys = ["image"]
    mask_keys = ["mask"] if has_masks else []
    both_keys = img_keys + mask_keys

    t = []

    # IO / load (top-level functions → picklable)
    t.append(Lambdad(keys=img_keys, func=read_dicom, allow_missing_keys=False))
    if mask_keys:
        t.append(Lambdad(keys=mask_keys, func=read_and_merge_masks, allow_missing_keys=True))

    # Adds a channel if missing
    t.append(Lambdad(keys=both_keys, func=add_channel_if_2d, allow_missing_keys=True))

    # Size/cropping
    if augment and has_masks and crop_train:
        # Make mask binary so positive voxels are 1, background 0
        # This ensures RandCropByPosNegLabeld treats label>0 as "pos".
        t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))

        # Convert desired ratio to counts (no pos_ratio arg)
        pos_k, neg_k = _pos_neg_from_ratio(pos_ratio, num_samples)

        # Lesion-aware sampler: picks centers from mask>0 for 'pos_k' and from mask==0 for 'neg_k'
        t.append(RandCropByPosNegLabeld(
            keys=both_keys,
            label_key="mask",
            spatial_size=tuple(input_shape),
            pos=pos_k,                       # counts, NOT class id
            neg=neg_k,                       # counts, NOT class id
            num_samples=int(num_samples),
            image_key="image",
            image_threshold=0.0,
        ))
        # No resize after cropping; patches are already input_shape [H,W] = input_shape.
    else:
        # Val/test (or train without crop): global resize
        t.append(
            ResizeD(
                keys=both_keys,
                spatial_size=input_shape,
                mode=("bilinear", "nearest") if has_masks else "bilinear",
                allow_missing_keys=True,
            )
        )

    # Intensity & typing (image only for intensity)
    t.append(ScaleIntensityRangePercentilesd(keys=img_keys, lower=2, upper=98, b_min=-1.0, b_max=1.0, clip=True))
    t.append(NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True))
    t.append(EnsureTyped(keys=both_keys, dtype=torch.float32, allow_missing_keys=True))

    # Augmentations (train only)
    if augment:
        t.append(RandFlipd(keys=both_keys, prob=0.5, spatial_axis=-2))
        t.append(RandRotate90d(keys=both_keys, prob=0.5, spatial_axes=(-2, -1)))

    # Mask post-processing (final target formatting for the model & loss)
    if mask_keys:
        if seg_target == "indices":
            if num_classes <= 2:
                # keep binary 0/1; we already discretized earlier for the crop
                t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))
            else:
                t.append(AsDiscreted(keys=mask_keys, argmax=True, allow_missing_keys=True))
            # model usually expects [H,W] integer labels for indices
            t.append(SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True))
            t.append(EnsureTyped(keys=mask_keys, dtype=torch.long, allow_missing_keys=True))
        else:
            # channels target -> keep float (optionally one-hot for >2 classes)
            if num_classes <= 2:
                t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))
            else:
                t.append(AsDiscreted(keys=mask_keys, to_onehot=num_classes, allow_missing_keys=True))

    return Compose(t)

    # # Resize
    # t.append(
    #     ResizeD(
    #         keys=both_keys,
    #         spatial_size=input_shape,
    #         mode=("bilinear", "nearest") if has_masks else "bilinear",
    #         allow_missing_keys=True,
    #     )
    # )

    # # Intensity & typing
    # t.append(ScaleIntensityRangePercentilesd(keys=img_keys, lower=2, upper=98, b_min=-1.0, b_max=1.0, clip=True))
    # t.append(NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True))
    # t.append(EnsureTyped(keys=both_keys, dtype=torch.float32, allow_missing_keys=True))

    # # Augmentations (train only)
    # if augment:
    #     t.append(RandFlipd(keys=both_keys, prob=0.5, spatial_axis=-2))
    #     t.append(RandRotate90d(keys=both_keys, prob=0.5, spatial_axes=(-2, -1)))

    # # Mask post-processing
    # if mask_keys:
    #     if seg_target == "indices":
    #         if num_classes <= 2:
    #             t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))
    #         else:
    #             t.append(AsDiscreted(keys=mask_keys, argmax=True, allow_missing_keys=True))
    #         # model usually expects [H,W] integer labels for indices
    #         t.append(SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True))
    #         t.append(EnsureTyped(keys=mask_keys, dtype=torch.long, allow_missing_keys=True))
    #     else:
    #         # channels target → keep float
    #         if num_classes <= 2:
    #             t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))
    #         else:
    #             t.append(AsDiscreted(keys=mask_keys, to_onehot=num_classes, allow_missing_keys=True))

    # return Compose(t)


# Public APIs
def make_train_transforms(
    *,
    input_shape: Tuple[int, int] = (256, 256),
    task: str = "multitask",
    seg_target: str = "indices",  # "indices" | "channels"
    num_classes: int = 2,
    bin_thresh: float = 0.5,
    crop_train: bool = True,
    lesion_class: int = 1,
    pos_ratio: float = 0.5,
    num_samples: int = 4,
) -> Compose:
    return _make_transforms(
        input_shape=input_shape,
        task=task,
        seg_target=seg_target,
        num_classes=num_classes,
        bin_thresh=bin_thresh,
        augment=True,
        crop_train=crop_train,
        lesion_class=lesion_class,
        pos_ratio=pos_ratio,
        num_samples=num_samples,
    )


def make_val_transforms(
    *,
    input_shape: Tuple[int, int] = (256, 256),
    task: str = "multitask",
    seg_target: str = "indices",
    num_classes: int = 2,
    bin_thresh: float = 0.5,
) -> Compose:
    return _make_transforms(
        input_shape=input_shape,
        task=task,
        seg_target=seg_target,
        num_classes=num_classes,
        bin_thresh=bin_thresh,
        augment=False,
        crop_train=False,
    )
