# data_transforms.py
from __future__ import annotations
from typing import Tuple, List, Union
import numpy as np
import pydicom
import torch

from monai.transforms import (
    Compose, EnsureTyped, Lambdad,
    NormalizeIntensityd, RandFlipd, RandRotate90d,
    ScaleIntensityRangePercentilesd, ResizeD,
    AsDiscreted, SqueezeDimd
)


# Low-level DICOM helpers
def _read_dicom(path: str) -> np.ndarray:
    """Load a single DICOM to float32 HxW with slope/intercept and MONOCHROME1 handled."""
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0) or 0.0)
    img = img * slope + intercept
    if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        img = img.max() - img
    return img  # HxW


def _read_and_merge_masks(obj: Union[str, List[str]]) -> np.ndarray:
    """
    Accepts a single path or a list of paths. If list, resize each to the first mask's
    spatial size (nearest) and take elementwise max (logical OR for binary masks).
    Returns a single float32 HxW array.
    """
    import cv2  # local import to avoid hard dep at import time
    if isinstance(obj, str):
        m = _read_dicom(obj)
        return m.astype(np.float32)

    # obj is a list[str]
    assert isinstance(obj, list) and len(obj) > 0, "mask list must be non-empty"
    base = _read_dicom(obj[0]).astype(np.float32)
    H, W = base.shape  # reference size
    if len(obj) == 1:
        return base

    merged = base.copy()
    for p in obj[1:]:
        mi = _read_dicom(p).astype(np.float32)
        if mi.shape != (H, W):
            mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, mi)
    return merged


# Transforms
def make_train_transforms(
    *,
    input_shape: Tuple[int, int] = (256, 256),
    task: str = "multitask",
    seg_target: str = "indices",
    num_classes: int = 2,
    bin_thresh: float = 0.5,
    aug_flip_prob: float = 0.5,
    aug_rot90_prob: float = 0.5,
) -> Compose:
    has_seg = task.lower() != "classification"
    img_keys = ["image"]
    mask_keys = ["mask"] if has_seg else []

    t = [
        Lambdad(keys=img_keys, func=_read_dicom, allow_missing_keys=False),
        Lambdad(keys=mask_keys, func=_read_and_merge_masks, allow_missing_keys=True),

        # add channel dim
        Lambdad(keys=img_keys + mask_keys, func=lambda x: x[None, ...] if getattr(x, "ndim", None) == 2 else x,
                allow_missing_keys=True),

        ResizeD(
            keys=img_keys + mask_keys,
            spatial_size=input_shape,
            mode=("bilinear", "nearest") if has_seg else "bilinear",
            allow_missing_keys=True,
        ),

        # robust centering
        ScaleIntensityRangePercentilesd(keys=img_keys, lower=2, upper=98, b_min=-1.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True),
        EnsureTyped(keys=img_keys + mask_keys, dtype=torch.float32, allow_missing_keys=True),
    ]

    # augs
    if has_seg:
        t += [
            RandFlipd(keys=img_keys + mask_keys, prob=aug_flip_prob, spatial_axis=-2),
            RandRotate90d(keys=img_keys + mask_keys, prob=aug_rot90_prob, spatial_axes=(-2, -1)),
        ]
    else:
        t += [
            RandFlipd(keys=img_keys, prob=aug_flip_prob, spatial_axis=-2),
            RandRotate90d(keys=img_keys, prob=aug_rot90_prob, spatial_axes=(-2, -1)),
        ]

    # mask post
    if has_seg:
        if seg_target == "indices":
            if num_classes <= 2:
                t += [
                    AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True),
                    SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True),
                    EnsureTyped(keys=mask_keys, dtype=torch.long, allow_missing_keys=True),
                ]
            else:
                t += [
                    AsDiscreted(keys=mask_keys, argmax=True, allow_missing_keys=True),
                    SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True),
                    EnsureTyped(keys=mask_keys, dtype=torch.long, allow_missing_keys=True),
                ]
        else:
            # channels target → keep float
            if num_classes <= 2:
                t += [AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True)]
            else:
                t += [AsDiscreted(keys=mask_keys, to_onehot=num_classes, allow_missing_keys=True)]

    return Compose(t)


def make_val_transforms(
    *,
    input_shape: Tuple[int, int] = (256, 256),
    task: str = "multitask",
    seg_target: str = "indices",
    num_classes: int = 2,
    bin_thresh: float = 0.5,
) -> Compose:
    # same as train, but without random augs
    return make_train_transforms(
        input_shape=input_shape,
        task=task,
        seg_target=seg_target,
        num_classes=num_classes,
        bin_thresh=bin_thresh,
        aug_flip_prob=0.0,
        aug_rot90_prob=0.0,
    )

# from __future__ import annotations
# import numpy as np
# import pydicom
# from typing import Tuple
# from monai.transforms import (
#     Compose, EnsureTyped, EnsureChannelFirstd, Lambdad, NormalizeIntensityd, RandFlipd, RandRotate90d, LoadImaged, ScaleIntensityRangePercentilesd
# )
# # ResizeD, AsDiscreted, SqueezeDimd

# # Minimal DICOM loader used via Lambdad
# def _load_dicom(path: str) -> np.ndarray:
#     dcm = pydicom.dcmread(path)
#     img = dcm.pixel_array.astype(np.float32)
#     slope = float(getattr(dcm, "RescaleSlope", 1.0) or 1.0)
#     intercept = float(getattr(dcm, "RescaleIntercept", 0.0) or 0.0)
#     img = img * slope + intercept
#     if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
#         img = img.max() - img
#     return img  # HxW

# def _merge_masks(data):
#     """data['mask'] can be str or list[str]; load step makes them tensors.
#        If list, take max across the list (logical OR for binary masks)."""
#     import torch
#     m = data.get("mask")
#     if m is None:
#         return data
#     if isinstance(m, list):
#         # LoadImaged will have turned each path into a tensor already if applied per-element;
#         # If we apply LoadImaged once on the list, it returns a list of arrays.
#         # So merge here:
#         stacked = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in m], dim=0)
#         data["mask"] = stacked.max(dim=0).values
#     return data


# def make_train_transforms(input_shape, task, seg_target, num_classes):
#     keys = ["image"] + (["mask"] if task != "classification" else [])
#     t = [
#         LoadImaged(keys=keys, image_only=True),
#         EnsureChannelFirstd(keys=keys),
#         # Robust centering (choose one):
#         # Option A: per-image robust z-score via percentiles
#         ScaleIntensityRangePercentilesd(keys=["image"], lower=2, upper=98, b_min=-1.0, b_max=1.0, clip=True),
#         NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),  # ~z-score on nonzero region
#         EnsureTyped(keys=keys, dtype="torch.float32"),
#         RandFlipd(keys=keys, prob=0.5, spatial_axis=-2),
#         RandRotate90d(keys=keys, prob=0.5, spatial_axes=(-2, -1)),
#         Lambdad(keys=["mask"], func=lambda x: (x > 0.5).float(), allow_missing_keys=True),
#     ]
#     if task != "classification":
#         t.insert(0, Lambdad(keys=["mask"], func=lambda m: m, allow_missing_keys=True))  # placeholder
#         t.insert(3, Lambdad(keys=None, func=_merge_masks))  # merge after load, before aug
#     # add discretization post if we need indices target...
#     return Compose(t)


# def make_val_transforms(
#     *,
#     input_shape: Tuple[int, int] = (256, 256),
#     task: str = "multitask",
#     seg_target: str = "indices",
#     num_classes: int = 2,
#     bin_thresh: float = 0.5,
# ):
#     # same as train, without augmentation
#     return make_train_transforms(
#         input_shape=input_shape,
#         task=task,
#         seg_target=seg_target,
#         num_classes=num_classes,
#         bin_thresh=bin_thresh,
#         aug_flip_prob=0.0,
#         aug_rot90_prob=0.0,
#     )


# def make_train_transforms(
#     *,
#     input_shape: Tuple[int, int] = (256, 256),
#     task: str = "multitask",
#     seg_target: str = "indices",
#     num_classes: int = 2,
#     bin_thresh: float = 0.5,
#     aug_flip_prob: float = 0.5,
#     aug_rot90_prob: float = 0.5,
# ):
#     has_seg = task.lower() != "classification"
#     img_keys = ("image",)
#     mask_keys = ("mask",) if has_seg else tuple()

#     common = [
#         # load
#         Lambdad(keys=img_keys, func=_load_dicom, allow_missing_keys=True),
#         Lambdad(keys=mask_keys, func=_load_dicom, allow_missing_keys=True),
#         EnsureTyped(keys=img_keys + mask_keys, data_type="tensor", allow_missing_keys=True),
#         EnsureChannelFirstd(keys=img_keys, channel_dim=0, allow_missing_keys=True),       # -> [1,H,W]
#         EnsureChannelFirstd(keys=mask_keys, channel_dim=0, allow_missing_keys=True),      # -> [1,H,W]
#         ResizeD(keys=img_keys + mask_keys, spatial_size=input_shape, mode=("bilinear","nearest"), allow_missing_keys=True),
#         NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True, allow_missing_keys=True),
#     ]

#     aug = [
#         RandFlipd(keys=img_keys + mask_keys, prob=aug_flip_prob, spatial_axis=-2, allow_missing_keys=True),
#         RandRotate90d(keys=img_keys + mask_keys, prob=aug_rot90_prob, spatial_axes=(-2, -1), allow_missing_keys=True),
#     ] if has_seg else [
#         RandFlipd(keys=img_keys, prob=aug_flip_prob, spatial_axis=-2, allow_missing_keys=True),
#         RandRotate90d(keys=img_keys, prob=aug_rot90_prob, spatial_axes=(-2, -1), allow_missing_keys=True),
#     ]

#     post = []
#     if has_seg:
#         if seg_target == "indices":
#             if num_classes > 2:
#                 post += [AsDiscreted(keys=mask_keys, argmax=True, allow_missing_keys=True)]
#             else:
#                 post += [AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True)]
#             post += [SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True)]  # [1,H,W] -> [H,W]
#         else:
#             if num_classes > 2:
#                 post += [AsDiscreted(keys=mask_keys, to_onehot=num_classes, allow_missing_keys=True)]
#             else:
#                 post += [AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True)]

#     return Compose(common + aug + post)
