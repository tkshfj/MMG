# data_transforms.py
from __future__ import annotations
from typing import Tuple, List, Union
import numpy as np
import pydicom
import torch
from monai.transforms import (
    Compose, Lambdad, EnsureChannelFirstd, EnsureTyped,
    AsDiscreted, Resized, ScaleIntensityRangePercentilesd, RandSpatialCropd,
    # Orientationd, Spacingd, QuietOrientationD,
    MapTransform, NormalizeIntensityd, RandFlipd, SqueezeDimd
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
    Read DICOM mask(s) safely:
      - DO NOT apply slope/intercept or MONOCHROME1 inversion.
      - Rebase labels so min becomes 0 (handles {1,2} etc).
      - Binarize: >0 -> 1. Returns float32 HxW (0/1).
    If a list is passed, resize others to the first mask's shape (nearest) and OR-merge.
    """
    import cv2

    def _read_mask(path: str) -> np.ndarray:
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array.astype(np.int64)       # raw pixels
        arr = arr - arr.min()                        # rebase => background=0 even if {1,2}
        arr = (arr > 0).astype(np.float32)           # strict binary
        return arr

    if isinstance(obj, str):
        return _read_mask(obj)

    assert isinstance(obj, list) and len(obj) > 0, "mask list must be non-empty"
    base = _read_mask(obj[0])
    H, W = base.shape
    if len(obj) == 1:
        return base
    merged = base.copy()
    for p in obj[1:]:
        mi = _read_mask(p)
        if mi.shape != (H, W):
            mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, mi)
    return merged


# def read_and_merge_masks(obj: Union[str, List[str]]) -> np.ndarray:
#     """
#     Accepts a single path or a list of paths. If list, resize each to the first mask's
#     spatial size (nearest) and take elementwise max (logical OR for binary masks).
#     Returns a single float32 HxW array.
#     """
#     import cv2  # local import to avoid hard dep at import time
#     if isinstance(obj, str):
#         return read_dicom(obj).astype(np.float32)

#     assert isinstance(obj, list) and len(obj) > 0, "mask list must be non-empty"
#     base = read_dicom(obj[0]).astype(np.float32)
#     H, W = base.shape
#     if len(obj) == 1:
#         return base
#     merged = base.copy()
#     for p in obj[1:]:
#         mi = read_dicom(p).astype(np.float32)
#         if mi.shape != (H, W):
#             mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
#         merged = np.maximum(merged, mi)
#     return merged


def _ensure_cf_2d(keys=("image", "mask")):
    """
    Force channel-first for 2D grayscale inputs that arrive as [H, W].
    We don't rely on metadata; we tell MONAI there is *no channel* yet.
    """
    k = list(keys)
    return [
        EnsureChannelFirstd(keys=[k[0]], channel_dim="no_channel"),
        EnsureChannelFirstd(keys=[k[1]], channel_dim="no_channel"),
    ]


class AssertSameSpatialD(MapTransform):
    def __init__(self, keys=("image", "mask")):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        if "image" in d and "mask" in d and d["image"] is not None and d["mask"] is not None:
            # compare spatial dims only (ignore channels/batch)
            im_sp = tuple(d["image"].shape[-2:])
            mk_sp = tuple(d["mask"].shape[-2:])
            assert im_sp == mk_sp, f"image/mask spatial mismatch: {im_sp} vs {mk_sp}"
        return d


class EnsureSameSpatialD(MapTransform):
    """
    Keep image/mask spatial shapes in sync (ignoring channel dim).
    fix:
      - 'none' -> raise on mismatch (strict mode)
      - 'resize_mask_to_image' -> resize mask to image spatial size (nearest)
    Place this BEFORE RandCropByPosNegLabeld.
    """
    def __init__(self, keys=("image", "mask"), *, fix="resize_mask_to_image", warn=True):
        super().__init__(keys)
        if len(self.keys) != 2:
            raise ValueError("EnsureSameSpatialD expects exactly 2 keys, e.g. ('image','mask').")
        if fix not in ("none", "resize_mask_to_image"):
            raise ValueError("fix must be 'none' or 'resize_mask_to_image'.")
        self.fix = fix
        self.warn = bool(warn)

    @staticmethod
    def _spatial(shape):
        # Always return (H, W), ignoring channels and any leading dims
        return tuple(shape[-2:])

    def __call__(self, data):
        d = dict(data)
        if self.keys[0] not in d or self.keys[1] not in d:
            return d

        img, msk = d[self.keys[0]], d[self.keys[1]]
        shp_i, shp_m = tuple(getattr(img, "shape", ())), tuple(getattr(msk, "shape", ()))
        if not shp_i or not shp_m:
            return d
        sp_i, sp_m = self._spatial(shp_i), self._spatial(shp_m)
        if sp_i == sp_m:
            return d

        if self.fix == "resize_mask_to_image":
            # Only touch the mask; keep labels crisp (nearest)
            res = Resized(keys=[self.keys[1]], spatial_size=sp_i, mode="nearest", anti_aliasing=False)
            d = res(d)
            if self._spatial(tuple(d[self.keys[1]].shape)) != sp_i:
                raise RuntimeError("EnsureSameSpatialD: failed to fix mask spatial size.")
            if self.warn:
                print(f"[EnsureSameSpatialD] Auto-fixed mask spatial {sp_m} -> {sp_i} (img={shp_i}).")
            return d

        raise RuntimeError(
            "EnsureSameSpatialD: image/mask spatial mismatch.\n"
            f"  image shape={shp_i}, mask shape={shp_m}\n"
            "Set fix='resize_mask_to_image' or ensure upstream transforms keep them aligned."
        )


def _make_transforms(
    *,
    input_shape: tuple[int, int],
    task: str,
    seg_target: str,
    num_classes: int,
    bin_thresh: float,
    augment: bool,
    crop_train: bool = False,
    lesion_class: int = 1,      # (kept for signature compatibility)
    pos_ratio: float = 0.5,
    num_samples: int = 4,
) -> Compose:
    has_masks = task.lower() in {"segmentation", "multitask"}
    img_keys = ["image"]
    mask_keys = ["mask"] if has_masks else []
    both_keys = img_keys + mask_keys
    # sp_mode = ("bilinear", "nearest") if has_masks else "bilinear"

    t: list = []

    # 1) Load
    t.append(Lambdad(keys=img_keys, func=read_dicom, allow_missing_keys=False))
    if has_masks:
        t.append(Lambdad(keys=mask_keys, func=read_and_merge_masks, allow_missing_keys=True))
        # Ensure binary immediately after load
        t.append(AsDiscreted(keys=mask_keys, threshold=0.5, allow_missing_keys=True))

    # # 1) IO / load — unchanged
    # t.append(Lambdad(keys=img_keys, func=read_dicom, allow_missing_keys=False))
    # if has_masks:
    #     t.append(Lambdad(keys=mask_keys, func=read_and_merge_masks, allow_missing_keys=True))

    # 2) Channel-first (explicit, already good)
    if has_masks:
        t += _ensure_cf_2d(("image", "mask"))
    else:
        t.append(EnsureChannelFirstd(keys=img_keys, channel_dim="no_channel", allow_missing_keys=True))

    # # 2) Channel-first for BOTH (explicit channel_dim so we never depend on meta)
    # if has_masks:
    #     t += _ensure_cf_2d(("image", "mask"))
    # else:
    #     t.append(EnsureChannelFirstd(keys=img_keys, channel_dim="no_channel", allow_missing_keys=True))

    # After channel-first
    # t.append(Orientationd(keys=both_keys, axcodes="RAS", allow_missing_keys=True))
    # QuietOrientationD(keys=both_keys, axcodes_2d="RA", axcodes_3d="RAS", allow_missing_keys=True)
    # t.append(Spacingd(keys=both_keys, pixdim=(1.0, 1.0), mode=sp_mode, allow_missing_keys=True))

    # 3) Keep image/mask spatial shapes in sync BEFORE any crop
    if has_masks:
        # t.append(EnsureSameSpatialD(keys=("image", "mask"), fix="resize_mask_to_image", warn=True))
        t.append(EnsureSameSpatialD(keys=("image", "mask"), fix="resize_mask_to_image", warn=False))

    # 4) Train/Eval path
    if augment and has_masks and crop_train:
        t.append(RandSpatialCropd(keys=both_keys, roi_size=tuple(input_shape), random_center=True, random_size=False))
        # Re-binarize after crop (safety; nearest should preserve, but be explicit)
        t.append(AsDiscreted(keys=mask_keys, threshold=0.5, allow_missing_keys=True))
        t.append(AssertSameSpatialD(keys=("image", "mask")))
    else:
        t.append(Resized(keys=["image"], spatial_size=tuple(input_shape), mode="bilinear", anti_aliasing=False, allow_missing_keys=False))
        if has_masks:
            t.append(Resized(keys=["mask"], spatial_size=tuple(input_shape), mode="nearest", anti_aliasing=False, allow_missing_keys=False))
            # Re-binarize after resize (safety)
            t.append(AsDiscreted(keys=mask_keys, threshold=0.5, allow_missing_keys=True))
            t.append(AssertSameSpatialD(keys=("image", "mask")))

    # # 4) Train vs Eval spatial path
    # if augment and has_masks and crop_train:
    #     # Paired crop
    #     t.append(RandSpatialCropd(
    #         keys=both_keys,
    #         roi_size=tuple(input_shape),
    #         random_center=True,
    #         random_size=False
    #     ))
    #     # Assert immediately after the paired geometry op
    #     t.append(AssertSameSpatialD(keys=("image", "mask")))
    # else:
    #     # Eval / no-crop: per-key, deterministic resize
    #     t.append(Resized(keys=["image"], spatial_size=tuple(input_shape), mode="bilinear", anti_aliasing=False, allow_missing_keys=False))
    #     if has_masks:
    #         t.append(Resized(keys=["mask"], spatial_size=tuple(input_shape), mode="nearest", anti_aliasing=False, allow_missing_keys=False))
    #         # Assert immediately after paired resize
    #         t.append(AssertSameSpatialD(keys=("image", "mask")))

    # 5) Intensity — image only
    # If some images are constant (divide-by-zero warnings), NormalizeIntensityd(nonzero=True) will skip zeros.
    t.append(ScaleIntensityRangePercentilesd(keys=img_keys, lower=2, upper=98, b_min=-1.0, b_max=1.0, clip=True))
    t.append(NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True))

    # 6) Ensure dtypes
    t.append(EnsureTyped(keys=img_keys, dtype=torch.float32, allow_missing_keys=True))
    if has_masks:
        # keep mask flexible until final formatting below
        t.append(EnsureTyped(keys=mask_keys, allow_missing_keys=True))

    # 7) Light geometric augments — apply to BOTH to preserve alignment
    if augment:
        t.append(RandFlipd(keys=both_keys, prob=0.5, spatial_axis=-2))

    # 8) Final mask formatting for model/loss
    if has_masks:
        if seg_target == "indices":
            if num_classes <= 2:
                # binary 0/1
                t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))
            else:
                # multi-class -> indices via argmax
                t.append(AsDiscreted(keys=mask_keys, argmax=True, allow_missing_keys=True))
            # squeeze an accidental channel and cast to long
            t.append(SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True))
            t.append(EnsureTyped(keys=mask_keys, dtype=torch.long, allow_missing_keys=True))
        else:
            # channels (one-hot for >2), keep float
            if num_classes <= 2:
                t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))
            else:
                # Ensure index mask first, then one-hot
                t.append(SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True))
                t.append(AsDiscreted(keys=mask_keys, to_onehot=num_classes, allow_missing_keys=True))
            t.append(EnsureTyped(keys=mask_keys, dtype=torch.float32, allow_missing_keys=True))

    return Compose(t)


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

# # Core builder
# def _make_transforms(
#     *,
#     input_shape: Tuple[int, int],
#     task: str,
#     seg_target: str,
#     num_classes: int,
#     bin_thresh: float,
#     augment: bool,
#     crop_train: bool = False,
#     lesion_class: int = 1,
#     pos_ratio: float = 0.5,
#     num_samples: int = 4,
# ) -> Compose:
#     has_masks = task.lower() in {"segmentation", "multitask"}
#     img_keys = ["image"]
#     mask_keys = ["mask"] if has_masks else []
#     both_keys = img_keys + mask_keys

#     t = []

#     # IO / load (top-level functions → picklable)
#     t.append(Lambdad(keys=img_keys, func=read_dicom, allow_missing_keys=False))
#     if mask_keys:
#         t.append(Lambdad(keys=mask_keys, func=read_and_merge_masks, allow_missing_keys=True))

#     # Adds a channel if missing
#     t.append(Lambdad(keys=both_keys, func=add_channel_if_2d, allow_missing_keys=True))

#     # Size/cropping
#     if augment and has_masks and crop_train:
#         # Make mask binary so positive voxels are 1, background 0
#         # This ensures RandCropByPosNegLabeld treats label>0 as "pos".
#         t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))

#         # Convert desired ratio to counts (no pos_ratio arg)
#         pos_k, neg_k = _pos_neg_from_ratio(pos_ratio, num_samples)

#         # Lesion-aware sampler: picks centers from mask>0 for 'pos_k' and from mask==0 for 'neg_k'
#         t.append(RandCropByPosNegLabeld(
#             keys=both_keys,
#             label_key="mask",
#             spatial_size=tuple(input_shape),
#             pos=pos_k,                       # counts, NOT class id
#             neg=neg_k,                       # counts, NOT class id
#             num_samples=int(num_samples),
#             image_key="image",
#             image_threshold=0.0,
#         ))
#         # No resize after cropping; patches are already input_shape [H,W] = input_shape.
#     else:
#         # Val/test (or train without crop): global resize
#         t.append(
#             ResizeD(
#                 keys=both_keys,
#                 spatial_size=input_shape,
#                 mode=("bilinear", "nearest") if has_masks else "bilinear",
#                 allow_missing_keys=True,
#             )
#         )

#     # Intensity & typing (image only for intensity)
#     t.append(ScaleIntensityRangePercentilesd(keys=img_keys, lower=2, upper=98, b_min=-1.0, b_max=1.0, clip=True))
#     t.append(NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True))
#     t.append(EnsureTyped(keys=both_keys, dtype=torch.float32, allow_missing_keys=True))

#     # Augmentations (train only)
#     if augment:
#         t.append(RandFlipd(keys=both_keys, prob=0.5, spatial_axis=-2))
#         t.append(RandRotate90d(keys=both_keys, prob=0.5, spatial_axes=(-2, -1)))

#     # Mask post-processing (final target formatting for the model & loss)
#     if mask_keys:
#         if seg_target == "indices":
#             if num_classes <= 2:
#                 # keep binary 0/1; we already discretized earlier for the crop
#                 t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))
#             else:
#                 t.append(AsDiscreted(keys=mask_keys, argmax=True, allow_missing_keys=True))
#             # model usually expects [H,W] integer labels for indices
#             t.append(SqueezeDimd(keys=mask_keys, dim=0, allow_missing_keys=True))
#             t.append(EnsureTyped(keys=mask_keys, dtype=torch.long, allow_missing_keys=True))
#         else:
#             # channels target -> keep float (optionally one-hot for >2 classes)
#             if num_classes <= 2:
#                 t.append(AsDiscreted(keys=mask_keys, threshold=bin_thresh, allow_missing_keys=True))
#             else:
#                 t.append(AsDiscreted(keys=mask_keys, to_onehot=num_classes, allow_missing_keys=True))

#     return Compose(t)
