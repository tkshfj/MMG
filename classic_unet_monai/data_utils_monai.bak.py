import pandas as pd
import torch
from data_utils import MammoSegmentationDataset, get_monai_transforms

def build_dataloaders(
    metadata_csv,
    input_shape=(256,256),
    batch_size=8,
    task="segmentation",
    split=(0.7, 0.15, 0.15)
):
    df = pd.read_csv(metadata_csv)
    n = len(df)
    train_n = int(n * split[0])
    val_n = int(n * split[1])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df.iloc[:train_n]
    val_df = df.iloc[train_n:train_n+val_n]
    test_df = df.iloc[train_n+val_n:]

    train_tf, val_tf = get_monai_transforms(task, input_shape)

    train_ds = MammoSegmentationDataset(train_df, input_shape, task, transform=train_tf)
    val_ds   = MammoSegmentationDataset(val_df, input_shape, task, transform=val_tf)
    test_ds  = MammoSegmentationDataset(test_df, input_shape, task, transform=val_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# import pandas as pd
# import numpy as np
# import cv2
# from ast import literal_eval
# from monai.transforms import (
#     Compose, LoadImaged, EnsureTyped, EnsureChannelFirstd, LambdaD, ScaleIntensityd, 
#     RandFlipd, RandRotate90d, Resized, ToNumpyd, ToTensord, MapTransform
# )
# from monai.data import Dataset, DataLoader

# class MergeMultiMaskd(MapTransform):
#     """
#     Custom transform to load and merge multiple mask DICOMs into a single mask.
#     Accepts both np.ndarray and MetaTensor, returns np.ndarray [H, W] (float32).
#     """
#     def __init__(self, keys, output_shape):
#         super().__init__(keys)
#         self.output_shape = output_shape

#     def __call__(self, data):
#         d = dict(data)
#         mask_paths = d.get("mask_paths", [])
#         if isinstance(mask_paths, str):
#             try:
#                 mask_paths = literal_eval(mask_paths)
#             except Exception as e:
#                 print(f"[DICOM ERROR] mask_paths not valid list: {e}")
#                 mask_paths = []
#         mask = np.zeros(self.output_shape, dtype=np.float32)
#         mask_found = False
#         for mpath in mask_paths:
#             try:
#                 m = LoadImaged(keys=["temp_mask"], reader="PydicomReader")({"temp_mask": mpath})["temp_mask"]
#                 # Accept MetaTensor or np.ndarray
#                 if hasattr(m, "numpy"):
#                     m = m.detach().cpu().numpy()  # MetaTensor to np.ndarray
#                 if m.ndim == 3 and m.shape[-1] == 1:
#                     m = m[..., 0]
#                 if m.shape != self.output_shape:
#                     m = cv2.resize(m, (self.output_shape[1], self.output_shape[0]), interpolation=cv2.INTER_NEAREST)
#                 mask = np.maximum(mask, m)
#                 mask_found = True
#             except Exception as e:
#                 print(f"[DICOM ERROR] {mpath}: {e}")
#         if not mask_found:
#             print(f"[DICOM ERROR] No valid masks found for row. Returning all-zero mask.")
#         mask = (mask > 0.5).astype(np.float32)
#         if mask.ndim != 2:
#             print(f"[DICOM ERROR] Merged mask has wrong ndim ({mask.ndim}), fixing to {self.output_shape}")
#             mask = np.zeros(self.output_shape, dtype=np.float32)
#         d["mask"] = mask
#         return d

# class PrintTypesShapesd(MapTransform):
#     """Debug: Print type and shape of selected keys if debug=True."""
#     def __init__(self, keys, debug=False):
#         super().__init__(keys)
#         self.debug = debug
#     def __call__(self, data):
#         if self.debug:
#             for k in self.keys:
#                 v = data.get(k, None)
#                 print(f"DEBUG: Key: {k}, Type: {type(v)}, Shape: {getattr(v, 'shape', 'NA')}, Dtype: {getattr(v, 'dtype', 'NA')}")
#         return data

# class PrintShapeD(MapTransform):
#     def __init__(self, keys):
#         super().__init__(keys)
#     def __call__(self, data):
#         for k in self.keys:
#             v = data.get(k, None)
#             print(f"[DEBUG] {k} shape: {getattr(v, 'shape', None)}")
#         return data

# def force_2d_slice(x):
#     import numpy as np
#     # Convert to numpy array
#     if hasattr(x, "detach") and hasattr(x, "cpu"):
#         x = x.detach().cpu().numpy()
#     elif hasattr(x, "numpy"):
#         x = x.numpy()
#     # If 4D, squeeze out singletons
#     x = np.squeeze(x)
#     # If 3D (multi-frame), use center slice
#     if x.ndim == 3:
#         x = x[x.shape[0] // 2]
#     # Now should be (H, W)
#     if x.ndim != 2:
#         raise ValueError(f"force_2d_slice failed, got shape {x.shape}")
#     return x.astype(np.float32)

# def get_monai_transforms(task="segmentation", input_shape=(256,256), debug=False):
#     keys = ["image", "mask"] if task in ["segmentation", "multitask"] else ["image"]
#     common = [
#         LoadImaged(keys=["image"], reader="PydicomReader"),
#         MergeMultiMaskd(keys=["mask_paths"], output_shape=input_shape),
#         ToNumpyd(keys=["image", "mask"]),
#         LambdaD(keys=["image"], func=force_2d_slice),
#         LambdaD(keys=["mask"], func=force_2d_slice),
#         Resized(keys=["image"], spatial_size=input_shape, mode="bilinear"),
#         Resized(keys=["mask"], spatial_size=input_shape, mode="nearest"),
#         LambdaD(keys=["mask"], func=lambda x: (x > 0).astype(np.float32)),
#         EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
#         ScaleIntensityd(keys=["image"]),
#         ToTensord(keys=["image", "mask"])
#     ]
#     train_tf = Compose(common + [
#         RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
#         RandRotate90d(keys=keys, prob=0.5)
#     ])
#     val_tf = Compose(common)
#     return train_tf, val_tf


# def build_dataloaders(
#     metadata_csv, input_shape=(256, 256), batch_size=8, task="segmentation", split=(0.7, 0.15, 0.15), debug=False
# ):
#     df = pd.read_csv(metadata_csv)
#     n = len(df)
#     train_n = int(n * split[0])
#     val_n = int(n * split[1])
#     df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#     if task in ["segmentation", "multitask"]:
#         dicts = df[["image_path", "mask_paths"] + (["label"] if task == "multitask" else [])].rename(
#             columns={"image_path": "image"}
#         ).to_dict(orient="records")
#     else:
#         dicts = df[["image_path", "label"]].rename(columns={"image_path": "image"}).to_dict(orient="records")

#     train_dicts = dicts[:train_n]
#     val_dicts = dicts[train_n:train_n + val_n]
#     test_dicts = dicts[train_n + val_n:]

#     train_tf, val_tf = get_monai_transforms(task, input_shape, debug=debug)
#     train_ds = Dataset(data=train_dicts, transform=train_tf)
#     val_ds = Dataset(data=val_dicts, transform=val_tf)
#     test_ds = Dataset(data=test_dicts, transform=val_tf)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader, test_loader
