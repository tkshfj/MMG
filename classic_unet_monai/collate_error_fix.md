# **Refactor Report: Robust Medical Image Data Pipeline with MONAI**

## **Background**

We have encountered persistent batching/collate errors (`stack expects each tensor to be equal size...`) in our MONAI-based segmentation pipeline (`data_utils_monai.py`). Our analysis shows these errors arise from inconsistent sample shapes reaching the DataLoader, due to edge cases or bugs in the MONAI transform chain.

Our custom PyTorch dataset class in `data_utils.py` (`MammoSegmentationDataset`) does **not** produce these errors, as it strictly enforces output shape and type before batching.

---

## **Current State**

* **`data_utils_monai.py`**
  Relies on MONAI’s transform pipeline for all image/mask loading, preprocessing, and augmentation. Shape consistency is assumed, but not always guaranteed, leading to batching errors.

* **`data_utils.py`**
  Uses a custom Dataset class (`MammoSegmentationDataset`) that:

  * Loads images/masks.
  * Resizes them to target dimensions using OpenCV.
  * Merges multi-mask data.
  * Expands dimensions as needed for `[C, H, W]` format.
  * Ensures every sample is uniform **before batching**.

---

## **Implications**

* Our hand-written approach is robust and safe, but less modular.
* The MONAI transform pipeline is modular and powerful, but more fragile:

  * Any transform outputting the wrong shape allows errors to propagate until batching.
* Hybridization can combine the strengths of both approaches:

  * We use robust, explicit data loading and shaping.
  * We use MONAI’s advanced augmentation as a `transform` step.

---

## **Recommended Refactor**

### **Action: Refactor `data_utils_monai.py` as Follows**

1. **Replace MONAI-native Dataset usage** with our robust `MammoSegmentationDataset` (from `data_utils.py`).
2. **Use MONAI’s `Compose` transforms for augmentation and normalization**, passing them as the `transform` parameter to our custom Dataset.
3. **Retain PyTorch’s standard DataLoader** for batching.

### **Why?**

* This guarantees that every sample is shape- and type-consistent before batching, **eliminating collate errors**.
* We retain all the flexibility and power of MONAI’s augmentation/normalization transforms.
* Future transform or model changes are less likely to break our data pipeline.

---

## **Refactor Steps (Summary Table)**

| Step                             | From                     | To                                        |
| -------------------------------- | ------------------------ | ----------------------------------------- |
| Data loading/shaping             | MONAI Dataset/Transforms | Custom Dataset (MammoSegmentationDataset) |
| Augmentation (flip, scale, etc.) | MONAI Compose            | MONAI Compose (as `transform`)            |
| Batching                         | MONAI DataLoader         | PyTorch DataLoader                        |

---

## **Sample Code (Hybrid Pattern)**

```python
from data_utils import MammoSegmentationDataset, get_monai_transforms

# Prepare DataFrames from CSV (train_df, val_df, test_df)

train_tf, val_tf = get_monai_transforms(task='segmentation', input_shape=(256,256))

train_ds = MammoSegmentationDataset(train_df, input_shape=(256,256), task='segmentation', transform=train_tf)
val_ds   = MammoSegmentationDataset(val_df,   input_shape=(256,256), task='segmentation', transform=val_tf)
test_ds  = MammoSegmentationDataset(test_df,  input_shape=(256,256), task='segmentation', transform=val_tf)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=8, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=8, shuffle=False)
```

---

## **Expected Outcomes**

* All images/masks in every batch will have consistent shape `[1, H, W]` (e.g., `[1, 256, 256]`).
* Collate/batching errors will be eliminated.
* Our pipeline remains modular and extensible, with all the advantages of MONAI transforms.

---

## **Next Steps**

1. **Implement the refactor as described above in `data_utils_monai.py`.**
2. **Test with debug prints or assertions to confirm that every sample has the correct shape before batching.**
3. **Gradually add/expand MONAI transform use as needed, while retaining robust shape checks.**

---

## **Conclusion**

By combining our reliable, custom data loading logic with MONAI’s augmentation tools, we achieve a robust, flexible, and error-free data pipeline. This approach is scalable for future changes and will prevent data-related training interruptions.
