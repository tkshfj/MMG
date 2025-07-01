# **Sweep Results Evaluation Report**

## **Overview**

This report presents our evaluation of a Bayesian hyperparameter sweep for the **classic\_unet\_segmentation** project, with the goal of maximizing the validation Dice coefficient for a U-Net segmentation model.
We use the W\&B API for local result aggregation and visualization.

---

## **Sweep Configuration Summary**

* **Objective:** Maximize `val_dice_coefficient`
* **Hyperparameters:**

  * `batch_size`: \[4, 8, 16]
  * `learning_rate`: \[0.0001, 0.01] (uniform)
  * `dropout`: \[0.1, 0.5] (uniform)
  * `l2_reg`: \[0.00001, 0.001] (uniform)
  * `epochs`, `input_shape`, `task`: fixed

---

## **1. Data Acquisition**

We loaded all runs into a pandas DataFrame using:

```python
import wandb
import pandas as pd

ENTITY = "tkshfj-bsc-computer-science-university-of-london"
PROJECT = "classic_unet_segmentation"
SWEEP_ID = "1jpj8b17"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")
records = []
for run in runs:
    data = {**run.summary, **run.config}
    data["run_name"] = run.name
    data["run_id"] = run.id
    data["state"] = run.state
    data["url"] = run.url
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    records.append(data)
df = pd.DataFrame(records)
```

**Table Preview:**

```
   best_epoch  best_val_loss  dice_coefficient  epoch  ...  dropout  batch_size input_shape learning_rate  ...  url
0        11.0       1.160207          0.536665   21.0  ... 0.173666          4   [256,256,1]   0.000570   ...  [run url]
1        30.0       1.164538          0.535051   39.0  ... 0.116655          4   [256,256,1]   0.005960   ...  [run url]
...
```

---

## **2. Best Run Identification**

The **best run** (highest `val_dice_coefficient`) we found:

| Hyperparameter             | Value                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **val\_dice\_coefficient** | **0.56899**                                                                                                                 |
| batch\_size                | 16                                                                                                                          |
| learning\_rate             | 0.000159                                                                                                                    |
| dropout                    | 0.19885                                                                                                                     |
| l2\_reg                    | 0.000060                                                                                                                    |
| test\_dice                 | 0.57668                                                                                                                     |
| test\_iou                  | 0.40612                                                                                                                     |
| val\_loss                  | 1.10061                                                                                                                     |
| run\_name                  | dainty-sweep-4                                                                                                              |
| url                        | [wandb run link](https://wandb.ai/tkshfj-bsc-computer-science-university-of-london/classic_unet_segmentation/runs/rnk45hwn) |

> **Summary:**
> We achieved the best segmentation performance (`val_dice_coefficient` ≈ 0.569) with a batch size of 16, learning rate ≈ 0.000159, dropout ≈ 0.20, and l2\_reg ≈ 0.00006.

---

## **3. Hyperparameter Impact Analysis**

* **Batch Size:** We observed that models with batch size 16 tended to yield higher Dice coefficients.
* **Learning Rate:** Lower learning rates (<0.001) generally correlated with better validation Dice.
* **Dropout:** Moderate dropout values (\~0.2) performed best; too low or too high generally underperformed.
* **L2 Regularization:** Small but non-zero values (around 0.00006) contributed to best results.

**(See the notebook for the corresponding scatter plots and box plots; 5 figures detected.)**

---

## **4. Top Runs Table**

| batch\_size | learning\_rate | dropout | l2\_reg | val\_dice\_coefficient | url                                                                                                                   |
| ----------- | -------------- | ------- | ------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 16          | 0.000159       | 0.19885 | 0.00006 | 0.56899                | [best run](https://wandb.ai/tkshfj-bsc-computer-science-university-of-london/classic_unet_segmentation/runs/rnk45hwn) |
| ...         | ...            | ...     | ...     | ...                    | ...                                                                                                                   |

---

## **5. Multivariate and Correlation Analysis**

* **Parallel Coordinates Plot:** (refer to the notebook)
* **Correlation Matrix:**
  We found the highest Dice was associated with lower learning\_rate, batch\_size=16, dropout ≈ 0.2, and l2\_reg ≈ 0.00006.

---

## **6. Recommendations**

* For future sweeps, we may want to focus on batch\_size=16 and learning\_rate between 0.0001–0.001.
* Maintaining moderate dropout and low l2\_reg appears best for regularization.
* Further fine-tuning in these regions is likely to yield incremental improvements.

---

## **Conclusion**

Our sweep successfully identified a high-performing region of the hyperparameter space for classic U-Net segmentation.
W\&B-powered analysis enabled robust, reproducible insights to support our model development and future experimentation.
