# **Sweep Results Evaluation Report**

## **Overview**

This report outlines the approach and results of hyperparameter sweep analysis for the **classic\_unet\_segmentation** project, leveraging the W\&B API to programmatically evaluate, aggregate, and visualize performance across different hyperparameter combinations. The sweep used a Bayesian optimization strategy to maximize the validation Dice coefficient (`val_dice_coefficient`).

---

## **Sweep Configuration Summary**

**Objective:**
Maximize `val_dice_coefficient` (validation Dice score).

**Hyperparameters:**

* **batch\_size**: \[4, 8, 16] (discrete)
* **learning\_rate**: 0.0001–0.01 (uniform)
* **dropout**: 0.1–0.5 (uniform)
* **l2\_reg**: 0.00001–0.001 (uniform)
* **epochs**: 40 (fixed)
* **input\_shape**: \[256, 256, 1] (fixed)
* **task**: segmentation (fixed)

---

## **Analysis Workflow**

### **1. Data Acquisition**

All sweep runs were fetched using the W\&B API. For each run, relevant metrics and hyperparameters were extracted and compiled into a pandas DataFrame for analysis.

<details>
<summary><strong>Sample Extraction Code</strong></summary>

```python
import wandb
import pandas as pd

api = wandb.Api()
ENTITY = "tkshfj-bsc-computer-science-university-of-london"
PROJECT = "classic_unet_segmentation"
SWEEP_ID = "your-sweep-id"  # Replace with actual sweep id

runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"sweep": {"$eq": SWEEP_ID}})
records = []
for run in runs:
    data = {**run.summary, **run.config}
    data["run_name"] = run.name
    data["url"] = run.url
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    records.append(data)
df = pd.DataFrame(records)
```

</details>

---

### **2. Best Run Identification**

The run achieving the **highest validation Dice coefficient** was identified, along with its hyperparameters:

```python
best_run = df.loc[df['val_dice_coefficient'].idxmax()]
print(best_run[['batch_size', 'learning_rate', 'dropout', 'l2_reg', 'val_dice_coefficient', 'url']])
```

**Result:**

* **Best val\_dice\_coefficient:** *\[Value]*
* **Hyperparameters:**

  * batch\_size: *\[Value]*
  * learning\_rate: *\[Value]*
  * dropout: *\[Value]*
  * l2\_reg: *\[Value]*
  * [Link to run](best_run[%22url%22])

---

### **3. Hyperparameter Impact Analysis**

#### **A. Batch Size**

Batch size performance was aggregated to evaluate its effect on Dice score:

```python
agg = df.groupby('batch_size')['val_dice_coefficient'].agg(['mean', 'std', 'max', 'count'])
print(agg)
```

#### **B. Continuous Parameters**

The relationship between each continuous hyperparameter and validation Dice coefficient was visualized:

* **Learning Rate:**
  Log-scale scatter plot showed performance across the search space.
* **Dropout:**
  Scatter plot highlighted the effect of regularization.
* **L2 Regularization:**
  Log-scale scatter plot to visualize impact on generalization.

<details>
<summary><strong>Sample Plot Code</strong></summary>

```python
import matplotlib.pyplot as plt

plt.scatter(df['learning_rate'], df['val_dice_coefficient'])
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Dice Coefficient')
plt.title('Learning Rate vs. Dice')
plt.show()
```

</details>

---

### **4. Top Runs**

A table of the **top 10 runs** sorted by validation Dice was generated to facilitate model checkpoint access and comparison.

```python
top_n = 10
top = df.sort_values('val_dice_coefficient', ascending=False).head(top_n)
print(top[['batch_size', 'learning_rate', 'dropout', 'l2_reg', 'val_dice_coefficient', 'url']])
```

---

### **5. Multi-Parameter Visualization**

A parallel coordinates plot provided an overview of how combinations of hyperparameters led to higher performance.

<details>
<summary><strong>Sample Code</strong></summary>

```python
from pandas.plotting import parallel_coordinates

cols = ['val_dice_coefficient', 'batch_size', 'learning_rate', 'dropout', 'l2_reg']
data = df[cols].dropna()
data['dice_cat'] = pd.qcut(data['val_dice_coefficient'], 4, labels=False)
parallel_coordinates(data, 'dice_cat')
plt.title('Hyperparameters and Dice (Color: Quartile)')
plt.show()
```

</details>

---

### **6. Correlation Analysis**

To identify which hyperparameters most strongly influenced Dice, correlations were computed:

```python
print(df[['val_dice_coefficient', 'learning_rate', 'dropout', 'l2_reg']].corr())
```

---

## **Findings and Recommendations**

* The **best performance** was achieved with:
  *(fill in actual values after analysis)*.
* **Batch size:** Showed (describe any trend—e.g., higher/lower is better or no clear pattern).
* **Learning rate:** (Describe whether lower/higher rates performed better or if there’s a sweet spot.)
* **Dropout & L2:** (Summarize regularization trends—e.g., moderate values help, extremes hurt.)
* For future sweeps, consider narrowing the parameter space around high-performing values for finer optimization.

---

## **Conclusion**

By programmatically fetching and visualizing sweep results:

* The most effective hyperparameter combinations for segmentation with classic UNet have been identified.
* The influence of each parameter on model performance has been clarified, enabling informed choices for future training and experimentation.

**All analysis can be re-run or automated with the provided scripts, allowing for efficient hyperparameter tuning and reproducibility.**
