### Analysis & Implications of Baseline CNN Sweep Logs

#### **1. baseline\_shallow\_cnn (n=7)**

* **Validation Accuracy:** Mean ≈ 0.71 (up to 0.73),
  **Val AUC:** Mean ≈ 0.78 (up to 0.79),
  **Eval/Test:** Accuracy 0.70, AUC 0.76
* **Precision/Recall:** Mean precision 0.66, recall 0.60
* **Loss:** Validation loss is relatively low (mean ≈ 0.67)
* **Implication:**
  Even without dropout or augmentation, the shallow CNN achieves the highest AUC among baselines. However, the small run count (n=7) means this is less robust, and potential overfitting is likely due to lack of regularization.

---

#### **2. baseline\_cnn\_dropout (n=27)**

* **Validation Accuracy:** Mean ≈ 0.67,
  **Val AUC:** Mean ≈ 0.73–0.75
* **Precision/Recall:** Mean precision ≈ 0.61, recall ≈ 0.57
* **Loss:** Validation loss mean ≈ 0.74 (higher than others)
* **Implication:**
  Adding dropout (but no augmentation) stabilizes results over more runs, but doesn’t improve metrics compared to the shallow baseline. Validation loss is higher, and gains in robustness may be offset by insufficient regularization alone.

---

#### **3. baseline\_cnn\_dropout\_aug (n=107)**

* **Validation Accuracy:** Mean ≈ 0.60,
  **Val AUC:** Mean ≈ 0.61,
  **Max Val AUC:** 0.73
* **Precision/Recall:** Mean precision ≈ 0.49, recall ≈ 0.32
* **Loss:** Validation loss mean ≈ 0.67
* **Implication:**
  Adding augmentation with dropout yields more stable, generalizable runs but with a modest drop in mean metrics—likely due to regularization and increased difficulty. However, the best runs (upper quartile) approach the metrics of previous models, indicating that optimal hyperparameter tuning is critical when using augmentation.

---

#### **4. baseline\_cnn\_dropout\_aug\_rev (n=30)**

* **Validation Accuracy:** Mean ≈ 0.60,
  **Val AUC:** Mean ≈ 0.65,
  **Max Val AUC:** 0.73
* **Precision/Recall:** Mean precision ≈ 0.50, recall ≈ 0.30
* **Loss:** Validation loss mean ≈ 0.64 (lowest)
* **Implication:**
  This configuration achieves lower mean validation accuracy/AUC but has the lowest validation loss. The upper quartile for AUC overlaps with the best in other groups, suggesting a minority of well-tuned runs are competitive. Regularization is effective but may require finer hyperparameter control to avoid underfitting.

---

### **Overall Implications**

* **Highest Mean Performance:**
  The simple, shallow CNN (no dropout, no augmentation) achieves the highest mean validation accuracy and AUC, but at the cost of likely overfitting and limited robustness (due to low run count and lack of regularization).
* **Stability & Generalizability:**
  Introducing dropout and data augmentation makes results more stable across runs and likely generalizes better to new data, but may slightly lower average metrics unless hyperparameters are well-tuned.
* **Best-Case Performance:**
  All configurations can reach similar top validation AUC/accuracy with optimal settings, but robust regularization is essential for reliable, reproducible results.
* **Recommendation:**
  For future models, favor configurations with both dropout and augmentation, but invest in more systematic hyperparameter optimization to achieve consistently high performance.

---

**Summary Table of Means (Key Metrics):**

| Model                            | n   | Val Acc (mean) | Val AUC (mean) | Val Loss (mean) | Val Precision (mean) | Val Recall (mean) |
| -------------------------------- | --- | -------------- | -------------- | --------------- | -------------------- | ----------------- |
| baseline\_shallow\_cnn           | 7   | 0.71           | 0.78           | 0.67            | 0.66                 | 0.60              |
| baseline\_cnn\_dropout           | 27  | 0.67           | 0.73           | 0.74            | 0.61                 | 0.57              |
| baseline\_cnn\_dropout\_aug      | 107 | 0.60           | 0.61           | 0.67            | 0.49                 | 0.32              |
| baseline\_cnn\_dropout\_aug\_rev | 30  | 0.60           | 0.65           | 0.64            | 0.50                 | 0.30              |

---
