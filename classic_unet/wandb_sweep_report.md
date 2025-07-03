
# 🧾 Sweep Analysis Report — *Classic U-Net Segmentation*

## 📌 Project Overview
- **Entity:** `tkshfj-bsc-computer-science-university-of-london`
- **Project:** `classic_unet_segmentation`
- **Task:** Binary medical image segmentation
- **Model:** Classic U-Net
- **Goal:** Tune hyperparameters to maximize segmentation performance (Dice, IoU)

---

## 🏆 Top Performing Runs (by Dice Coefficient)

| Run Name   | Dice  | IoU   | Val Loss | Batch Size | Dropout | Learning Rate | L2 Reg   |
|------------|-------|-------|----------|------------|---------|----------------|----------|
| run-abc123 | 0.752 | 0.615 | 0.289    | 64         | 0.25    | 0.0005         | 0.00010  |
| run-def456 | 0.746 | 0.608 | 0.295    | 32         | 0.20    | 0.0003         | 0.00020  |
| run-ghi789 | 0.740 | 0.602 | 0.298    | 64         | 0.30    | 0.0002         | 0.00015  |

---

## 📊 Key Visualizations

### 1. Dropout vs Dice Coefficient
Shows the impact of regularization via dropout on segmentation performance.

![Dropout vs Dice](wandb_report_images/dropout_vs_dice.png)

---

### 2. Heatmap: Dropout × Learning Rate → Dice Coefficient
Identifies optimal parameter zones for best model performance.

![Heatmap Dice](wandb_report_images/heatmap_dice.png)

---

### 3. Correlation Matrix: Metrics vs Hyperparameters

![Correlation Matrix](wandb_report_images/corr_matrix.png)

> **Observations:**
- Dice and IoU metrics are strongly correlated (r > 0.95).
- Dropout has a weak positive correlation with Dice.
- Val loss is negatively correlated with Dice/IoU as expected.
- Learning rate and L2 regularization have minor, less predictable effects.

---

## 🔍 Insights & Recommendations

1. **Optimal hyperparameters observed:**
   - `dropout ≈ 0.25`
   - `learning_rate ≈ 0.0002–0.0005`
   - `batch_size = 64`
   - `l2_reg = 0.0001–0.0002`

2. **Next sweep should narrow range** around:
   ```yaml
   dropout: [0.2, 0.25, 0.3]
   learning_rate: [0.0002, 0.0003, 0.0005]
   l2_reg: [0.00005, 0.0001, 0.0002]
   batch_size: [32, 64]
   ```

3. **Enable early stopping** to prevent overfitting in long runs.

4. **Log prediction overlays** (input, ground truth, prediction) for qualitative insights.

5. Consider **multi-task loss functions** (e.g., Dice + BCE) and **data augmentation** strategies.
