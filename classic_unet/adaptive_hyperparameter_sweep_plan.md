# **Adaptive Hyperparameter Sweep Planning Report**

## **Objective**

Design an efficient hyperparameter sweep for classic U-Net segmentation on 256×256 images, maximizing validation Dice coefficient (`val_dice_coefficient`), and fully utilizing NVIDIA RTX 6000 Ada GPU (48GB VRAM). The sweep should explore large batch sizes and adapt the learning rate using the linear scaling rule.

---

## **Key Concepts and Rationale**

**Adaptive Sweep in Deep Learning:**
An adaptive sweep dynamically adjusts hyperparameters—especially batch size and learning rate—during training. The learning rate is often scaled proportionally to batch size to maintain effective gradient updates.

* **Batch Size Impact:**
  Large batches speed up training but may reduce generalization; small batches add noise, sometimes improving robustness.
* **Learning Rate Role:**
  The learning rate controls update size. As batch size increases, a higher learning rate can be used (linear scaling rule: `learning rate ∝ batch size`).
* **Limitations:**
  Very large batches can cause the optimal learning rate to plateau or decrease (“surge phenomenon”), especially with adaptive optimizers like Adam.
* **Adaptive Techniques:**
  Modern research (e.g., Balles et al., 2017) explores dynamically adapting batch size based on gradient variance, simplifying hyperparameter tuning.

---

## **Summary of Key Learnings**

* **Batch size and learning rate are coupled.** Tune them together, not separately, to ensure stable optimization and faster convergence.
* **Linear scaling rule is a strong baseline,** but watch for diminishing returns or loss of generalization at very large batch sizes.
* **Dynamic/adaptive scheduling (Balles et al.)** can automate tuning, further improving efficiency.
* **Practical batch sizes (64–256) are typically optimal** for U-Net on current hardware. For fine-tuning, use smaller batches and lower learning rates to avoid overfitting.

---

## **Sweep Design**

### **Sweep Parameters**

* **batch\_size:** \[16, 32, 64, 128] (all feasible with 48GB VRAM)
* **base\_learning\_rate:** 0.0002 (reference value, batch size 16)
* **lr\_multiplier:** Uniform \[0.8, 1.2] (for robustness)
* **dropout:** Uniform \[0.15, 0.25]
* **l2\_reg:** Uniform \[0.00001, 0.00015]
* **epochs:** 40
* **input\_shape:** \[256, 256, 1]
* **task:** segmentation

#### **Example adaptive\_sweep.yaml**

```yaml
program: train.py
method: bayes
metric:
  name: val_dice_coefficient
  goal: maximize
parameters:
  batch_size:
    values: [16, 32, 64, 128]
  base_learning_rate:
    value: 0.0002
  lr_multiplier:
    distribution: uniform
    min: 0.8
    max: 1.2
  dropout:
    distribution: uniform
    min: 0.15
    max: 0.25
  l2_reg:
    distribution: uniform
    min: 0.00001
    max: 0.00015
  epochs:
    value: 40
  input_shape:
    value: [256, 256, 1]
  task:
    value: segmentation
```

### **Training Script Changes (train.py)**

Add adaptive LR calculation after `wandb.init()`:

```python
config = wandb.config
learning_rate = config.base_learning_rate * (config.batch_size / 16) * config.lr_multiplier
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
```

---

## **Experimental Considerations**

* **Monitor for diminishing returns:** If larger batch sizes (e.g., 128) degrade validation Dice, cap the range at 32 or 64.
* **Aggregate sweep results:** Identify the best-performing combination for both throughput and accuracy.
* **For future enhancement:** Try dynamic/adaptive batch size methods (Balles et al.) that increase batch size during training based on gradient variance.

---

## **Example Table: Adaptive Learning Rates per Batch Size**

| Batch Size | Learning Rate (w/ multiplier=1.0) |
| ---------- | --------------------------------- |
| 16         | 0.0002                            |
| 32         | 0.0004                            |
| 64         | 0.0008                            |
| 128        | 0.0016                            |

---

## **References**

* Balles, L., Romero, J., & Hennig, P. (2017). [Coupling Adaptive Batch Sizes with Learning Rates](https://arxiv.org/pdf/1612.05086.pdf)
* Goyal, P., et al. (2018). [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf)

---

## **Action Points & Communication for Stakeholders**

* **Experiment design is grounded in current research** (Balles et al., Goyal et al.) and prior results.
* **Batch size and learning rate must be coupled** for effective and efficient sweeps.
* **RTX 6000 Ada hardware enables larger batch sizes,** reducing wall time per epoch. Empirically confirm generalization does not degrade.
* **Sweep configuration ensures**: fast convergence, minimal manual intervention, reproducibility.
* **Monitor for surge effect**: Watch for generalization drops at the largest batch sizes.
* **Advanced directions:** Try dynamic batch size adaptation using Balles et al.’s methods for even more robust tuning.

---

## **Summary Statement**

> Your plan leverages state-of-the-art hardware, sound theoretical principles, and robust sweep design to accelerate U-Net segmentation research. Adaptive sweeps coupling batch size and learning rate reduce manual effort, increase throughput, and yield more reliable model performance. Next-level enhancements can use dynamic adaptation to further optimize efficiency and resource use.

---

## **Appendix: Balles et al. Dynamic Method vs. Linear Scaling**

| Method                         | Batch Size          | Learning Rate        | Notes                             |
| ------------------------------ | ------------------- | -------------------- | --------------------------------- |
| Classic SGD                    | Static              | Static or Decayed    | Manual tuning required            |
| Linear Scaling (current sweep) | Static (per run)    | lr ∝ batch\_size     | One sweep per pair                |
| Balles et al. (dynamic)        | Adaptive (per step) | Coupled (constant/↑) | Variance-based, simplifies tuning |

---
