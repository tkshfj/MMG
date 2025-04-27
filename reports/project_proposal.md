---
title: "Deep Learning for Breast Cancer Detection"
subtitle: "A Project Proposal based on the MLNN Project Idea, presented as part of the CM3070 Computer Science Final Project"
author: Takeshi Fujii, MD
date: 26 April 2025
format: pptx
---

## Introduction

### Why is early detection of breast cancer important?
- Breast cancer is the most frequently diagnosed cancer and the leading cause of cancer-related death among women worldwide. According to the GLOBOCAN database, in 2020 it accounted for approximately 2.3 million new cases (representing 1 in 4 new cancer diagnoses) and 685,000 deaths (1 in 6 cancer deaths) among women globally. This substantial global burden highlights the critical need for early detection, effective treatment, and continued research to reduce mortality rates.
- arly detection of breast cancer is essential because it significantly improves five-year survival rates—exceeding 90% when identified at an early stage—while enabling less aggressive interventions, such as breast-conserving surgery instead of mastectomy. It also reduces healthcare costs, preserves quality of life by minimizing treatment side effects and psychological impact, and strengthens the case for screening programs by demonstrating their effectiveness in identifying cancer at treatable stages.
- In alignment with these goals, on 4 February 2025, the UK government launched a world-leading clinical trial involving nearly 700,000 women to evaluate the role of artificial intelligence in early breast cancer detection. Announced on World Cancer Day, this initiative is part of the NHS’s 10-Year Health Plan to transition from analogue to digital infrastructure. The trial aims to support radiologists, reduce cancer mortality, and enhance healthcare efficiency. It also includes a national call for evidence to inform future cancer care strategy.
- https://ascopubs.org/doi/10.1200/JCO.2023.41.16_suppl.10528
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9465273/
- https://www.gov.uk/government/news/world-leading-ai-trial-to-tackle-breast-cancer-launched

---

## Mammography

---

## Dataset

Digital Database for Screening Mammography (DDSM)

---

## Models

- Segmentation: Where the tumor is exactly.
- Detection: Finding tumors (often less precise than segmentation).
- Classification: Diagnoses (deciding whether the tumor is benign or malignant).

![](../working/models_1.png)

![](../working/models_2.png)

---

## Previous Works

| **Main Focus** | **Segmentation** of breast lesions | **Detection** of breast tumors | **Classification** of breast tumors |
|:--|:--|:--|:--|
| Wang 2024 | Table 3 | Table 4 | Table 5 |
| **Task Type** | Pixel-wise labeling (where exactly the lesion is) | Bounding box or region proposal (where tumors are located) | Labeling as benign or malignant (or subtype) |
| **Model Examples** | U-Net variants (e.g., CRU-Net, RU-Net), Attention U-Nets, GANs, Mask R-CNN (for segmentation) | YOLO, Mask R-CNN, Faster R-CNN (for detection) | CNNs, Transfer Learning (VGG, ResNet), Ensemble classifiers (for diagnosis) |
| **Typical Input/Output** | Input: Mammogram → Output: Segmentation map (lesion mask) | Input: Mammogram → Output: Tumor location (bounding box/mask) | Input: Cropped lesion or whole image → Output: Class label (benign or malignant) |
| **Common Evaluation Metrics** | Dice coefficient (DSC), Jaccard index (intersection over union; IoU), pixel accuracy | Sensitivity, Specificity, Accuracy, AUC | Accuracy, Precision, Recall, F1-score, AUC |
| **Application in Workflow** | Helps localize and delineate tumors or abnormalities | Helps detect the presence and localization of tumors | Helps diagnose tumors (e.g., benign or malignant) |

---

## Next Steps

- Current Challenges
  - Data Scarcity
  - Annotation Quality
  - Imbalanced Datasets
  - Computational Resources
  - 
- Future Directions
  - Transfer Learning (TL)
    - Fine-tune pre-trained models (VGG16, ResNet50).
    - Replace classification head for binary classification.
  - Hyperparameter tuning (learning rate, batch size).
  - Ablation studies (e.g., full image vs ROI)

---
