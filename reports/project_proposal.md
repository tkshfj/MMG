---
title: "Deep Learning for Breast Cancer Detection"
subtitle: "A Project Proposal based on the MLNN Project Idea, presented as part of the CM3070 Computer Science Final Project"
author: Takeshi Fujii, MD
date: 5 May 2025
format: pptx
---

## Introduction

### Why Is Early Detection of Breast Cancer Important?

- High Global Burden:
Breast cancer is the most commonly diagnosed cancer and the leading cause of cancer-related death among women worldwide. In 2020, it accounted for **2.3 million new cases** (1 in 4 cancer diagnoses) and **685,000 deaths** (1 in 6 cancer deaths) globally (GLOBOCAN 2020).

- Improved Survival:
Early detection increases five-year survival rates to over 90%, enabling less invasive treatments (e.g., breast-conserving surgery instead of mastectomy) and better quality of life.

- Lower Costs & Better Outcomes:
It reduces healthcare costs, minimizes treatment-related side effects, and supports psychological well-being.

- Policy Momentum:
- On 4 February 2025, the UK government launched a landmark AI-driven breast cancer screening trial involving nearly 700,000 women. This initiative, part of the NHS 10-Year Plan, aims to:
- Enhance early detection with AI support
- Alleviate radiologist workload
- Reduce mortality and improve system efficiency

> References:
> - [ASCO JCO Abstract 10528 (2023)](https://ascopubs.org/doi/10.1200/JCO.2023.41.16_suppl.10528)
> - [PMC Article (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9465273/)
> - [UK Government AI Trial Announcement](https://www.gov.uk/government/news/world-leading-ai-trial-to-tackle-breast-cancer-launched)

---

## Medical Imaging in Breast Cancer Detection

### Breast Cancer Screening
- **Mammography** is the clinical gold standard for early detection.
- However, it has notable **limitations**:
- Reduced sensitivity in **dense breast tissue**
- High rate of **false positives**, leading to unnecessary biopsies
- **Limited specificity** in distinguishing benign from malignant lesions

### Complementary Imaging Modalities
- **Digital Breast Tomosynthesis (DBT)**:
  - 3D imaging that improves lesion visibility and reduces recall rates
- **Ultrasound / MRI**:
  - Effective for dense breasts and high-risk populations
- **PET (Positron Emission Tomography)**:
  - Offers functional imaging with high sensitivity
  - Limited by high cost and complexity

---

## How to Interpret Mammograms (Part 1)

### 1. Breast Density Evaluation
- **BI-RADS A to D**:
- A: Almost entirely fatty
- B: Scattered fibroglandular densities
- C: Heterogeneously dense
- D: Extremely dense
- **Dense breasts (C/D)** may obscure tumors and lower sensitivity

### 2. View Orientation
- **Standard views**:
- **CC (Cranio-Caudal)**
- **MLO (Mediolateral Oblique)**
- Evaluate:
- Symmetry
- Positioning
- Complete tissue coverage

### 3. Lesion Assessment
- **Masses**: Shape (round, oval, irregular); margin (circumscribed, spiculated)
- **Calcifications**: Size, morphology (coarse vs. micro), distribution (clustered, linear)
- **Architectural distortion**: Disruption of normal breast structure
- **Asymmetry**: One-sided density differences, especially if new or increasing

---

## How to Interpret Mammograms (Part 2)

### 4. BI-RADS Classification
- **Category 0**: Incomplete – needs further imaging
- **Category 1–2**: Negative or benign findings
- **Category 3**: Probably benign – short-term follow-up
- **Category 4–5**: Suspicious or highly suggestive of malignancy → **biopsy recommended**
- **Category 6**: Known biopsy-proven malignancy

### 5. Biopsy (If BI-RADS 4 or 5)
- **Techniques**:
- **Stereotactic biopsy**: Targets calcified or subtle lesions
- **Ultrasound-guided biopsy**: Real-time guidance for solid masses
- **MRI-guided biopsy**: Reserved for MRI-only visible abnormalities

- **Histopathology confirms**:
- *Benign*: Fibroadenoma, cysts
- *Premalignant*: Atypical hyperplasia
- *Malignant*: DCIS, invasive ductal carcinoma (IDC)

- **Determines treatment strategy** (e.g., surgery, oncology referral)

---

## Dataset

### Digital Database for Screening Mammography (DDSM)
- A legacy, publicly available dataset of normal and abnormal mammograms.
- Developed to support early research in computer-aided detection (CAD) for breast cancer.

### Curated Breast Imaging Subset of DDSM (CBIS-DDSM)
- A modern, curated version of DDSM, hosted on The Cancer Imaging Archive (TCIA).
- Contains 10,000+ images with expert annotations of lesions (masses, calcifications, architectural distortions).
- Divided into training, validation, and test sets to ensure robust model evaluation.
- Each case includes:
- Lesion annotations: location, type and size
- Metadata: patient age, breast side, view type, etc.
- Supports segmentation, detection, and classification — all within a unified dataset — making it ideal for multi-task learning and cross-modal analysis.
- Widely used in breast cancer AI research.

---

## Models

This project will follow the deep learning workflow outlined by **Chollet (2021)**, encompassing:
- Data preprocessing  
- Model construction  
- Training and evaluation  
- Iterative optimization and refinement  

The implementation will use **TensorFlow/Keras**, following best practices for medical image analysis.

---

### Core Tasks in Breast Imaging
The model will address three key tasks commonly encountered in mammography-based diagnosis:

- **Segmentation**: Delineate tumor boundaries and lesion shapes  
- **Detection**: Identify and localize suspicious regions (bounding boxes or masks)  
- **Classification**: Determine tumor type (e.g., benign vs. malignant)  
<!-- - **Risk Prediction**: Estimate likelihood of malignancy based on imaging features -->

---

### Representative Deep Learning Approaches

- **Convolutional Neural Networks (CNNs)**: Fundamental for feature extraction and visual pattern recognition  
- **Transfer Learning (TL)**: Utilizes pretrained models (e.g., VGG16, ResNet50) to reduce training time and improve generalization  
- **Ensemble Learning (EL)**: Combines outputs from multiple models to enhance robustness  
- **Attention Mechanisms**: Improves focus on diagnostically relevant image regions

---

### Task-Specific Architectures

#### **Segmentation**
- **U-Net**: Classic encoder–decoder design with skip connections  
  → Captures both spatial detail and global context  
- **Attention U-Net**: Integrates attention gates to suppress irrelevant regions  
- **GANs / cGANs**: Employ adversarial learning to generate and refine lesion masks  
- **Tubule-U-Net**: Optimized for detecting glandular structures in dense tissue

#### **Detection**
- **YOLO (You Only Look Once)**: Fast, single-pass detector ideal for real-time scenarios  
- **Faster R-CNN**: Two-stage detector with region proposal and classification components  
- **Mask R-CNN**: Extends Faster R-CNN by adding segmentation masks for each detected object

#### **Classification**
- **CNNs**: Predict diagnostic labels from raw or pre-segmented image regions  
- **Transfer Learning**: Fine-tune pretrained models (VGG, ResNet, DenseNet) for binary/multiclass diagnosis  
- **Ensemble Methods**: Improve classification accuracy by combining multiple model predictions

---

### Evaluation Metrics

To assess performance across tasks, the following metrics will be employed:

- **Sensitivity**: Ability to detect true positives (recall)  
- **Specificity**: Ability to identify true negatives  
- **Accuracy**: Proportion of correct predictions across all cases  
- **F1 Score**: Harmonic mean of precision and recall, balancing false positives and negatives  
- **AUC (Area Under the ROC Curve)**: Measures overall discriminative ability of the model

![](../working/models_1.png)

![](../working/models_2.png)

---

### Previous Works: Segmentation & Detection

| **Aspect**| **Segmentation** | **Detection** |
|-----------------------|--------------------------------------------|---------------------------------------------|
| **Main Focus**| Segment breast lesions | Detect tumor locations|
| **Reference** | Wang 2024, Table 3 | Wang 2024, Table 4|
| **Task Type** | Pixel-wise labeling| Bounding box or region proposal |
| **Model Examples**| U-Net, CRU-Net, RU-Net, …, Mask R-CNN| YOLO, Mask R-CNN, Faster R-CNN|
| **Input → Output**| Mammogram → Segmentation map | Mammogram → Tumor location (box/mask) |
| **Metrics** | Dice (DSC), IoU, Pixel Accuracy| Sensitivity, Specificity, Accuracy, AUC |
| **Role in Workflow**| Localization and boundary delineation| Tumor localization for further analysis |

---

### Previous Works: Classification

| **Aspect**| **Classification** |
|-----------------------|--------------------------------------------|
| **Main Focus**| Classify tumors (e.g., benign vs malignant)|
| **Reference** | Wang 2024, Table 5 |
| **Task Type** | Image- or ROI-based labeling |
| **Model Examples**| CNNs, VGG, ResNet, …, Ensemble classifiers |
| **Input → Output**| Mammogram / ROI → Class label|
| **Metrics** | Accuracy, Precision, Recall, F1 Score, AUC |
| **Role in Workflow**| Diagnostic classification and risk scoring|


---

## Next Steps

### Current Challenges
- Data Scarcity – Limited annotated medical data
- Annotation Quality – Variability in expert labeling
- Imbalanced Datasets – Fewer malignant cases
- Computational Resources – High demands for training deep models

### Transfer Learning
- Load pretrained base (e.g., VGG16, ResNet50)
- Replace classification head for binary diagnosis tasks
- Fine-tune on CBIS-DDSM

### Multi-Task Learning (MTL)
- Jointly segment and classify tumors
- Shared encoder improves generalization and efficiency 
- Customize loss functions per task

### Vision Transformers
- These models learn global representations more efficiently than CNNs, which is especially valuable in capturing long-range dependencies in mammogram images.
- Emerging architecture for all three tasks
- Strong performance on medical image classification and segmentation
- Particularly promising for tasks with limited data due to their high sample efficiency and strong representation learning

---

### Project Goal

- The initial prototype will aim to achieve **measurable statistical power**, establishing a robust baseline for model evaluation and iterative refinement.  
- The overall goal is to **design, implement, and evaluate** a **multi-task deep learning pipeline** using the **CBIS-DDSM dataset** to perform **simultaneous segmentation, detection, and classification** of breast tumors in mammographic images.

---
