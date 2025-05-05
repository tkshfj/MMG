## Project Proposal Presentation

Deep Learning for Breast Cancer Detection
Takeshi Fujii, MD
CM3070 – BSc in Computer Science, University of London

---

<!-- ### \[Slide 0: Title Slide] -->

Hello, I’m Takeshi Fujii. This video is to present my project proposal: Deep Learning for Breast Cancer Detection, based on the MLNN project idea 3.2.

---

<!-- ### \[Slide 1: Motivation and Background] -->

Breast cancer is a leading cause of cancer-related death worldwide, and the early detection of breast cancer is therefore important.

---

<!-- ### \[Slides 2–3: Medical Imaging and Dataset] -->

Mammography remains the gold standard for early cancer detection, but interpreting scans can be challenging even for experienced radiologists since early cancer signs can be subtle and easily missed. 

---

We will use a curated version of the Digital Database for Screening Mammography, containing thousands of annotated images with the Breast Imaging Reporting and Data System labels and biopsy results, which supports both classification and lesion localization tasks.

---

<!-- ### \[Slide 4: Workflow] -->

We will follow the universal deep learning workflow as shown on the slide.

---

<!-- ### \[Slide 5: Previous Work in Deep Learning for Breast Imaging] -->

Previous studies evaluate models across three tasks: segmentation, detection, and classification.
Convolutional neural networks like U-Net are widely used for segmentation. VGG and ResNet backbones are common in classification.
Transformer-based models are gaining popularity for their global context modeling. Hybrid models combine both approaches.
There is growing interest in multi-task learning to unify diagnosis, localization, and explainability.

---

<!-- ### \[Slides 6–8: Model Architecture – Staged Approach] -->

We will follow a staged modeling framework, progressing from baseline CNNs to advanced transformer-based multi-task models for classification and lesion localization on the mammogram dataset.

---


In Stage 1: We start with a baseline convolutional neural network for binary classification of benign versus malignant findings, using binary cross-entropy loss.

---

In Stage 2: We will enhance the model using transfer learning and regularization techniques. This stage also introduces U-Net for segmentation, trained using Dice and IoU losses.

---

In Stage 3: We will implement multi-task models that predict both diagnosis and lesion localization. This includes transformer-based models like Vision Transformer and hybrid models.
All models will be evaluated on accuracy and interpretability using Gradient-weighted Class Activation Mapping and attention maps.

---

<!-- ### \[Slide 9: Evaluation Metrics] -->

Two core metrics will be used: Area under the ROC curve for classification and Dice similarity coefficient for segmentation, which align with clinical needs—detecting cancer early while minimizing false positives and improving trust in the model.

---

<!-- ### \[Slide 10: Project Goal and Summary] -->

The goal is to build a model that not only classifies but also localizes lesions—making it more useful in real-world screening.
It uses a public mammography dataset with annotated labels, combines convolutional and transformer-based models and follows a structured, multi-task learning pipeline.
The outcome will be a prototype model that balances performance with clinical interpretability.
Thank you for your attention.

---
