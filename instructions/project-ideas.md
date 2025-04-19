### Project Summary
This project investigates whether convolutional neural networks (CNNs) can improve the accuracy of breast cancer detection using X-ray mammography, focusing on the Digital Database for Screening Mammography (DDSM). The aim is to develop a deep learning model that matches or exceeds radiologist-level performance, potentially reducing workload and diagnostic delays in clinical settings.

### Context and Motivation
Breast cancer is a leading cause of death among women globally, and early detection significantly improves treatment outcomes. In 2025, the UK NHS launched a national AI research project to explore whether deep learning can replace one of the two radiologists typically involved in mammogram screening. CNNs, known for their image recognition capabilities, are particularly well-suited for analyzing mammograms in this context.

### Recommended Resources
- Wang L. (2024). *Mammography with deep learning for breast cancer detection*. Front Oncol. [https://doi.org/10.3389/fonc.2024.1281922](https://doi.org/10.3389/fonc.2024.1281922)  
- Lee et al. (2017). *A curated mammography data set for use in computer-aided detection and diagnosis research*. Sci Data. [https://doi.org/10.1038/sdata.2017.177](https://doi.org/10.1038/sdata.2017.177)  
- Francois Chollet (2018). *Deep Learning with Python*. Manning  
- [TensorFlow documentation](https://www.tensorflow.org)

### Deliverables/Expected Outcomes
- A trained CNN model that predicts the presence of breast cancer with measurable accuracy
- Quantitative evaluation using metrics such as accuracy, sensitivity, specificity, and AUC
- A report detailing model architecture, training process, and performance benchmarking against radiologist-level accuracy

### Prototype Expectations
- A functional CNN trained on a subset of the DDSM dataset with measurable performance
- Incorporation of data preprocessing, training, testing, and validation techniques

### Core Techniques and Concepts
- Deep Learning Frameworks: TensorFlow, Keras
- Model Type: Convolutional Neural Networks (CNNs)
- Dataset: DDSM
- Key Techniques: Data augmentation, model regularization, transfer learning (e.g., VGG), metric analysis and optimization

### Evaluation Criteria
- Pass (3rd class): A basic CNN trained and evaluated with minimal workflow adherence
- Good (2:2 – 2:1): Multiple regularized CNN models, well-documented methodology, strong alignment with best practices
- Outstanding (1st class): Use of transfer learning, reference to advanced techniques (e.g., Wang 2024), and results that approach publishable quality

---

## 3.2 MLNN: Deep Learning Breast Cancer Detection

> What problem is this project solving, or what is the project idea?

The aim is to establish if Deep Learning assisted X-ray mammography can improve the accuracy of breast cancer screening. The project will achieve this aim by modelling the Digital Database for Screening Mammography (DDSM) with convolutional neural networks (CNNs).

> What is the background and context to the question or project idea above?

The UK national health service (NHS) launched, in early 2025, a major research project to address the above research question. If successful, a DL system could replace one of the two radiologists currently reporting on scans with the consequence of faster diagnostic turn-around and the liberation of specialists for other tasks. CNNs are deep learning neural networks that apply a succession of filters to the input layer. They are capable of impressive image recognition tasks.

> Here are some recommended sources for you to begin your research.
- Wang L. Mammography with deep learning for breast cancer detection. *Front Oncol.* 2024 Feb 12;14:1281922. doi: 10.3389  
  [https://doi.org/10.3389/fonc.2024.1281922](https://doi.org/10.3389/fonc.2024.1281922)  
  PMID: 38410114; PMCID: PMC10894909.
- Lee, R., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M. & Rubin, D. L. "A curated mammography data set for use in computer-aided detection and diagnosis research." *Sci Data* 4, 170177 (2017).  
  [https://doi.org/10.1038/sdata.2017.177](https://doi.org/10.1038/sdata.2017.177)
- Francois Chollet (2018). *Deep Learning with Python.* Manning, Shelter Island
- [https://www.tensorflow.org](https://www.tensorflow.org)

> What would the final product or final outcome look like?

A CNN statistical model, predictions from this model and comparisons to specialist reporting accuracy.

> What would a prototype look like?

A small CNN capable of achieving statistical power.

> What kinds of techniques/processes/CS fundamentals are relevant to this project?

Artificial neural networks; dataset splitting, model building with tensorflow, training and testing.

> What would the output of these techniques/processes/CS fundamentals look like?

One or more test metrics for the best statistical model trained on the DDSM dataset.

> How will this project be evaluated and assessed by the student (i.e. during iteration of the project)? What criteria are important?

The student will seek to improve the chosen test metrics by network scaling up and regularisation.

> For this brief, what might a minimum pass (e.g. 3rd) student project look like?

A working CNN that at the minimum has statistical power. Some attempt to follow the deep learning workflow, as described in the textbook by Chollet.

> For this brief, what might a good (e.g. 2:2 – 2:1) student project look like?

A succession of regularised CNN networks; strict adherence to the Deep Learning workflow.

> For this brief, what might an outstanding (e.g. 1st) student project look like?

Application of transfer learning (e.g. VGG models), and any of the alternatives listed in Table 3 of Wang 2024.  
For a high first: near publishable results for an original model that is competitive with the best reported DL models in the literature.
