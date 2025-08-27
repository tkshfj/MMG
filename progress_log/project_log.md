# Project Log

# 2025-08-27
- Refactor MONAI: stabilize cls head, warm up multitask loss, and unify W&B epoch logging to fix step warnings

# 2025-08-26
- Refactor MONAI pipeline: unify model/evaluator API; add two-pass evaluator with trainer-routed val metrics; fix CM/metric wiring, handler/JSON/tensor-truthiness; standardize keys and W&B step semantics; add single-logit BCE + imbalance tooling; enforce FP32 and spawn-safe loaders; step ReduceLROnPlateau after validation.

# 2025-08-21
- Refactor main: use fork on Linux (spawn elsewhere), build DataLoaders before CUDA init, pass mp context, and set pin_memory=False under fork.

# 2025-08-20
- Launch W&B sweeps on M4PRRO for refactored MONAI/Ignite Multitask U-Net (sweep_id: 6wgror3q, 2 runs/finished, August 19-20)
- Launch W&B sweeps on X870E for refactored MONAI/Ignite Multitask U-Net (sweep_id: bxc8qzvz, 1 runs/finished, August 20)
- Refactor MONAI pipeline: correct shape/layout issues, replace unpicklable lambdas for multiprocessing, standardize env (Zarr/CUDA/TF32), enforce deterministic runs, and add fast sanity checks.
- Refactor main.py: enforce Linux ‘fork’ start method; build DataLoaders before any CUDA; pass mp context to DataLoader; delay CUDA/TF32 setup until after loaders for fork-safety
- Refactor MONAI pipeline main.py: keep macOS on spawn (safe) and makes Linux use fork (fast)

# 2025-08-19
- Launch W&B sweeps on X870E for refactored MONAI/Ignite Multitask U-Net (sweep_id: 9eoogu2o/f7dt0ou5, 9 finished/31 crashed/40 runs, August 19)
- Fix: enforce FP32-only/TF32 off policy and fix transform dtypes

# 2025-08-18
- Launch W&B sweeps on X870E for refactored MONAI/Ignite Multitask U-Net (sweep_id: ftmiimtj/ksh8xv3i, 7 finished/39 crashed/46 runs, August 18)
- Launch W&B sweeps on X870E for refactored MONAI/Ignite Simple CNN (sweep_id: wusk5yxi, 6 finished/10 crashed/16 runs, August 18)
- Launch W&B sweeps on X870E for refactored MONAI/Ignite ViT (sweep_id: a5nhd6ll/zsvrfyyt, 5 finished/10 crashed/8 runs, August 17-18)

# 2025-08-17
- Refactor W&B sweep config to log val/multi, val/dice, val/iou, val/prec and val/recall as the vector means
- Launch W&B sweeps on X870E for refactored MONAI/Ignite Multitask U-Net (sweep_id: 3ewpciv6/1brphv69, 3 runs/finished, August 17)
- Refactor simpleCNN: class-weighted CE + multi-class prior bias; auto class_counts; sweep/eval diagnostics
- Refactor MONAI/Ignite classification and segmentation models to align with a common BaseModel API and standardized metric/output transforms
- Refactor ViT/DenseNet121/UNet/SwinUNETR to unified BaseModel API; standardize metric transforms and centralized losses; add class-imbalance/bias-init handling; align W&B sweeps and fix step semantics.

# 2025-08-16
- Rename dirs: mtlunet_monai->monai_v1; previous->monai_v2; recent->monai_v3
- Refactor schedulers, sweeps and metrics in MONAI/Ignite pipeline
- Launch W&B sweeps on X870E for refactored MONAI/Ignite Multitask U-Net (sweep_id: 6g24co4r, 6 runs/finished, August 17)

# 2025-08-15
- Archive legacy MONAI/Ignite code
- Close W&B sweeps for recent MONAI/Ignite Multitask U-Net (121 runs, August 09-14)
- Launch W&B sweeps for refactored MONAI/Ignite Multitask U-Net (sweep_id: 69481690, 14 runs, August 15-16)
- Refactor: Integrate a weighted harmonic mean val_multi into make_metrics

# 2025-08-10 to 2025-08-14
- Refactor MONAI/Ignite Multitask U-Net pipeline: Add task-aware metric construction, consistent output transforms, W&B step semantics, and hardened logging/checkpointing
- TODO: Add evaluator scheduling

# 2025-08-09
- refactor(model): use sweep-defined alpha/beta in multitask get_loss_fn
- refactor(pipeline): solidify transforms/metrics, clamp seg\_num\_classes≥2, robust cls OT (token/GAP), tolerant multitask loss, ABC+ROC\_AUC fixes

# 2025-08-08
- Fix: Standardize metrics, fix AUC/confmat, refactor W&B image logging (engine.state.output) and clean up config/handlers
- WIP: Add weighted multitask loss

# 2025-08-03
- Debug: AUC Output Transform

# 2025-08-02
- Refactor: Modular Model Registry Protocol Pipeline
- Debug: Classification Output Transform

# 2025-08-01
- Debug configuration-driven, modular MONAI/Ignite pipeline
- Migrate to configuration-driven, modular MONAI/Ignite pipeline
- Update all references to use "modular MONAI/Ignite pipeline" instead of "MCP"
  - Locate and replace all incorrect uses of "MCP" in reports and log entries
  - Rename ModelContextProtocol to ModelRegistryProtocol in codebase

# 2025-07-31
- Report: Model Context Protocol and Model Registry Refactoring
- Report: Model Context Protocol (MCP) Integration in MONAI/Ignite Project
- Issue: val_cls_confmat is not logged into wandb
- Refactor model and metric utilities for metric logging and reduced redundancy

# 2025-07-30
- Refactor handlers to ensure sweep runs log metrics to Wandb
- Revert to commit deb3574 from 28 July
- Lauch refactored MONAI/Ignite multitask pipeline for multitask U-Net and ViT (20 runs each, July 30-)

# 2025-07-28
- Fix: FileNotFoundError When Deleting Old Checkpoints
- Refactor handlers to include the architecture name in checkpoint and best model filenames
- Refactor handers to prevent out-of-order warnings

# 2025-07-27
- Refactor: Generalize Final Model Saving with Architecture Names
- Delete fit logs and model checkpoints from previous runs

# 2025-07-26
- Refactor and debug MONAI/Ignite multitask pipeline to use nested label dictionary
- Reorganize directories
- Add Baseline CNN and ViT Models to MONAI/Ignite Pipeline
- Report: Model Extension Feasibility using MONAI/Ignite framework
- Lauch refactored MONAI/Ignite multitask pipeline for multitask U-Net and ViT (20 runs each, July 26-)
- Fix: Checkpoint Handling in Ignite/MONAI

# 2025-07-25
- Debug and Refactor MONAI/Ignite Multitask Pipeline
- Resolve bincount_cuda not implemented for Float error in segmentation output transforms for ConfusionMatrix

# 2025-07-24
- Refactor to Set Early Stopping Conditions in MONAI/Ignite Pipeline
- Report: Confusion Matrix Example and Interpretation
- Analyze Sweep Results
  - Analyze the top 5 runs from the latest sweep
  - Refine hyperparameter ranges based on top-performing runs
- Adjust the sweep hyperparameters
- Relauch refactored MONAI/Ignite multitask pipeline (* runs from amber-sweep-1, July 24-)

# 2025-07-23
- Create minimal requirements.txt
- Report: Minimal requirements.txt & Environment Dependency Management
- Report: Debugging and Refactoring MONAI/Ignite multitask pipeline
  - Shape Mismatch in Ignite Metrics: ValueError: y_pred does not have correct number of classes: 256 vs 2
- Code review: MONAI/Ignite multitask pipeline
- Refactor: add logging for data loading errors, use dynamic metric keys in handlers
- Save multitask modular pipeline as of 2025-07-21
- Lauch refactored MONAI/Ignite multitask pipeline (7 runs, July 24)

# 2025-07-22
- Refactor U-Net multitask modular pipeline with MONAI/Ignite integration
- Report: Refactoring Unified MONAI/Ignite Pipeline with Task Argument
- Report: Segmentation Metrics in MONAI with Ignite

# 2025-07-21
- Refactor U-Net multitask modular pipeline with MONAI/Ignite integration
- Launch sweep with MONAI/Ignite integration: U-Net multitask modular pipeline (34 runs, July 21-24)
- Report: MONAI/Ignite Modular Deep Learning Pipeline (MMG Project)
- Report: Understanding the `y_pred does not have correct number of classes` Error in Ignite Segmentation Metrics

# 2025-07-20
- Refactor U-Net multitask modular pipeline with MONAI integration
- Launch sweep with MONAI integration: U-Net multitask modular pipeline (11 runs, July 20-21)
- Report: Logging and Metrics Management in MONAI/Wandb Deep Learning Pipeline
- Report: MONAI Multitask Deep Learning Pipeline — Technical Report (20 July 2025)

# 2025-07-19
- Refactor U-Net multitask modular pipeline with MONAI integration
- Report: U-Net multitask Pipeline with MONAI with Ignite integration
- Report: Updated list of core error messages

# 2025-07-18
- Create U-Net multitask modular pipeline with MONAI integration
- Refactor U-Net multitask modular pipeline with MONAI integation
- Report: Modular U-Net multitask Pipeline with MONAI & W&B

# 2025-07-17
- Refactor build dataloaders function for multiprocessing support
- Refactor U-Net multitask pipeline with enhanced MONAI integration
- Refactor classic U-Net segmentation pipeline with enhanced MONAI integration
- Report: Multitask U-Net Training, Debugging and Sweep Analysis

# 2025-07-16
- Lauch refactored MONAI multitask pipeline (57 runs, July 16-20)

# 2025-07-15
- Refactor MONAI multitask pipeline: corrects metric computation, adds Dice/IoU, streamlines validation
- Report: Multitask Pipeline with MONAI
- Report: Debugging and Refactoring Multitask U-Net Validation with MONAI Metrics

# 2025-07-14
- Launch sweep: MONAI multitask pipeline (12 runs, July 14-15)
- Report: MONAI-based Multitask Pipeline

# 2025-07-13
- Add and submit testing report for peer review
- Watch Weeks 15-16 lectures
- Refactor MONAI multitask pipeline: corrects metric computation, adds Dice/IoU, streamlines validation
- Report: Multi-Task Classification & Segmentation Pipeline

# 2025-07-10
- Refactor baseline code: add MONAI DataLoader and wandb sweep support for flexible hyperparameter tuning and standardized metric logging.
- Add dropout, threshold tuning, and class-weighted loss to MONAI DenseNet; update sweep.yaml for improved recall and AUC.
- Add Focal Loss, stronger data augmentation, weight decay sweep, and class-balanced sampling for improved model recall and precision.
- Lauch adaptive sweep with monai integration: baseline CNN model for classification task (60 runs, July 10)

# 2025-07-09
- Add robust per-epoch validation logging for Dice and IoU metrics (both MONAI and manual)
- Lauch adaptive sweep with monai integration: classic U-Net seg monai  (21 runs, July 9-13)
- Report: MONAI Segmentation Validation and PyTorch Checkpointing
- Report: U-Net Segmentation Sweep Runs and MONAI Migration

# 2025-07-08
- Refactor validation to aggregate DiceMetric over full validation set, log both per-epoch val_dice_coefficient and manual Dice, add output/mask shape and stats debug, and improve exception handling for robust MONAI segmentation analysis.
- Refactor validation to compute val_dice_coefficient on full validation set per epoch for accurate MONAI metric logging.

# 2025-07-07
- Refactor data_utils_monai.py to fix monai dataloader collate error
- Launch adaptive sweep with monai integration: classic U-Net seg monai
- Report: MONAI DataLoader Collate Error and Solutions
- Report: Refactor Report: Robust Medical Image Data Pipeline with MONAI

# 2025-07-06
- Refactor data_utils_monai.py to integrate MONAI core components for transforms and data loading.
- Known issue: MONAI DataLoader collate error persists due to inconsistent sample shapes entering batching. Further investigation is needed—particularly into force_2d_slice logic and upstream mask/image preprocessing—to ensure uniform output shape before batching.

# 2025-07-03
- Refactor to integrate the core components of MONAI
  - including native DICOM support, MetaTensor integration, ready-to-use medical imaging networks such as the built-in U-Net and related architectures, as well as MONAI’s specialized losses and metrics. 

# 2025-07-02
- Refactor and launch adaptive sweep: classic U-Net segmentation (261 runs, July 2-6)

# 2025-07-01
- Analyze results from the classic U-Net segmentation sweep

# 2025-06-21
- Refactor U-Net segmentation code
- Launch a large-scale sweep (480 runs, June 21–26) for systematic hyperparameter optimization

# 2025-06-20
- Fix classic U-Net segmentation model and test training on a smaller sample size

# 2025-06-18
- Refactored the train.py and sweep.yaml code and stated running sweeps

# 2025-06-17
- Implemented a classic U-Net segmentation model for the CBIS-DDSM dataset.

# 2025-06-16
- Compiled and submitted the preliminary project report.
- Screen recorded the demonstration for the submission.

# 2025-06-15
- Run a sweep to find optimal hyperparameters for the baseline CNN model.
```sh
$ wandb sweep --project baseline_cnn_dropout_aug_rev sweep.yaml
$ wandb agent tkshfj-bsc-computer-science-university-of-london/baseline_cnn_dropout_aug_rev/p6smuds6
```

# 2025/06/11-13
- Run a Weights & Biases sweep to optimize hyperparameters for the baseline CNN model with dropout and augmentation
- Run a sweep to find optimal hyperparameters for the baseline CNN model.

# 2025/06/03-05
- Revised the code to handle augumentation in the pipeline.
- Run a Weights & Biases sweep to optimize hyperparameters for the baseline CNN model with dropout and augmentation.

# 2025/06/02
- Run a sweep to find optimal hyperparameters for the baseline CNN model.

# 2025/06/01
- Revised the baseline CNN classifier for the CBIS-DDSM dataset.
  - Set up Weights & Biases (wandb) on Ubuntu for experiment tracking.
  - Modularize dataset loading logic into data_utils.py; update train.py and sweep.yaml for command-line training and sweep workflow.

# 2025-05-31
- Revised the baseline CNN classifier for the CBIS-DDSM dataset.
  - Accuracy rises from approximately 0.54 to just above 0.60 (both training and validation), which is only slightly better than random chance (0.5 for binary classification), indicating that the model functions as a weak classifier.
  - AUC improves from around 0.55 to just above 0.62 (training and validation). The model demonstrates limited ability to distinguish between classes. For medical imaging tasks, an AUC above 0.80 is generally considered necessary for clinical relevance.

## 2025-05-30
- Revised the data pipeline code to properly handle pathology results in the clinical metadata.
- Started developing the baseline CNN classifier for the CBIS-DDSM dataset.
  - Accuracy ~0.60 (train/val) and AUC: ~0.61 (val), still far below the clinical threshold 77.4% accuracy, 78.8% precision, and 77.8% recall (AUC ≥ 0.80).

## 2025-05-25, 2025-05-26
- Drafted the project design, work plan nd evaluation plan.

## 2025-05-24
- Completed watching the lectures for Weeks 7 and 8.

## 2025-05-23
- Removed quarto/pandoc templates from the repository.
- Set up templates in the Obsidian vault on Dropbox.

## 2025-05-22
- Submitted the literature review.

## 2025-05-19, 2025-05-21, 
- Revised the literature review draft.

## 2025-05-17
- Moved section drafts to the Obsidian vault on DropBox.

## 2025-05-13
- Revised the literature review draft.

## 2025-05-12
- Installed Obsidian on macOS and created a vault.
- Moved section drafts to the Obsidian MyVault on iCloud Drive.

## 2025-05-11
- Installed Zotero on Ubuntu

```bash
wget "https://www.zotero.org/download/client/dl?channel=release&platform=linux-x86_64" -O zotero.tar.bz2
tar -xjf zotero.tar.bz2
sudo mv Zotero_linux-x86_64 /opt/zotero
sudo ln -s /opt/zotero/zotero /usr/local/bin/zotero
nano ~/.local/share/applications/zotero.desktop
```

```ini
[Desktop Entry]
Name=Zotero
Exec=/opt/zotero/zotero
Icon=/opt/zotero/chrome/icons/default/default256.png
Type=Application
Categories=Education;Office;
```

```bash
zotero &
```

- Installed Better BibTeX on Ubuntu
1. Open Zotero > Tools > Plugins > Gear > Install Add-on From File
2. Select the downloaded `.xpi`, then restart Zotero.
   [https://github.com/retorquere/zotero-better-bibtex/releases](https://github.com/retorquere/zotero-better-bibtex/releases)
  - https://retorque.re/zotero-better-bibtex/

- Installed [Obsidian](https://obsidian.md/) on Ubuntu and MacOS
  - https://snapcraft.io/obsidian

```sh
sudo snap install obsidian --classic
```

- Revised the literature review template to be compatible with Obsidian

## 2025-05-10

- Revised the data pipeline code to support multi-task learning by outputting tuples in the format: (image, {"segmentation": mask, "classification": label})

## 2025-05-07

- Continued reviewing relevant literature.
- Consulted the tutor regarding recommended referencing style.
- Downloaded INbreast dataset
  - Kaggle: https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset
  - DOI: 10.1016/j.acra.2011.09.014
  - The INbreast dataset contains 115 cases and 410 full-field digital mammograms.
  - 90 cases have 4 images each (both breasts), and 25 cases have 2 images each (post-mastectomy).
  - Images are in DICOM format with resolutions of 3328×4084 or 2560×3328 pixels.
  - Each image includes annotations for masses, calcification, asymmetries, and architectural distortions, along with BI-RADS breast density labels and CC/MLO views.

## 2025-05-06

- Completed watching the lectures for Weeks 5 and 6.
- Created templates and Quarto settings for the literature review.
- Downloaded ACM reference style from Zotero.
  - Referencing Styles https://onlinelibrary.london.ac.uk/support/referencing/referencing-styles
  - Zotero Style Repository https://www.zotero.org/styles
  - Association for Computing Machinery (ACM)
  - Cite Them Right 12th edition - Harvard (no "et al.")

## 2025-05-05

- Submitted the project proposal presentation.

## 2025-05-04

- Revised the project proposal slides.
- Reorganized the directory structure for the project proposal.

## 2025-05-03

- Revised the project proposal slides.
  - Created a table for the Previous Works section.

## 2025-05-01

- Revised the project proposal slides.

## 2025-04-30

- Revised the project proposal slides, focusing on content structure.

## 2025-04-29

- Searched for multi-task learning (MTL) in medical imaging.
- [Cross-Task Attention Network: Improving Multi-Task Learning for Medical Imaging Applications](https://dl.acm.org/doi/10.1007/978-3-031-47401-9_12)
- Searched for vision transformers (ViT) and their applications in medical imaging.
  - [Segmentation for mammography classification utilizing deep convolutional neural network](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-024-01510-2)
  - Transformer-based deep learning models for breast cancer diagnosis

## 2025-04-28

- Output the directory structure of /data/ into a tree.txt file.
  - An ordinary "tree . > tree.txt" doesn't work on Ubuntu.
  - Solution: 
```sh
script -q -c "tree -n data/" tree.txt
```

## 2025-04-27

- Revised project proposal slides
  - Add descriptions on Models and Previous Works

- Installed TensorFlow 2.15.0 with GPU support (CUDA 12.2, cuDNN 8.9).
- Installed PyTorch 2.2.0 with CUDA 12.1 support.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow==2.15.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- Completed data pipeline code for the CBIS-DDSM dataset.
  - Scan DICOM Files
  - Extract Metadata from File Paths
  - Pair Images and Masks
  - Merge Image–Mask Metadata and Clinical Metadata
  - Build TensorFlow Dataset

## 2025-04-26

- Installed Quarto on Ubuntu 22.04
```sh
sudo apt update
sudo apt install -y wget gdebi-core
cd ~/Downloads
wget https://quarto.org/download/latest/quarto-linux-amd64.deb
sudo gdebi quarto-linux-amd64.deb
quarto --version
sudo apt install -y texlive-xetex texlive-fonts-recommended texlive-latex-extra
xelatex --version
sudo apt install -y ttf-mscorefonts-installer
```

- Started data pipeline development for the CBIS-DDSM dataset.
  - Described an overview of the data pipeline architecture and its components.
  - Implemented functions to read and parse the clinical metadata CSV files.

- Started writing project proposal slides using Markdown.
```sh
pandoc project_proposal.md -t pptx -o _output/project_proposal.pptx
```

## 2025-04-25

- Conducted a preliminary review of the U-Net architecture for image segmentation.
  - The paper titled ["U-Net: Convolutional Networks for Biomedical Image Segmentation”](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox introduces the U-Net architecture, a convolutional neural network designed for precise biomedical image segmentation.

## 2025/04/24

- Installed an Intel AX210 Wi-Fi 6E adapter, replacing the onboard MediaTek MT7927 module.
- Installed NVIDIA RTX 4080 Super as a second GPU.  
- Installed NVIDIA proprietary driver version 550.144.03 and CUDA toolkit 12.4 for GPU acceleration.  
- Set up Python virtual environments using `venv` for isolated package management.  
- Installed TensorFlow 2.15.0 with GPU support (CUDA 12.2, cuDNN 8.9).  
- Installed PyTorch 2.2.0 with CUDA 12.1 support.  
- Installed and configured Jupyter Notebook and Visual Studio Code for development.  
- Configured SSH keys for GitHub version control.

## 2025/04/22-23

- Studied image segmentation techniques using Convolutional Neural Networks (CNNs).
  - Chapters 8–9 of *Deep Learning with Python, 2nd Edition* by François Chollet.

## 2025/04/20

- Set up writing tools for documentation
  - Quarto Framework: Combines Markdown, LaTeX, and code for scientific publishing.  
  - Output: Generates PDF (via XeLaTeX) and HTML with consistent styling and title-cased TOC.  
  - Structure: Modular layout using `_quarto.yml` and `report.qmd`.  
  - Customization: Fonts and citation styles configured via `.bib` and `.csl` files.  
  - Version Control: Managed with Git and GitHub for reproducibility.  
  - Tools: Authored in VSCode, using Zotero with Better BibTeX for reference management.  
  - Workflow: Supports live preview, code execution, and source-controlled outputs.

## 2025/04/19

- Assembled and configured a workstation for AI workloads.
  - Installed Ubuntu 22.04 LTS on a workstation with the following specifications:
  - Motherboard: ASUS ProArt X870E Creator Wifi
  - PSU: Corsair RM1000e
  - CPU: AMD Ryzen 9 7950X
  - GPU: NVIDIA RTX 6000 Ada
  - RAM: 96 GB DDR5
  - Storage: 4 TB NVMe SSD
  - CPU Cooler: Noctua NH-D15 G2
  - PC Case: be quiet! Dark Base PRO 901 Black (BGW50 / DRK-BSE-PRO-901/BK)

## 2025/04/17

- Set up GitHub repository: [tkshfj/MMG](https://github.com/tkshfj/MMG)  
- Added README with project details
- Drafted project plan, timeline and project milestones for the project

## 2025/04/16

- Briefly explored the CBIS-DDSM dataset and its structure.

## 2025/04/15-16

- Downloaded the CBIS-DDSM dataset (163.54 GB) from [TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).  
  - Source: Lee et al., *Scientific Data* (2017), DOI: [10.1038/sdata.2017.177](https://doi.org/10.1038/sdata.2017.177)
  - Used NBIA Search Portal > filtered by collection: CBIS-DDSM
  - Added selected cases to cart > downloaded `.tcia` manifest
  - Retrieved data using NBIA Data Retriever (overnight download)

- Downloaded metadata CSVs:  
  - `calc_case_description_test_set.csv`  
  - `calc_case_description_train_set.csv`  
  - `mass_case_description_test_set.csv`  
  - `mass_case_description_train_set.csv`

- Each metadata file includes:  
  - Patient ID, View (CC/MLO), Pathology, BI-RADS, Lesion Subtlety

## 2025/04/10-15

Discussion on Project Ideas:
- Consulted with the tutor, Mr. Haris Bin Zia, on the Discussion Forum regarding project ideas and feasibility.
- Expressed interest in two project ideas:
  - MLNN: Deep Learning Breast Cancer Detection
  - AI Orchestrating Multiple Models for Whole Slide Image (WSI) Analysis
- Noted the need to assess workstation performance for handling resource-intensive AI tasks before finalizing the project direction.

Instructor Feedback:
- Advised verifying the availability of labeled datasets.
- Emphasized the need to incorporate models from multiple data modalities (e.g., text, image, audio) if pursuing the "Orchestrating AI" template.
- Clarified that the "Orchestrating AI" template is designed for agentic workflows and requires pre-trained models from different data spaces.
- Recommended that the WSI project would be better suited under the MLNN template, which does not require multimodal input.

Project Direction:
- Confirmed the decision to proceed with the MLNN: Deep Learning Breast Cancer Detection project.

Further Inquiries:
- Raised concern over the high benchmark performance reported in previous works (AUC > 0.95), often achieved using:
  - Transfer learning  
  - ROI localization  
  - Multi-view fusion

Clarification from Instructor:
- Reiterated that the purpose of the project is to guide students through the machine learning project lifecycle:
  - Begin with a baseline model  
  - Apply best practices in model development, training, evaluation, and error analysis  
- Achieving state-of-the-art performance is encouraged but not required.

## 2025/04/05

- Browsed the project ideas/templates released on the Coursera platform.