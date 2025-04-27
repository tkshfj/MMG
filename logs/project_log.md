# Project Log

## Todo

- Review image segmentation, detection and classification techniques
  - Chapters 1-9 of *Deep Learning with Python, 2nd Edition* by François Chollet.

## 2025-04-28

- Output /data/ directory structure
```sh
script -q -c "tree -n data/" tree.txt
```

## 2025-04-27

- Revised project proposal slides
  - Add descriptions on Models and Previous Works

- Completed data pipeline code for the CBIS-DDSM dataset.
  - Scan DICOM Files
  - Extract Metadata from File Paths
  - Pair Images and Masks
  - Merge Image–Mask Metadata and Clinical Metadata
  - Build TensorFlow Dataset

- Installed TensorFlow 2.15.0 with GPU support (CUDA 12.2, cuDNN 8.9).
- Installed PyTorch 2.2.0 with CUDA 12.1 support.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow==2.15.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

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