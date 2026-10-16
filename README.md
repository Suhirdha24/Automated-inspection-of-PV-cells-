# ☀️ Automated Inspection of Photovoltaic (PV) Cells using Deep Learning & Explainable AI (Grad-CAM)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15%2B-red.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-PyTorch%20Image%20Models-orange.svg)](https://github.com/huggingface/pytorch-image-models)
[![Dataset](https://img.shields.io/badge/Dataset-ELPV%20Dataset-green.svg)](https://github.com/zae-bayern/elpv-dataset)
[![XAI](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-purple.svg)](https://arxiv.org/abs/1610.02391)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning framework for automated defect detection, severity grading, and explainable defect localization in solar photovoltaic (PV) cells from high-resolution **Electroluminescence (EL)** imagery.

---

## 📋 Table of Contents
- [📌 Overview](#-overview)
- [🔬 Industrial Motivation & Problem Statement](#-industrial-motivation--problem-statement)
- [📊 Dataset Information (ELPV)](#-dataset-information-elpv)
- [🏗️ Model Architectures](#️-model-architectures)
  - [1. Xception Backbone](#1-xception-backbone)
  - [2. EfficientNet-B3 + Full CBAM Attention](#2-efficientnet-b3--full-cbam-attention)
  - [3. EfficientNet-B0 + Light CBAM](#3-efficientnet-b0--light-cbam)
  - [4. Xception-71 Backbone](#4-xception-71-backbone)
- [⚙️ Training Strategy & Regularization](#️-training-strategy--regularization)
- [🔍 Explainable AI (Grad-CAM) & Industrial QA Dashboard](#-explainable-ai-grad-cam--industrial-qa-dashboard)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
- [📈 Evaluation & Results](#-evaluation--results)
- [🗺️ Enhancement Roadmap](#️-enhancement-roadmap)
- [📜 Citation & References](#-citation--references)

---

## 📌 Overview

Solar photovoltaic (PV) modules are exposed to severe environmental stressors (thermal cycling, mechanical loads, UV degradation, and hail impact) over their 20–30 year lifespans. Defects such as **micro-cracks**, **finger interruptions**, **cell breakages**, and **corrosion** significantly reduce power conversion efficiency and can cause dangerous localized hotspots.

This project implements an automated, high-precision inspection pipeline that:
1. **Classifies** solar cells into **Functional** vs. **Defective** states.
2. **Enhances subtle micro-cracks** using adaptive percentile luminance stretching.
3. **Localizes defect boundaries** via an **Explainable AI (Grad-CAM)** engine.
4. **Quantifies defect severity** using surface area ratios and an industrial **QA Decision (PASS / REJECT)** card.

```
+---------------------------------------------------------------------------------------------------------+
|                                    END-TO-END INSPECTION PIPELINE                                       |
+---------------------------------------------------------------------------------------------------------+
   [ Raw EL Cell Image ] 
            │
            ▼
   [ Data Preprocessing ] ────► AutoAugment (ImageNet) + CutMix + Resize (224x224 / 300x300)
            │
            ▼
   [ Deep CNN Backbone ]  ────► Xception / EfficientNet + CBAM (Channel & Spatial Attention)
            │
            ▼
   [ Classification Head ] ───► Loss: CrossEntropy with Label Smoothing (0.1) + AdamW + Cosine LR
            │
            ▼
   [ Explainable AI (XAI) ] ──► Grad-CAM Feature-Gradients Hook Engine
            │
            ▼
   [ 5-Panel QA Dashboard ] ──► 1. Raw EL Image
                                2. Contrast-Enhanced Crack Detail
                                3. Turbo Colormap Defect Heatmap
                                4. Boundary Segmentation & Contour Overlay
                                5. PASS / REJECT Decision + Defect Surface Ratio (%)
```

---

## 🔬 Industrial Motivation & Problem Statement

* **Standard Electroluminescence (EL) Imagery:** Uses near-infrared emission under forward bias to reveal internal lattice defects invisible to standard visual cameras.
* **The Black-Box Bottleneck:** Standard deep learning classifiers output only a probability score (`0` or `1`), without indicating **where** the crack or degradation zone is located.
* **Our Solution:** By coupling CNN backbones with attention mechanisms (**CBAM**) and an industrial **Grad-CAM 5-Panel QA Dashboard**, factory operators and QA inspectors can pinpoint the precise location, geometry, and surface area percentage of cell anomalies in real time.

---

## 📊 Dataset Information (ELPV)

The project utilizes the benchmark **ELPV (Electroluminescence Photovoltaic) Dataset**:
* **Total Samples:** 2,624 normalized grayscale EL images ($300 \times 300$ pixels, 8-bit).
* **Module Variety:** Extracted from 44 different crystalline silicon PV modules.
* **Cell Types:** Both **monocrystalline (mono)** and **polycrystalline (poly)** silicon cells.
* **Labels:** Originally annotated with defect probability estimates:
  - `0.0` : Completely functional (defect-free)
  - `0.33` : Mildly defective (isolated finger interruption / minor surface flaw)
  - `0.67` : Moderately defective (hairline crack / partial disconnection)
  - `1.0` : Severely defective (inactive cell segment / major structural fracture)
* **Dataset Splits:**
  - **Training Set:** 1,890 samples (~72%)
  - **Validation Set:** 472 samples (~18%)
  - **Test Set:** 262 samples (~10%)

---

## 🏗️ Model Architectures

The repository includes four model architectures tailored for solar defect inspection:

### 1. Xception Backbone
* Employs **Depthwise Separable Convolutions** to decouple cross-channel correlations from spatial correlations.
* Features a stem module, 11 middle flow residual blocks, and an exit flow with global average pooling.
* High representational capacity for fine crack textures.

### 2. EfficientNet-B3 + Full CBAM Attention
* Injects **Convolutional Block Attention Modules (CBAM)** into each sequential stage.
* **Channel Attention:** Computes cross-channel feature relationships using joint Adaptive AvgPool + MaxPool and an MLP bottleneck.
* **Spatial Attention:** Focuses on *where* defective patterns are located using inter-spatial feature descriptors and $7 \times 7$ convolutions.

### 3. EfficientNet-B0 + Light CBAM
* Optimized for **edge computing** and fast factory-line throughput.
* Strategically injects a light CBAM module ($3 \times 3$ spatial kernel) at the deepest convolutional block to minimize parameter overhead while preserving high defect sensitivity.
* Profiled with `thop` for FLOPs and parameter efficiency.

### 4. Xception-71 Backbone
* Deeper separable architecture (71 layers) for capturing multiscale defect signatures.

---

## ⚙️ Training Strategy & Regularization

| Hyperparameter / Component | Setting |
|---|---|
| **Optimizer** | `AdamW` (learning rate: `1e-4`, weight decay: `1e-5`) |
| **Scheduler** | `CosineAnnealingLR` ($T_{\max}=50$, $\eta_{\min}=10^{-6}$) |
| **Batch Size** | `32` |
| **Number of Epochs** | `50` |
| **Loss Function** | `CrossEntropyLoss` with label smoothing (`0.1`) |
| **Data Augmentation** | `AutoAugment` (ImageNet policy) + `CutMix` ($\alpha = 1.0$) |
| **Mixed Precision** | PyTorch AMP (`torch.amp.autocast`, `GradScaler`) for $2\times$ training speedup |

---

## 🔍 Explainable AI (Grad-CAM) & Industrial QA Dashboard

The notebook features a self-contained, publication-ready **Explainable AI (XAI)** engine based on **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

### 5-Panel QA Dashboard View
Each tested cell is evaluated through 5 synchronized diagnostic panels:
1. **[1] Raw EL Image:** Original cell image with true ground-truth label.
2. **[2] Enhanced Contrast:** Adaptive percentile luminance stretching to highlight faint hairline micro-cracks.
3. **[3] Defect Activation Heatmap:** Perceptually uniform `turbo` colormap displaying intensity values ($0.0 - 1.0$).
4. **[4] Defect Localization & Contours:** Automated threshold contour segmentation isolating defect boundaries.
5. **[5] Diagnostics & QA Decision:**
   - Confidence bar chart (Functional vs. Defective probabilities).
   - **Quality Decision Badge:** `PASS (FUNCTIONAL)` or `REJECT (DEFECTIVE)`.
   - **Defect Area Ratio:** Percentage of cell surface occupied by defect hotspots.
   - **Severity Triage:** `Nominal`, `Mild Defect`, `Moderate Defect`, or `Severe Defect`.

---

## 📁 Repository Structure

```
Automated-inspection-of-PV-cells-/
├── app.py                  # Interactive Gradio web application
├── requirements.txt        # Full dependencies for app & model
├── sample_images/          # Sample PV cell images for testing
│   ├── sample_functional_cell.png
│   ├── sample_cracked_cell.png
│   └── sample_finger_interruption.png
├── project_work2.ipynb     # Complete end-to-end training & Grad-CAM notebook
├── README.md               # Project documentation & inspection guide
└── checkpoints/            # Saved model weights (best_model.pth, best_model3.pth)
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Suhirdha24/Automated-inspection-of-PV-cells-.git
cd Automated-inspection-of-PV-cells-
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Gradio Web App
Launch the local web server to test drag-and-drop cell inspection:
```bash
python app.py
```
Open `http://localhost:7860` in your browser to inspect solar cell images with real-time Grad-CAM heatmaps and defect boundary overlays!

### 4. Run the Training & Diagnostic Notebook
Open and execute [`project_work2.ipynb`](project_work2.ipynb) in **Jupyter Lab**, **VS Code**, or **Google Colab** (with a T4/A100 GPU accelerator).

To generate Grad-CAM explanations for any model, run the dedicated inspection cell at the bottom of the notebook:
```python
# Launch the 5-Panel QA Dashboard on validation samples
visualize_pv_inspection_gallery(active_model, val_loader, num_samples=5, device=DEVICE, colormap="turbo")

# Run side-by-side comparison (Defective vs. Functional)
compare_defect_types(active_model, val_loader, n_each=3, device=DEVICE)
```

---

## 📈 Evaluation & Results

* **Validation F1-Score:** $\approx \mathbf{0.81}$ (peak performance reached with AdamW + Cosine Annealing + CBAM).
* **Validation Accuracy:** $\approx \mathbf{85.4\%}$
* **Inference Speed:** Real-time throughput (~15 ms per cell on an NVIDIA T4 GPU).
* **Explainability Verification:** Grad-CAM activation peaks closely align with verified micro-cracks, dark inactive clusters, and broken busbar contacts.

---

## 🗺️ Enhancement Roadmap

- [x] **CNN Backbones with Attention:** Xception, EfficientNet-B3 + CBAM, EfficientNet-B0 Light CBAM.
- [x] **Data Regularization:** AutoAugment + CutMix for robust generalization.
- [x] **Explainable AI Integration:** PyTorch hook-based Grad-CAM engine.
- [x] **Industrial Inspection Dashboard:** 5-panel diagnostic card with defect area quantification and contour segmentation.
- [x] **Web Deployment:** Interactive Gradio application (`app.py`) for drag-and-drop solar cell inspection.
- [ ] **Multi-Class Severity Classification:** Expanding to 4-tier classification (Functional, Mild, Moderate, Severe).
- [ ] **Vision Transformers:** Benchmarking Swin Transformer and ConvNeXt architectures.
- [ ] **Defect Object Detection:** YOLOv8-based bounding-box defect localization.

---

## 📜 Citation & References

If you use or build upon this work, please cite the ELPV dataset:
```bibtex
@article{deitsch2019automatic,
  title={Automatic classification of defective photovoltaic module cells in electroluminescence images},
  author={Deitsch, Sergiu and Christlein, Vincent and Berger, Stephan and Buerhop-Lutz, Claudia and Camus, Christian and Maier, Andreas},
  journal={Solar Energy},
  volume={185},
  pages={455--468},
  year={2019},
  publisher={Elsevier}
}
```

Grad-CAM reference:
```bibtex
@inproceedings{selvaraju2017grad,
  title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedaldi, Andrea and Parikh, Devi and Batra, Dhruv},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  pages={618--626},
  year={2017}
}
```

---

**Author:** [Suhirdha K S](https://github.com/Suhirdha24)  
**Project:** Automated Inspection of PV Cells  
**License:** [MIT License](LICENSE)
