# Explainable AI for COVID-19 Detection using Chest X-Ray Images

## Overview

This project presents an Explainable Artificial Intelligence (XAI) framework for automated COVID-19 detection from Chest X-Ray (CXR) images. The system combines state-of-the-art Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) with multiple explainability techniques to provide both accurate predictions and interpretable decision-making.

The objective is not only to classify chest X-ray images into diagnostic categories but also to visualize and quantitatively evaluate the reasoning behind model predictions, thereby improving trustworthiness in AI-assisted healthcare systems.

---

## Key Features

- Multi-class Chest X-Ray Classification
- ResNet-50 and Vision Transformer (ViT-Small) Models
- Explainable AI (XAI) Framework
- GradCAM Visual Explanations
- Attention Rollout for Vision Transformers
- LIME Explanations
- Quantitative Explainability Evaluation
- Novel Faithfulness-Localization Score (FLS)
- Comprehensive Performance Analysis

---

## Dataset

The project utilizes a large-scale Chest X-Ray dataset consisting of COVID-19, Normal, and Viral Pneumonia cases.

### Dataset Statistics

| Category | Count |
|-----------|-----------|
| Total Images | 40,000+ |
| COVID-19 | ~13,000+ |
| Normal | ~13,000+ |
| Viral Pneumonia | ~13,000+ |

### Dataset Split

| Split | Images |
|----------|----------|
| Training | 30,000 |
| Validation | 6,000 |
| Test | 4,000 |
| Total | 40,000 |

### Data Preprocessing

- Image Resizing (224 × 224)
- Pixel Normalization
- Data Augmentation
- Horizontal Flipping
- Random Rotation
- Train-Validation-Test Split
- Batch Loading using PyTorch DataLoaders

### Effective Training Distribution

Through preprocessing and augmentation, the effective training distribution exceeded **100,000+ image variations**, improving robustness and generalization.

---

## Models Implemented

### ResNet-50

A deep Convolutional Neural Network utilizing residual connections for efficient feature extraction and classification.

### Vision Transformer (ViT-Small)

A transformer-based architecture that treats images as sequences of patches and captures global contextual relationships through self-attention mechanisms.

---

## Explainability Techniques

### GradCAM

Generates class-discriminative localization maps highlighting image regions contributing most to CNN predictions.

### Attention Rollout

Visualizes the attention distribution across image patches in Vision Transformers.

### LIME

Produces local surrogate explanations that identify influential image regions affecting predictions.

---

## Quantitative Explainability Metrics

The generated explanations were evaluated using:

### Insertion AUC

Measures how rapidly model confidence increases when important pixels are progressively inserted.

### Deletion AUC

Measures the reduction in confidence when important pixels are removed.

### Entropy

Quantifies explanation concentration and focus.

### AOPC (Area Over Perturbation Curve)

Measures prediction sensitivity to perturbations of important regions.

### Faithfulness-Localization Score (FLS)

A novel metric proposed in this project that jointly evaluates:

- Explanation Faithfulness
- Spatial Localization Quality

---

# Results

## Classification Performance

| Metric | ResNet-50 | ViT-Small |
|----------|----------|----------|
| Accuracy | 99.1% | 99.8% |
| Precision | 99.1% | 99.8% |
| Recall | 99.1% | 99.8% |
| F1-Score | 99.1% | 99.8% |

---

## Confusion Matrix Results

### ResNet-50

- COVID Correctly Classified: 144 / 145
- Normal Correctly Classified: 143 / 145
- Viral Pneumonia Correctly Classified: 144 / 145

### ViT-Small

- COVID Correctly Classified: 144 / 145
- Normal Correctly Classified: 145 / 145
- Viral Pneumonia Correctly Classified: 145 / 145

---

## Explainability Evaluation

| Metric | Score |
|----------|----------|
| Insertion AUC | 0.736 |
| Deletion AUC | 0.449 |
| Entropy | 0.950 |
| AOPC | 0.030 |

---

## Generated Outputs

The framework automatically generates:

- Training Curves
- Accuracy Curves
- Loss Curves
- Confusion Matrices
- GradCAM Visualizations
- Attention Rollout Maps
- LIME Explanations
- Quantitative Explainability Metrics
- FLS Comparison Analysis

---

## Repository Structure

```text
covid19_cv_exp/
│
├── notebook/
│   └── COVID19_XAI.ipynb
│
├── results/
│   ├── sample_images.png
│   ├── training_curves.png
│   ├── confusion_matrix_resnet50.png
│   ├── confusion_matrix_vit_small.png
│   ├── gradcam_results.png
│   ├── attention_rollout.png
│   ├── lime_results.png
│   ├── results_summary.png
│   └── fls_comparison.png
│
├── report/
│   └── Final_Report.pdf
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Loki3001/covid19_cv_exp.git
```

Move into the project directory:

```bash
cd covid19_cv_exp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1

Download the dataset.

### Step 2

Place the dataset in:

```text
COVID_19_dataset/
```

### Step 3

Open:

```text
notebook/COVID19_XAI.ipynb
```

### Step 4

Run all cells sequentially.

---

## Applications

- Medical Image Analysis
- COVID-19 Screening
- Explainable Healthcare AI
- Clinical Decision Support Systems
- Trustworthy Artificial Intelligence

---

## Future Work

- Multi-center Clinical Datasets
- Radiologist Validation Studies
- Lung Segmentation Guided Explanations
- Uncertainty-Aware Predictions
- Real-Time Clinical Deployment
- Enhanced Explainability Evaluation Frameworks

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- TIMM
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Scikit-Learn
- LIME
- Google Colab

---

## Author

**Lokesh Yadav**

B.Tech Artificial Intelligence & Machine Learning (AIML)

University School of Automation and Robotics (USAR)

Guru Gobind Singh Indraprastha University

---



## License

This project is intended for educational, research, and academic purposes.
