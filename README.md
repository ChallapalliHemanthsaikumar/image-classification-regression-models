# Image Classification & Regression Models

This repository contains implementations of **EfficientNet, ResNet, and Vision Transformer (ViT)** models for **image classification** and **image regression** tasks. The goal is to demonstrate deep learning workflows for both discrete label prediction and continuous value prediction using modern computer vision architectures.

---

## 📂 Repository Structure


image-classification-regression-models/
│
├── README.md
├── requirements.txt # Python dependencies
├── data/ # Sample dataset or link instructions
├── notebooks/ # Jupyter notebooks
│ ├── image_classification.ipynb
│ └── image_regression.ipynb
├── models/ # Model definitions
│ ├── efficientnet.py
│ ├── resnet.py
│ └── vit.py
├── utils/ # Helper functions (data loaders, transforms)
│ └── utils.py
└── train.py # Script to train models



---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/image-classification-regression-models.git
cd image-classification-regression-models
pip install -r requirements.txt