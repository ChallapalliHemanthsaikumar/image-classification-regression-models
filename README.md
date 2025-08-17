# Image Classification & Regression Models

This repository showcases implementations of EfficientNet, ResNet, and Vision Transformer (ViT) models for image classification and image regression tasks. It demonstrates robust deep learning workflows for predicting discrete labels (classification) and continuous values (regression) using state-of-the-art computer vision architectures. The project includes well-documented Jupyter notebooks, training scripts, and utility functions to facilitate reproducible results.

The classification task focuses on a vegetable classification dataset, while the regression task tackles face age detection. Both tasks compare the performance of multiple models to highlight their strengths and weaknesses. Notebooks are compatible with Google Colab for easy experimentation.

## Repository Structure


    image-classification-regression-models/
    │
    ├── README.md                    # Project overview and instructions
    ├── requirements.txt             # Python dependencies
    ├── data/                       # Sample dataset or instructions to obtain data
    ├── notebooks/                  # Jupyter notebooks for experimentation
    │   ├── image_classification.ipynb  # Vegetable classification analysis
    │   └── image_regression.ipynb     # Face age detection analysis
    ├── output/                     # Model outputs and visualizations
    │   ├── image_classification_efficientnet_classification_report.png
    │   └── image_classification_resnet_classification_report.png
    ├── utils/                      # Helper functions (data loaders, transforms)
    └── utils.py                  


## 🚀 Features

- **Image Classification**: Evaluates ResNet and EfficientNet on a vegetable classification dataset with 15 classes (e.g., Bean, Tomato, Cucumber).
- **Image Regression**: Implements ResNet, EfficientNet, and ViT for face age detection, predicting continuous age values.
- **Comprehensive Metrics**: Includes classification metrics (Accuracy, Precision, Recall, F1-Score) and regression metrics (Mean Squared Error, R² Score).
- **Visualizations**: Classification reports and performance plots saved in the `output/` directory.
- **Google Colab Support**: Notebooks are designed to run seamlessly on Google Colab for GPU-accelerated training.
- **References to Research**: Includes citations to foundational papers for the models used (see References section).

## ⚙️ Installation

To get started, clone the repository and install the required dependencies:



Ensure you have Python 3.8+ and a compatible GPU for faster training (optional). The `requirements.txt` includes dependencies like `torch`, `torchvision`, `scikit-learn`, and `jupyter`. Alternatively, run the notebooks in Google Colab for a hassle-free setup with GPU support.

## 📊 Model Performance

### 1. Image Classification (Vegetable Dataset)

The classification task involves a dataset with 15 vegetable classes (e.g., Bean, Bitter Gourd, Tomato). Both ResNet and EfficientNet were evaluated, with EfficientNet outperforming ResNet in all metrics.

#### ResNet Results

- **Accuracy**: 0.981
- **F1-Score**: 0.981
- **Precision**: 0.982
- **Recall**: 0.981

Output: `output/image_classification_resnet_classification_report.png`

#### EfficientNet Results

- **Accuracy**: 0.999
- **F1-Score**: 0.999
- **Precision**: 0.999
- **Recall**: 0.999

Output: `output/image_classification_efficientnet_classification_report.png`

**Key Insight**: EfficientNet consistently achieved near-perfect performance, slightly outperforming ResNet, particularly for challenging classes like Brinjal and Cucumber.

### 2. Image Regression (Face Age Detection)

The regression task predicts continuous age values from face images using ResNet, EfficientNet, and Vision Transformer (ViT). ResNet outperformed both EfficientNet and ViT in this task.

#### ResNet Results

- **Mean Squared Error (MSE)**: 49.05
- **R² Score**: 0.916

#### EfficientNet Results

- **Mean Squared Error (MSE)**: 55.11
- **R² Score**: 0.906

#### Vision Transformer (ViT) Results

- **Mean Squared Error (MSE)**: 611.72
- **R² Score**: -0.037

**Key Insight**: ResNet demonstrated superior performance for age prediction, with the lowest MSE and highest R² score. ViT underperformed significantly, likely due to insufficient fine-tuning or dataset size.

## 📓 Usage

### Explore Notebooks:

- Run `notebooks/image_classification.ipynb` to train and evaluate ResNet and EfficientNet on the vegetable classification dataset.
- Run `notebooks/image_regression.ipynb` to train and evaluate ResNet, EfficientNet, and ViT on the face age detection dataset.

Both notebooks are compatible with Google Colab for GPU-accelerated training.

### Train Models:

Use the `train.py` script to train models from the command line:


### Visualize Results:

Classification reports and performance plots are saved in the `output/` directory.

## 🛠️ Utilities

The `utils/` directory contains helper functions in `utils.py` for:

- Data loading and preprocessing
- Custom transforms for image classification and regression
- Metric calculations (Accuracy, F1-Score, MSE, R²)

## 📚 References

This project is built upon foundational research in deep learning for computer vision. Key papers include:

- **ResNet**: He, K., et al. (2015). "Deep Residual Learning for Image Recognition." [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
- **EfficientNet**: Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." [arXiv:1905.11946](https://arxiv.org/abs/1905.11946).
- **Vision Transformer (ViT)**: Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).

## 📈 Future Improvements

- Add advanced data augmentation techniques to enhance model robustness.
- Fine-tune ViT for improved regression performance.
- Incorporate additional datasets for broader evaluation.
- Optimize training pipelines for faster convergence.

## 🙌 Contributions

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the repository. Please ensure code follows PEP 8 standards and includes clear documentation.

## 📬 Contact

For questions or feedback, connect with me on LinkedIn or open an issue in this repository.

[LinkedIn](https://www.linkedin.com/in/challapalli-hemanth-sai-kumar-7595931b0/)

Built by Hemanth Sai Kumar Challapalli