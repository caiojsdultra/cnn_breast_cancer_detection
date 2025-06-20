# Breast Cancer Detection from Mammograms using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify mammographic images into two categories: **healthy** or **cancerous**. It was developed as a personal project to study deep learning techniques applied to medical imaging, with a focus on building everything from scratch — from data handling to evaluation.

---

## Project Overview

- Binary image classification using grayscale mammograms
- Data downloaded directly from Kaggle via API (`kagglehub`)
- Image preprocessing (resize, grayscale, normalization)
- CNN built using TensorFlow and Keras
- Data augmentation for better generalization
- Evaluation using accuracy, confusion matrix, ROC and PR curves

---

## Motivation

Early detection of breast cancer significantly increases survival rates. Mammograms are a primary screening tool, but interpreting them is not trivial. This project explores how **deep learning** can support the diagnostic process by learning to classify images automatically and accurately.

---

## Technologies Used

- `Python`
- `TensorFlow` / `Keras`
- `OpenCV`
- `NumPy`
- `Matplotlib`
- `scikit-learn`
- `tqdm`
- `kagglehub`

---

## Dataset

- **Source**: [Kaggle - Breast Cancer Detection](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection)
- Automatically downloaded using the `kagglehub` API
- Organized in three folders: `train`, `valid`, `test`
- Each folder contains subfolders: `0` (healthy), `1` (cancer)

---

## Preprocessing Steps

- Convert images to grayscale (1 channel instead of 3)
- Resize to **224x224 pixels**
- Normalize pixel values to `[0, 1]`
- Augment training data (zoom, shift, rotation) with `ImageDataGenerator`

---

## CNN Architecture

```python
Conv2D(16, kernel_size=5, activation='relu')  
MaxPooling2D(pool_size=2)  
Conv2D(32, kernel_size=3, activation='relu')  
MaxPooling2D(pool_size=2)  
Flatten()  
Dense(16, activation='relu')  
Dense(8, activation='relu')  
Dense(2, activation='softmax')
