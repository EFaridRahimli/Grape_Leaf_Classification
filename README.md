# Grape Leaf Disease Classification for Viticulture Optimization

## Overview
This project addresses a critical challenge in the wine industry: maintaining vineyard health across large, sparse plantations. Manual inspection of grape leaves is labor-intensive and prone to human error. This repository contains a deep learning pipeline designed to automate the detection of grape diseases using Convolutional Neural Networks (CNNs).

The model classifies grape leaf images into four categories:
* **Black Rot**
* **ESCA (Black Measles)**
* **Leaf Blight**
* **Healthy**

## Business Problem
Disease outbreaks can significantly impact crop yield and wine quality. By implementing an automated classification system via plantation cameras, the company can:
* **Reduce Labor Costs:**
* **Improve Timeliness:**
* **Ensure Product Quality:**

## Workflow
1.  **Exploratory Data Analysis (EDA):**
2.  **Data Preprocessing & Augmentation:** Implemented `ImageDataGenerator` to simulate real-world conditions increase model robustness.
3.  **Model Architecture:** :
    * 3 Convolutional layers with increasing filters (32, 64, 128).
    * Batch Normalization.
    * Dropout (0.5).
    * Softmax output layer for multi-class classification.
4.  **Experimentation:** Conducted **10 different experiments** varying optimizers (Adam vs. RMSprop), layer depths, and hyperparameters to find the optimal balance between accuracy and training time.

## Results
* **Final Accuracy:** 98.4% on the test set.
* **Optimization:** Identified Experiment #4 as the best-performing architecture for production.
* **Metrics:**.

## Requirements
* TensorFlow / Keras
* NumPy / Pandas
* Matplotlib / Seaborn / Plotly
* Scikit-learn
