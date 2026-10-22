# Task 1: Multi-Class Benchmark Classification Report

## Executive Summary
This report presents the empirical benchmark results comparing **Logistic Regression (Softmax)** and **Decision Tree Classifier (CART)** on Ronald Fisher's Iris benchmark dataset.

## Dataset Specifications
- **Source**: `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv`
- **Total Samples**: 150 (50 Setosa, 50 Versicolor, 50 Virginica)
- **Features**: Sepal Length, Sepal Width, Petal Length, Petal Width (cm)
- **Partition Ratio**: 80% Train (120 samples), 20% Test (30 samples), Stratified
- **Random Seed**: 42

## Benchmark Evaluation Results

| Metric | Logistic Regression (Pipeline) | Decision Tree Classifier |
|---|---|---|
| **Accuracy** | **93.33%** | **93.33%** |
| **Macro Precision** | **94.44%** | **94.44%** |
| **Macro Recall** | **93.33%** | **93.33%** |
| **Macro F1-Score** | **93.27%** | **93.27%** |
| **Setosa F1** | **1.0000** | **1.0000** |
| **Versicolor F1** | **0.9000** | **0.9000** |
| **Virginica F1** | **0.9000** | **0.9000** |

## Key Findings & Visualizations
1. **Linear Separability**: `Iris setosa` is 100% linearly separable from Versicolor and Virginica using petal length alone.
2. **Boundary Classification**: Both models correctly classify 28 out of 30 test samples, with slight confusion occurring exclusively between Versicolor and Virginica along the overlapping petal threshold (~4.8 - 5.1 cm).
3. **Decision Boundary Analysis**: The 2-feature decision boundary surface (`decision_boundary_logistic_regression.png` and `decision_boundary_decision_tree.png`) highlights the contrast between the smooth linear hyperplanes of Logistic Regression and the orthogonal rectangular partition steps of the Decision Tree.
