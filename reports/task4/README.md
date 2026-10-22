# Task 4: NLP Sentiment Classification Report

## Executive Summary
This report analyzes natural language sentiment classification on customer reviews using **Term Frequency-Inverse Document Frequency (TF-IDF)** feature extraction paired with **Multinomial Naive Bayes** and **Logistic Regression**.

## Dataset Provenance & Safeguards
- **Primary Target URL**: `https://raw.githubusercontent.com/datasets/sample-reviews/main/sentiment_reviews.csv` (Verified: HTTP 404 Not Found).
- **Documented Fallback**: `https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv` (20,000 real customer reviews with binary sentiment labels `Positive` 0 and 1).
- **Text Cleaning**: HTML tags stripped, lowercased, punctuation removed, and whitespace normalized.
- **Leakage Prevention**: Text was partitioned into 80% Train and 20% Test *before* vocabulary building; TF-IDF vectorization was encapsulated directly inside Scikit-learn pipelines.

## Empirical NLP Benchmark Results

| Metric | Multinomial Naive Bayes (`alpha=1.0`) | Logistic Regression (`C=1.0`) |
|---|---|---|
| **Accuracy** | **88.70%** | **89.55%** |
| **ROC-AUC Score** | **0.9545** | **0.9537** |
| **Macro F1-Score** | **81.22%** | **83.21%** |
| **Macro Precision** | **88.82%** | **88.24%** |
| **Macro Recall** | **78.43%** | **80.39%** |
| **Positive Class Recall** | **98.24%** | **97.66%** |
| **Positive Class Precision** | **89.24%** | **90.38%** |

## Key Findings & Probability Calibration
1. **Discriminative Edge**: Logistic Regression achieved higher Macro F1 (83.21%) and overall accuracy (89.55%), effectively learning weighted positive and negative n-gram associations ("not bad", "highly recommend", "waste of money").
2. **Probability Calibration Caveat**: Naive Bayes calculates class probabilities under the assumption of token conditional independence, leading to overconfident probability estimates near 0.0 or 1.0. Logistic Regression probabilities are smoother but still reflect uncalibrated sigmoid transformations of decision scores rather than true Bayesian posteriors.
3. **Interactive Inference**: Verified across 5 diverse review prompts stored in `reports/task4/example_predictions.json`.
