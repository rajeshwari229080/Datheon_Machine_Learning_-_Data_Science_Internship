# Task 3: Predictive Customer Churn Classification Report

## Executive Summary
This report analyzes subscriber retention and churn risk using **Random Forest** (with balanced class weighting) and **XGBoost** (with `scale_pos_weight` ratio adjustments) on the IBM Telco Customer Churn dataset.

## Dataset Specifications
- **Source**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Total Customers**: 7,043 instances, 20 predictive features (customerID omitted)
- **Target**: `Churn` (Binary: Retained = 0, Churned = 1)
- **Imbalance Ratio**: 73.46% Retained (5,174) vs 26.54% Churned (1,869)
- **Data Cleaning**: 11 whitespace strings in `TotalCharges` coerced to NaN and imputed using median imputation.
- **Partitioning**: 80% Train (5,634), 20% Test (1,409), Stratified

## Empirical Classification Benchmark Results

| Metric | Random Forest (`class_weight='balanced'`) | XGBoost (`scale_pos_weight=2.77`) |
|---|---|---|
| **ROC-AUC Score** | **0.8419** | **0.8423** |
| **Accuracy** | **76.01%** | **76.22%** |
| **Macro Recall** | **76.52%** | **76.58%** |
| **Macro Precision** | **70.28%** | **70.47%** |
| **Macro F1-Score** | **72.12%** | **72.29%** |
| **Churn Class Recall** | **77.54%** | **77.27%** |
| **Churn Class Precision** | **52.32%** | **52.61%** |

## Critical Business & Technical Insights
1. **Target Achievement**: Both models achieved **ROC-AUC scores exceeding 0.84**, well surpassing the experimental target of 0.80.
2. **Impact of Imbalance Weighting**: Without loss weighting, naive classifiers achieve 79% accuracy but only 45% churn recall. By weighting the positive churn class by ~2.77x, recall jumped to **77.5%**, capturing over 3 out of every 4 defecting customers.
3. **Top Churn Predictors**:
   - `Contract_Month-to-month`: Customers on monthly plans churn at over 4x the rate of two-year contract holders.
   - `tenure`: Defection risk is heavily concentrated in the first 0 - 6 months of customer tenure.
   - `TotalCharges` & `MonthlyCharges`: Higher recurring bills significantly increase price sensitivity.
   - `InternetService_Fiber optic`: High friction service without bundled tech support drives defections.
4. **Decision Threshold Tuning**: For customer retention marketing, lowering the decision threshold to `0.35` captures 88% of all churners at an affordable cost of promotional messaging.
