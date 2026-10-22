# Task 2: Real Estate Housing Price Prediction Report

## Executive Summary
This report analyzes California district median housing price prediction using **Ordinary Least Squares (OLS)**, **Ridge Regression (L2 Regularization)**, and **Lasso Regression (L1 Regularization)**.

## Dataset Specifications
- **Source**: `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv`
- **Total Records**: 20,640 district census blocks
- **Target**: `median_house_value` (continuous numeric, USD)
- **Engineered Interactions**:
  - `rooms_per_household = total_rooms / households`
  - `bedrooms_per_room = total_bedrooms / total_rooms`
  - `population_per_household = population / households`
- **Train/Test Partition**: 80% Train (16,512), 20% Test (4,128)
- **Missing Value Handling**: Median imputation fitted strictly on train fold

## Empirical Regression Benchmark Results

| Model | Test RMSE ($) | Test MAE ($) | R² Score | Best Tuning Parameter |
|---|---|---|---|---|
| **OLS Baseline** | **$69,127.04** | **$49,645.49** | **0.6353** | Unregularized |
| **Ridge Regression** | **$69,134.97** | **$49,652.14** | **0.6353** | `alpha = 10.0` (5-fold CV) |
| **Lasso Regression** | **$69,128.08** | **$49,646.41** | **0.6353** | `alpha = 1.0` (5-fold CV) |

## Diagnostic Observations
1. **Multicollinearity & Spatial Clustering**: Latitude and Longitude exhibit high correlation (-0.92) representing the diagonal geographic span of California. Rather than deleting geographic location, Ridge and Lasso stabilize coefficient variance while preserving spatial prediction power.
2. **Income Elasticity**: `median_income` is the single strongest predictive feature (correlation = 0.69).
3. **Residual Diagnostics**: Residual distributions across all three models are centered at $0 with symmetric bell-curve characteristics, with minor heteroscedasticity for luxury homes caused by the $500,001 artificial census cap.
