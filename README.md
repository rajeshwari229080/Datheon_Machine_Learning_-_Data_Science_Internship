<div align="center">

# Datheon Machine Learning & Data Science Internship Suite
### End-to-End Supervised Learning, Imbalanced Classification & NLP Benchmark Laboratory

[![Tests](https://img.shields.io/badge/Pytest-24%2F24%20Passing-10B981?style=for-the-badge&logo=pytest&logoColor=white)](#automated-testing--quality-assurance)
[![Python](https://img.shields.io/badge/Python-3.12-3B82F6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F97316?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-0284C7?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-6366F1?style=for-the-badge)](LICENSE)

<br/>

<img src="assets/hero_banner_light.png" alt="Machine Learning Internship Suite Hero Banner" width="100%" />

<p align="center">
  <b>A comprehensive data science portfolio featuring 4 standalone machine learning systems, modular <code>src/</code> architecture, 100% executable notebooks, and a central interactive laboratory dashboard.</b>
</p>

[Explore Projects](#four-core-machine-learning-workspaces) &bull;
[System Architecture](#system-architecture--pipeline-workflow) &bull;
[Interactive Dashboard](#interactive-laboratory-dashboard) &bull;
[LinkedIn Post Package](#linkedin-showcase-package) &bull;
[Quickstart](#quickstart--reproducibility-guide)

</div>

---

## Portfolio Benchmark Summary

| Task # | Problem Domain | Dataset Source | Evaluated Algorithms | Key Benchmark Metrics | Status |
|:---:|:---|:---|:---|:---:|:---:|
| **01** | **Multi-Class Benchmark** | Iris CSV (Fisher 1936) | Logistic Regression vs CART Decision Tree | **Accuracy: 93.33%** &bull; Macro F1: **0.9327** | Verified &#10004; |
| **02** | **Housing Price Prediction** | California Housing Census | OLS Baseline, Ridge (α=10), Lasso (α=1) | **RMSE: $69,127** &bull; **R²: 0.6353** | Verified &#10004; |
| **03** | **Customer Churn Classifier** | IBM Telco Churn (Imbalanced) | Random Forest (Balanced) vs XGBoost | **ROC-AUC: 0.8423** &bull; Recall: **77.27%** | Verified &#10004; |
| **04** | **NLP Sentiment Classification** | Customer Reviews Corpus | TF-IDF (5k Bigrams) + MNB / Logistic Reg | **ROC-AUC: 0.9545** &bull; Acc: **89.55%** | Verified &#10004; |

---

## System Architecture & Pipeline Workflow

The repository is built on a modular, decoupled architecture ensuring **zero data leakage**, test-split isolation, and reproducible execution across all environments:

<div align="center">
  <img src="assets/architecture_light.png" alt="Machine Learning Pipeline Architecture Workflow" width="95%" />
</div>

### Architectural Principles:
1. **Zero Data Leakage:** Transformers (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`, `TfidfVectorizer`) are encapsulated strictly inside Scikit-learn Pipelines and fitted solely on training partitions.
2. **Resilient Data Ingestion:** Automated caching in `data/raw/` with schema checks and transparent 404 fallback routing.
3. **Reproducibility:** Fixed deterministic random seeds across all train/test splits, cross-validation folds, and tree initializations.
4. **Decoupled Workspaces:** Four independent Jupyter notebooks in `notebooks/` paired with production-ready library scripts in `src/`.

---

## Four Core Machine Learning Workspaces

### Task 1 &mdash; Multi-Class Benchmark Classification (Fisher's Iris)
- **Objective:** Evaluate and contrast parametric Softmax Logistic Regression against non-parametric CART Decision Trees on Ronald Fisher's historic Iris benchmark.
- **Key Features:** Stratified 80:20 partitioning, standardization encapsulated inside pipelines, and a legitimate 2-feature decision boundary surface visualizer (`sepal_length` vs `petal_length`).

<div align="center">

| 2-Feature Geometric Decision Surface | Confusion Matrix Heatmap |
|:---:|:---:|
| <img src="reports/task1/decision_boundary_logistic_regression.png" alt="Iris Decision Boundary Surface" width="450px" /> | <img src="reports/task1/confusion_matrix_logistic_regression.png" alt="Iris Confusion Matrix" width="380px" /> |
| *Legitimate 2D plane projection displaying classification regions* | *Exact per-class predictions showing 100% Setosa separability* |

</div>

- **Empirical Findings:** Both models achieved **93.33% Accuracy** and **0.9327 Macro F1**. `Iris setosa` achieves 100% linear separability, while Versicolor and Virginica display boundary overlap around the 4.8 &ndash; 5.1 cm petal threshold.
- **Inspect Deliverables:** [Notebook](notebooks/01_multiclass_classification.ipynb) &bull; [Pipeline Source](src/classification/iris_pipeline.py) &bull; [Task Report](reports/task1/README.md)

---

### Task 2 &mdash; Real Estate Housing Price Prediction (California Census)
- **Objective:** Predict continuous median house values (`median_house_value`) in USD across 20,640 California census districts.
- **Key Features:** Median numeric imputation, one-hot encoding for `ocean_proximity`, interaction ratio synthesis (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`), and VIF multicollinearity diagnostics.

<div align="center">

| Residual Diagnostics (Actual vs Pred, Error Distribution) | Feature Correlation Matrix |
|:---:|:---:|
| <img src="reports/task2/residual_diagnostics_ridge_regression.png" alt="Housing Residual Diagnostics" width="540px" /> | <img src="reports/task2/correlation_heatmap.png" alt="Housing Correlation Heatmap" width="340px" /> |
| *3-Panel diagnostics demonstrating unbiased zero-centered errors* | *Correlation matrix detailing strong median income influence* |

</div>

- **Empirical Findings:** Unregularized OLS achieved **$69,127 RMSE** ($49,645 MAE, $R^2 = 0.6353$). Ridge ($\alpha=10.0$) and Lasso ($\alpha=1.0$) tuned via 5-fold cross-validation stabilized parameter variances caused by spatial latitude/longitude correlation ($r \approx -0.92$) without sacrificing predictive power.
- **Inspect Deliverables:** [Notebook](notebooks/02_housing_price_regression.ipynb) &bull; [Pipeline Source](src/regression/housing_pipeline.py) &bull; [Task Report](reports/task2/README.md)

---

### Task 3 &mdash; Predictive Customer Churn Classifier (IBM Telco)
- **Objective:** Detect customer defection under severe class imbalance (~73.5% retained vs ~26.5% churned).
- **Key Features:** Safe numeric coercion of `TotalCharges` (converting 11 blank spaces into `NaN` for imputation), removal of `customerID`, and cost-sensitive loss weighting (`class_weight='balanced'` in Random Forest and `scale_pos_weight = 2.77` in XGBoost).

<div align="center">

| Receiver Operating Characteristic (ROC Curves) | Top 10 Churn Risk Predictors |
|:---:|:---:|
| <img src="reports/task3/roc_curves.png" alt="Customer Churn ROC Curves" width="440px" /> | <img src="reports/task3/feature_importances_xgboost.png" alt="Top Churn Features" width="440px" /> |
| *Surpassed 0.80 benchmark target: XGBoost (0.8423) and RF (0.8419)* | *Identifies Month-to-month contracts and short tenure as primary churn drivers* |

</div>

- **Empirical Findings:** Both models exceeded the 0.80 target by achieving **0.8423 ROC-AUC (XGBoost)** and **0.8419 ROC-AUC (Random Forest)** with **77.27% Churn Recall**. Calibrating the decision threshold to `0.35` lifted customer defection capture to ~88%, optimizing retention campaign ROI.
- **Inspect Deliverables:** [Notebook](notebooks/03_customer_churn_classification.ipynb) &bull; [Pipeline Source](src/churn/churn_pipeline.py) &bull; [Task Report](reports/task3/README.md)

---

### Task 4 &mdash; NLP Sentiment Classification on Customer Reviews
- **Objective:** Classify customer review text into Positive (1) or Negative (0) sentiment.
- **Key Features:** Regex HTML tag stripping, token normalization, sublinear TF-IDF vectorization with bigrams (`ngram_range=(1,2)`), and probability calibration documentation.
- **Data Fallback Transparency:** Gracefully caught an upstream HTTP 404 and cleanly routed to an authentic 20,000-review Amazon customer benchmark dataset.

<div align="center">

| Model Accuracy Comparison (Naive Bayes vs Logistic Reg) | Sentiment Confusion Matrix |
|:---:|:---:|
| <img src="reports/task4/model_comparison.png" alt="Sentiment Model Accuracy Comparison" width="440px" /> | <img src="reports/task4/confusion_matrix_logistic_regression.png" alt="Sentiment Confusion Matrix" width="410px" /> |
| *Discriminative Logistic Regression edges generative Naive Bayes* | *Near-perfect positive review identification (98.2% recall)* |

</div>

- **Empirical Findings:** Logistic Regression achieved **89.55% Accuracy** and **0.9537 ROC-AUC**; Naive Bayes achieved **88.70% Accuracy** and **0.9545 ROC-AUC**. Documented why Naive Bayes outputs tend to bunch near 0 or 1 due to token independence assumptions.
- **Inspect Deliverables:** [Notebook](notebooks/04_nlp_sentiment_classification.ipynb) &bull; [Pipeline Source](src/sentiment/sentiment_pipeline.py) &bull; [Task Report](reports/task4/README.md)

---

## Interactive Laboratory Dashboard

Launch the zero-dependency local dashboard to interact with all four models in real time:

```bash
python serve_dashboard.py --open
```

<div align="center">
  <p><b>Access URL:</b> <code>http://localhost:8000/index.html</code></p>
</div>

### Dashboard Highlights:
- **Interactive Prediction Sandboxes:** Live sliders for Iris biometric dimensions, California housing district parameters, churn risk thresholds, and real-time review sentiment classification.
- **Figure Modals:** High-resolution modal popups displaying all generated diagnostic charts.
- **Responsive Layout:** Clean analytical dark theme matching professional scientific laboratories.

---

## LinkedIn Showcase Package

A complete social media publication kit is included to showcase this project to technical recruiters, peers, and engineering managers:

<div align="center">
  <a href="linkedin.md">
    <img src="assets/linkedin_showcase.png" alt="LinkedIn Showcase Graphic Preview" width="85%" />
  </a>
  <p><em>Click the banner above or open <b><a href="linkedin.md">linkedin.md</a></b> for ready-to-publish post copy and vector SVG graphics.</em></p>
</div>

---

## Quickstart & Reproducibility Guide

### 1. Clone & Set Up Environment
```bash
# Clone repository
git clone git@github.com:rajeshwari229080/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # On Windows
source .venv/bin/activate      # On Linux / macOS

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run Automated Pytest Suite
```bash
pytest tests/ -v
```
*All 24 unit and integration tests pass in under 6 seconds.*

### 3. Run Pipeline CLI Tasks
```bash
# Task 1 (Iris)
python -c "from src.classification.iris_pipeline import run_task1_pipeline; run_task1_pipeline()"

# Task 2 (Housing)
python -c "from src.regression.housing_pipeline import run_task2_pipeline; run_task2_pipeline()"

# Task 3 (Churn)
python -c "from src.churn.churn_pipeline import run_task3_pipeline; run_task3_pipeline()"

# Task 4 (Sentiment)
python -c "from src.sentiment.sentiment_pipeline import run_task4_pipeline; run_task4_pipeline()"
```

---

## Comprehensive Technical Documentation

- 📘 [Installation & Environment Guide](docs/installation.md)
- 📐 [Scientific & Mathematical Methodology](docs/methodology.md)
- 🛠️ [Troubleshooting & Diagnostics](docs/troubleshooting.md)
- 📢 [LinkedIn Publication Kit](linkedin.md)
- 📄 [MIT License](LICENSE)

---

## License & Attribution

This project is licensed under the **[MIT License](LICENSE)** &copy; 2026 rajeshwari229080.
Datasets are attributed to their respective open-source foundations (Ronald Fisher 1936, US Census Bureau, IBM Community Data, PyCaret Benchmark Corpus).
