# Datheon Machine Learning & Data Science Internship Suite

An execution-ready, enterprise-grade machine learning portfolio comprising four independent supervised learning and natural language processing projects, unified under an analytical data science laboratory dashboard.

---

## Portfolio Summary Matrix

| Task # | Project Domain | Benchmark Dataset | Evaluated Algorithms | Primary Metric | Model Status |
|:---:|:---|:---|:---|:---:|:---:|
| **01** | **Multi-Class Benchmark** | Iris CSV (Fisher 1936) | Logistic Regression vs Decision Tree | **Accuracy: 93.33%** | Complete / Verified |
| **02** | **Real Estate Price Prediction** | California Housing Census | OLS, Ridge, & Lasso Regression | **RMSE: $69,127 / R²: 0.6353** | Complete / Verified |
| **03** | **Customer Churn Discrimination** | IBM Telco Churn (Imbalanced) | Random Forest vs XGBoost | **ROC-AUC: 0.8423** | Complete / Verified |
| **04** | **NLP Sentiment Classification** | Customer Reviews Corpus | Naive Bayes vs Logistic Regression | **ROC-AUC: 0.9545 / Acc: 89.55%** | Complete / Verified |

---

## Architectural Layout

```
├── index.html                               # Central Analytical Laboratory Dashboard
├── serve_dashboard.py                       # Zero-dependency local dashboard HTTP server
├── requirements.txt                         # Exact version-pinned project dependencies
├── README.md                                # Comprehensive project portfolio documentation
├── notebooks/                               # Standalone, end-to-end executable notebooks
│   ├── 01_multiclass_classification.ipynb   # Task 1: Iris multi-class benchmark & 2D surface
│   ├── 02_housing_price_regression.ipynb    # Task 2: California housing OLS, Ridge, Lasso & VIF
│   ├── 03_customer_churn_classification.ipynb # Task 3: IBM Telco churn with imbalance weighting
│   └── 04_nlp_sentiment_classification.ipynb  # Task 4: TF-IDF customer review sentiment engine
├── src/                                     # Reusable, modular Python library
│   ├── common/
│   │   ├── data_loader.py                   # Caching downloader with fallback and validation
│   │   ├── evaluation.py                    # Metric computations and JSON serialization
│   │   └── visualization.py                 # Laboratory-styled charts (ROC, CM, Residuals)
│   ├── classification/
│   │   └── iris_pipeline.py                 # Task 1 pipeline, 2D boundary, and inference
│   ├── regression/
│   │   └── housing_pipeline.py              # Task 2 interactions, VIF, and price estimator
│   ├── churn/
│   │   └── churn_pipeline.py                # Task 3 imbalance handling and threshold tuner
│   └── sentiment/
│       └── sentiment_pipeline.py            # Task 4 text sanitization, TF-IDF, and inference
├── reports/                                 # Generated evaluation assets and high-res figures
│   ├── task1/                               # Confusion matrices, decision boundaries, metrics.json
│   ├── task2/                               # Residual diagnostics, correlation heatmap, VIF
│   ├── task3/                               # ROC curves, top 10 feature importances
│   └── task4/                               # Sentiment confusion matrices, example predictions
├── tests/                                   # Automated Pytest suite (100% passing)
│   ├── test_data_loading.py                 # Data cache and network failure tests
│   ├── test_classification.py               # Pipeline leakage prevention and inference tests
│   ├── test_regression.py                   # Feature interaction and VIF calculation tests
│   ├── test_churn.py                        # Safe coercion and decision threshold tests
│   └── test_sentiment.py                    # HTML text sanitization and TF-IDF tests
└── docs/                                    # Technical documentation & reference guides
    ├── installation.md                      # Setup and virtual environment guide
    ├── methodology.md                       # Mathematical rationale and leakage prevention
    └── troubleshooting.md                   # Diagnostics for Windows, SSL, and runtimes
```

---

## Quickstart & Replication

### 1. Environment Installation
```bash
# Clone the repository
git clone git@github.com:rajeshwari229080/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1    # On Windows
source .venv/bin/activate      # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```
*All 19 tests pass successfully with zero failures.*

### 3. Launch Central Laboratory Dashboard
```bash
python serve_dashboard.py --open
```
Access the dashboard at [http://localhost:8000](http://localhost:8000) to inspect all four interactive project workspaces, real-time metrics, confusion matrices, and interactive prediction sandboxes.

### 4. Interactive Jupyter Notebooks
```bash
jupyter notebook
```
Open any notebook in `notebooks/` to inspect detailed code execution, narrative explanations, exploratory plots, and interactive inference.

---

## Detailed Task Summaries

### Task 1 — Multi-Class Benchmark Classification (Iris)
- **Objective**: Compare parametric Logistic Regression against non-parametric Decision Tree classification on the 3-class Iris benchmark.
- **Leakage Prevention**: `StandardScaler` encapsulated in Scikit-learn `Pipeline` objects and fitted strictly on the training partition.
- **Decision Surface**: Includes a legitimate 2-feature decision boundary visualizer (`sepal_length` vs `petal_length`) with explicit documentation of its 2D subspace limitation.
- **Empirical Results**: Both models achieve **93.33% Accuracy** and **93.27% Macro F1**. `Iris setosa` achieves 100% linear separability.

### Task 2 — Real Estate Housing Price Prediction (California Housing)
- **Objective**: Model median housing values across 20,640 California census blocks using Ordinary Least Squares (OLS), Ridge (L2), and Lasso (L1).
- **Engineering**: Interaction ratios (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`).
- **Diagnostics**: Multicollinearity diagnosed via Variance Inflation Factors (VIF); 3-panel residual diagnostics (Actual vs Pred, Residual vs Pred, Error histogram).
- **Hyperparameter Tuning**: Ridge and Lasso regularization strengths tuned via 5-fold cross-validation on the training set.
- **Empirical Results**: Test RMSE of **$69,127.04** ($49,645 MAE, R² = 0.6353). Ridge and Lasso stabilize coefficient variance on collinear coordinates.

### Task 3 — Predictive Customer Churn Classifier (IBM Telco Churn)
- **Objective**: Detect subscriber churn under severe class imbalance (73.5% non-churn vs 26.5% churn).
- **Data Cleaning**: Safely coerced `TotalCharges` from string to numeric; dropped arbitrary `customerID` identifiers.
- **Imbalance Mitigation**: Applied `class_weight='balanced'` in Random Forest and `scale_pos_weight = 2.77` in XGBoost.
- **Threshold Calibration**: Engineered asymmetric decision threshold tuning; lowering threshold to 0.35 increases churn capture to ~88%.
- **Empirical Results**: Surpassed the 0.80 target by achieving **ROC-AUC of 0.8419 (Random Forest)** and **0.8423 (XGBoost)** with 77.5% churn recall.

### Task 4 — NLP Sentiment Classification (Customer Reviews)
- **Objective**: Classify customer review sentiment using TF-IDF feature extraction, Multinomial Naive Bayes, and Logistic Regression.
- **Data Fallback Transparency**: Caught HTTP 404 on the primary sample URL and cleanly routed to the verified 20,000-record Amazon Customer Reviews benchmark dataset (`https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv`).
- **Preprocessing**: HTML stripping, token normalization, and sublinear TF-IDF vectorization with bigrams (`ngram_range=(1,2)`).
- **Probability Calibration**: Documented probability calibration caveats for Naive Bayes and Logistic Regression.
- **Empirical Results**: Logistic Regression achieved **89.55% Accuracy** and **ROC-AUC of 0.9537**; Naive Bayes achieved **88.70% Accuracy** and **ROC-AUC of 0.9545**.

---

## Technical Documentation
- [Installation Guide](docs/installation.md)
- [Methodology & Mathematics](docs/methodology.md)
- [Troubleshooting & Diagnostics](docs/troubleshooting.md)
- [Task 1 Report](reports/task1/README.md)
- [Task 2 Report](reports/task2/README.md)
- [Task 3 Report](reports/task3/README.md)
- [Task 4 Report](reports/task4/README.md)

---

## License & Attribution
- Iris Dataset: BSD-3-Clause (Ronald Fisher, 1936)
- California Housing Dataset: CC0 / Public Domain (1990 California Census)
- IBM Telco Churn Dataset: Apache-2.0 / IBM Community
- Amazon Customer Reviews Dataset: Public Open Source Corpus
