# Installation & Environment Configuration Guide

## Overview
This document provides step-by-step instructions for establishing a reproducible Python environment to run, test, and replicate the **Datheon Machine Learning & Data Science Internship Suite**.

---

## 1. System Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python Version**: Python 3.10 to Python 3.12 (Python 3.12.10 verified)
- **Hardware**: Standard student laptop (CPU-friendly, no GPU or cloud API required)
- **Disk Space**: ~1.5 GB for packages and cached datasets

---

## 2. Environment Setup

### Option A: Standard Virtual Environment (Recommended)
```bash
# Clone the repository
git clone git@github.com:rajeshwari229080/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

---

## 3. Verifying Installation & Automated Tests
Execute the comprehensive Pytest suite:
```bash
pytest tests/ -v
```
Expected output: 19 test cases passing across all 4 project pipelines and common utilities.

---

## 4. Launching Deliverables

### A. Central Laboratory Dashboard
Launch the zero-dependency local dashboard server:
```bash
python serve_dashboard.py
```
Open your browser to: [http://localhost:8000](http://localhost:8000)

### B. Jupyter Notebooks
Launch the Jupyter interactive environment:
```bash
jupyter notebook
```
Navigate to `notebooks/` to inspect any of the four standalone project notebooks:
1. `01_multiclass_classification.ipynb`
2. `02_housing_price_regression.ipynb`
3. `03_customer_churn_classification.ipynb`
4. `04_nlp_sentiment_classification.ipynb`

### C. Headless Pipeline Execution
Run any project pipeline directly from the command line:
```bash
# Multi-class Iris pipeline
python -c "from src.classification.iris_pipeline import run_task1_pipeline; run_task1_pipeline()"

# California Housing price regression
python -c "from src.regression.housing_pipeline import run_task2_pipeline; run_task2_pipeline()"

# Telco customer churn classifier
python -c "from src.churn.churn_pipeline import run_task3_pipeline; run_task3_pipeline()"

# NLP sentiment classification
python -c "from src.sentiment.sentiment_pipeline import run_task4_pipeline; run_task4_pipeline()"
```
Generated reports, figures, and metrics are written directly to `reports/task1/`, `reports/task2/`, `reports/task3/`, and `reports/task4/`.
