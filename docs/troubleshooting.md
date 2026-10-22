# Troubleshooting & Diagnostic Guide

## Overview
This document catalogs common environmental, network, and execution issues encountered when deploying or developing the **Datheon Machine Learning & Data Science Internship Suite**, along with verified remediation steps.

---

## 1. Network & Dataset Ingestion Issues

### Issue: Primary Sentiment URL returns HTTP 404
- **Symptom**: `urllib.error.HTTPError: HTTP Error 404: Not Found` on `https://raw.githubusercontent.com/datasets/sample-reviews/main/sentiment_reviews.csv`.
- **Root Cause**: The upstream GitHub repository path is inactive or deprecated.
- **Solution**: Handled automatically. The resilient data loader (`src/common/data_loader.py`) catches the 404, logs a warning, and switches to the verified fallback `https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv`.

### Issue: SSL Certificate Verification Failure
- **Symptom**: `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>`.
- **Root Cause**: Outdated corporate proxy certificates or missing Python certificate bundles on macOS / Windows.
- **Remediation**:
  ```bash
  # Windows: Upgrade certifi
  pip install --upgrade certifi

  # macOS: Run the Install Certificates command
  /Applications/Python\ 3.12/Install\ Certificates.command
  ```

---

## 2. Windows Environment Specifics

### Issue: PowerShell Execution Policy prevents activating virtual environments
- **Symptom**: `.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Remediation**: Run PowerShell as Administrator or set the process execution policy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```

### Issue: UnicodeEncodeError in Windows Console
- **Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character...`
- **Root Cause**: Legacy Windows console defaults to CP-1252 instead of UTF-8.
- **Remediation**: Set the Python UTF-8 environment variable before running scripts:
  ```powershell
  $env:PYTHONUTF8 = "1"
  python -c "import sys; print(sys.stdout.encoding)"
  ```

---

## 3. Machine Learning & Dependencies

### Issue: XGBoost installation fails on Windows
- **Symptom**: Missing Visual C++ Redistributable or OpenMP runtime error.
- **Root Cause**: XGBoost relies on OpenMP multi-threading libraries.
- **Remediation**:
  1. Install the Microsoft Visual C++ Redistributable: [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
  2. If XGBoost cannot be installed on a constrained machine, our pipeline automatically switches to `GradientBoostingClassifier` with zero code changes and identical evaluation interfaces.

### Issue: Scikit-learn Lasso convergence warning
- **Symptom**: `ConvergenceWarning: Objective did not converge...`
- **Explanation**: This occurs when `alpha` is small on dense correlated data before reaching full tolerance. We configured `max_iter=2000` which stabilizes optimization without altering test set metrics.

---

## 4. Dashboard & Port Availability

### Issue: Port 8000 already in use
- **Symptom**: `OSError: [WinError 10048] Only one usage of each socket address is normally permitted`.
- **Remediation**:
  Run `serve_dashboard.py` with a custom port:
  ```bash
  python serve_dashboard.py --port 8080
  ```
