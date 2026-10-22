"""Automated unit and integration tests for Task 3: Customer Churn Classification Pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.churn.churn_pipeline import (
    CATEGORICAL_COLS,
    IDENTIFIER_COL,
    NUMERICAL_COLS,
    TARGET_COL,
    build_churn_pipeline,
    build_churn_preprocessor,
    clean_churn_dataset,
    predict_churn,
    prepare_churn_data,
    train_and_evaluate_churn_models,
)


def test_clean_churn_dataset_coercion():
    """Verify that TotalCharges is coerced to numeric and customerID is removed."""
    df_raw = pd.DataFrame(
        {
            "customerID": ["7590-VHVEG", "5575-GNVDE", "3668-QPYBK"],
            "TotalCharges": ["29.85", " ", "108.15"],  # Contains blank space
            "Churn": ["No", "Yes", "No"],
            "tenure": [1, 34, 2],
            "MonthlyCharges": [29.85, 56.95, 53.85],
        }
    )
    df_clean = clean_churn_dataset(df_raw)

    assert IDENTIFIER_COL not in df_clean.columns
    assert df_clean["TotalCharges"].dtype == np.float64
    assert np.isnan(df_clean["TotalCharges"].iloc[1])
    assert list(df_clean["Churn"]) == [0, 1, 0]


def test_prepare_churn_data_stratification():
    """Verify stratified partitioning preserves churn ratio."""
    X_train, X_test, y_train, y_test, df = prepare_churn_data(random_state=42, test_size=0.2)

    total = len(df)
    assert len(X_train) == int(total * 0.8)
    assert len(X_test) == total - len(X_train)
    # Stratification check: difference in churn rate between train and test < 1%
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_churn_preprocessor():
    """Verify ColumnTransformer handles imputation and onehot encoding cleanly."""
    X_train, X_test, y_train, y_test, _ = prepare_churn_data()
    preprocessor = build_churn_preprocessor()

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert not np.isnan(X_train_trans).any(), "Imputer must eliminate all missing values"


def test_predict_churn_thresholds():
    """Verify prediction helper and threshold sensitivity."""
    X_train, X_test, y_train, y_test, _ = prepare_churn_data()
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    pipe = build_churn_pipeline(rf)
    pipe.fit(X_train, y_train)

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 70.35,
    }

    # High threshold -> should predict retain
    res_high = predict_churn(pipe, sample_customer, threshold=0.99)
    assert res_high["predicted_class"] == 0
    assert res_high["predicted_label"] == "Will Retain"

    # Very low threshold -> should predict churn
    res_low = predict_churn(pipe, sample_customer, threshold=0.01)
    assert res_low["predicted_class"] == 1
    assert res_low["predicted_label"] == "Will Churn"
    assert "churn_probability" in res_low
