"""IBM Telco Customer Churn Classification Pipeline.

Handles numeric coercion of TotalCharges, identifier removal, stratified partitioning,
class imbalance weighting, and decision threshold calibration across Random Forest and XGBoost.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.data_loader import load_dataset

logger = logging.getLogger(__name__)

TARGET_COL: str = "Churn"
IDENTIFIER_COL: str = "customerID"

NUMERICAL_COLS: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def clean_churn_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco churn dataset.

    - Removes customerID identifier.
    - Coerces TotalCharges to float (converting invalid blanks to NaN).
    - Maps target Churn ('Yes' -> 1, 'No' -> 0).
    """
    df = df_raw.copy()

    # Drop identifier
    if IDENTIFIER_COL in df.columns:
        df.drop(columns=[IDENTIFIER_COL], inplace=True)

    # Safe coercion of TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = int(df["TotalCharges"].isnull().sum())
    if nan_count > 0:
        logger.info(
            f"Safely coerced TotalCharges to numeric: detected {nan_count} blank/null values to be imputed."
        )

    # Encode binary target if string
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(np.int64)

    return df


def prepare_churn_data(
    random_state: int = 42, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load, clean, and partition Telco Churn dataset into stratified train and test sets."""
    df_raw, meta = load_dataset("task3_churn")
    df_clean = clean_churn_dataset(df_raw)

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Telco churn partitioned: {len(X_train)} train rows, {len(X_test)} test rows (stratified)."
    )
    return X_train, X_test, y_train, y_test, df_clean


def build_churn_preprocessor() -> ColumnTransformer:
    """Construct ColumnTransformer with numeric median imputation/scaling and categorical OneHotEncoding."""
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERICAL_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor
