"""California Housing Price Regression Pipeline.

Demonstrates leakage-free preprocessing, feature interaction engineering,
multicollinearity diagnostics (VIF), and regularized regression comparison (OLS, Ridge, Lasso).
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

TARGET_COL: str = "median_house_value"
RAW_NUMERIC_COLS: List[str] = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_COLS: List[str] = ["ocean_proximity"]
ENGINEERED_COLS: List[str] = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]
ALL_NUMERIC_COLS: List[str] = RAW_NUMERIC_COLS + ENGINEERED_COLS


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer meaningful domain ratio interactions from raw housing attributes."""
    df_feat = df.copy()
    # Avoid zero division with safe epsilon
    df_feat["rooms_per_household"] = df_feat["total_rooms"] / np.maximum(df_feat["households"], 1)
    df_feat["bedrooms_per_room"] = df_feat["total_bedrooms"] / np.maximum(df_feat["total_rooms"], 1)
    df_feat["population_per_household"] = df_feat["population"] / np.maximum(df_feat["households"], 1)
    return df_feat


def prepare_housing_data(
    random_state: int = 42, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load, engineer features, and partition California housing dataset.

    Splits the dataset BEFORE fitting imputers, scalers, or encoders to eliminate leakage.
    """
    df_raw, meta = load_dataset("task2_housing")
    df_engineered = add_interaction_features(df_raw)

    feature_cols = ALL_NUMERIC_COLS + CATEGORICAL_COLS
    X = df_engineered[feature_cols]
    y = df_engineered[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"California housing data partitioned: {len(X_train)} train rows, {len(X_test)} test rows."
    )
    return X_train, X_test, y_train, y_test, df_engineered


def build_preprocessor() -> ColumnTransformer:
    """Construct Scikit-learn ColumnTransformer for numeric and categorical pipelines.

    Numeric: SimpleImputer(median) -> StandardScaler()
    Categorical: OneHotEncoder(handle_unknown='ignore')
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, ALL_NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor
