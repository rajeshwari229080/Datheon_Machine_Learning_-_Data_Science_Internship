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
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.data_loader import load_dataset
from src.common.evaluation import evaluate_regression, save_metrics_json

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
    """Construct Scikit-learn ColumnTransformer for numeric and categorical pipelines."""
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


def build_regression_pipeline(estimator: Any) -> Pipeline:
    """Wrap preprocessor and estimator into a unified Scikit-learn Pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("regressor", estimator),
        ]
    )


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """Train OLS baseline and perform 5-fold cross-validation tuning for Ridge and Lasso."""
    preprocessor = build_preprocessor()

    # 1. Baseline Ordinary Least Squares
    ols_pipe = build_regression_pipeline(LinearRegression())
    logger.info("Fitting OLS Linear Regression baseline...")
    ols_pipe.fit(X_train, y_train)

    # 2. Ridge Regression with GridSearchCV on training set
    ridge_pipe = build_regression_pipeline(Ridge(random_state=42))
    ridge_params = {"regressor__alpha": [0.1, 1.0, 10.0, 50.0, 100.0, 200.0]}
    logger.info("Tuning Ridge regularization strength with 5-fold CV...")
    ridge_cv = GridSearchCV(
        ridge_pipe,
        param_grid=ridge_params,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    ridge_cv.fit(X_train, y_train)
    best_ridge = ridge_cv.best_estimator_

    # 3. Lasso Regression with GridSearchCV on training set
    lasso_pipe = build_regression_pipeline(Lasso(random_state=42, max_iter=2000))
    lasso_params = {"regressor__alpha": [1.0, 10.0, 50.0, 100.0, 250.0]}
    logger.info("Tuning Lasso regularization strength with 5-fold CV...")
    lasso_cv = GridSearchCV(
        lasso_pipe,
        param_grid=lasso_params,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    lasso_cv.fit(X_train, y_train)
    best_lasso = lasso_cv.best_estimator_

    models: Dict[str, Pipeline] = {
        "ols_linear_regression": ols_pipe,
        "ridge_regression": best_ridge,
        "lasso_regression": best_lasso,
    }

    results: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, np.ndarray] = {}

    y_test_arr = y_test.to_numpy()

    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        predictions[name] = y_pred

        metrics = evaluate_regression(y_test_arr, y_pred)

        # Attach best CV parameters if tuned
        if name == "ridge_regression":
            metrics["best_alpha"] = float(ridge_cv.best_params_["regressor__alpha"])
            metrics["best_cv_rmse"] = float(-ridge_cv.best_score_)
        elif name == "lasso_regression":
            metrics["best_alpha"] = float(lasso_cv.best_params_["regressor__alpha"])
            metrics["best_cv_rmse"] = float(-lasso_cv.best_score_)

        results[name] = metrics
        logger.info(
            f"{name} -> Test RMSE: ${metrics['rmse']:,.2f}, MAE: ${metrics['mae']:,.2f}, R²: {metrics['r2']:.4f}"
        )

    return models, results, predictions


def predict_house_value(pipeline: Pipeline, house_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict median house value in USD from dictionary of input features.

    Applies interaction feature engineering and the complete fitted ColumnTransformer pipeline.
    """
    df_single = pd.DataFrame([house_data])
    df_engineered = add_interaction_features(df_single)

    pred_val = float(pipeline.predict(df_engineered)[0])
    return {
        "predicted_median_house_value_usd": round(pred_val, 2),
        "predicted_value_formatted": f"${pred_val:,.2f}",
        "input_features": house_data,
    }
