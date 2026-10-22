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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.data_loader import load_dataset
from src.common.evaluation import evaluate_classification, save_metrics_json

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

    if IDENTIFIER_COL in df.columns:
        df.drop(columns=[IDENTIFIER_COL], inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = int(df["TotalCharges"].isnull().sum())
    if nan_count > 0:
        logger.info(
            f"Safely coerced TotalCharges to numeric: detected {nan_count} blank/null values to be imputed."
        )

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


def build_churn_pipeline(classifier: Any) -> Pipeline:
    """Enclose ColumnTransformer and churn classifier into a single Scikit-learn Pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_churn_preprocessor()),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate_churn_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """Train Random Forest and XGBoost with class imbalance mitigation and evaluate test performance."""
    # Compute imbalance ratio for XGBoost scale_pos_weight
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    logger.info(
        f"Training set imbalance ratio: {n_neg} negative vs {n_pos} positive (scale_pos_weight={scale_pos_weight:.2f})"
    )

    # 1. Random Forest with balanced class weighting
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_pipe = build_churn_pipeline(rf_clf)

    # 2. XGBoost with scale_pos_weight or documented fallback
    try:
        from xgboost import XGBClassifier

        xgb_clf = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        xgb_name = "xgboost"
        logger.info("Instantiated XGBClassifier with scale_pos_weight.")
    except Exception as e:
        logger.warning(
            f"XGBoost unavailable ({e}). Employing documented fallback: GradientBoostingClassifier."
        )
        xgb_clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
        )
        xgb_name = "gradient_boosting_fallback"

    xgb_pipe = build_churn_pipeline(xgb_clf)

    models: Dict[str, Pipeline] = {
        "random_forest": rf_pipe,
        xgb_name: xgb_pipe,
    }

    results: Dict[str, Dict[str, Any]] = {}
    probabilities: Dict[str, np.ndarray] = {}

    y_test_arr = y_test.to_numpy()

    for name, pipe in models.items():
        logger.info(f"Fitting {name} pipeline on {len(X_train)} instances...")
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_test)
        y_pred = pipe.predict(X_test)
        probabilities[name] = y_prob[:, 1]

        metrics = evaluate_classification(
            y_true=y_test_arr,
            y_pred=y_pred,
            y_prob=y_prob,
            labels=[0, 1],
            target_names=["Retain (0)", "Churn (1)"],
        )
        results[name] = metrics
        logger.info(
            f"{name} -> Test ROC-AUC: {metrics.get('roc_auc', 0):.4f}, "
            f"Recall: {metrics['macro_recall']:.4f}, F1: {metrics['macro_f1']:.4f}"
        )

    return models, results, probabilities


def predict_churn(
    pipeline: Pipeline,
    customer_data: Dict[str, Any],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Predict customer churn with customizable decision threshold.

    Allows adjusting decision threshold to manage precision-recall tradeoffs
    in customer retention interventions.
    """
    df_single = pd.DataFrame([customer_data])
    prob_churn = float(pipeline.predict_proba(df_single)[0, 1])
    predicted_class = int(prob_churn >= threshold)
    label = "Will Churn" if predicted_class == 1 else "Will Retain"

    return {
        "churn_probability": round(prob_churn, 4),
        "decision_threshold": threshold,
        "predicted_class": predicted_class,
        "predicted_label": label,
        "retention_recommendation": (
            "High churn risk: initiate proactive retention incentive."
            if predicted_class == 1
            else "Low churn risk: maintain standard service."
        ),
    }
