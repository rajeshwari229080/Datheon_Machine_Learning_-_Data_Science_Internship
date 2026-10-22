"""Iris Multi-Class Benchmark Classification Pipeline.

Demonstrates rigorous data splitting, leakage prevention via Scikit-learn pipelines,
model comparison between Logistic Regression and Decision Tree, and 2D boundary projection.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.common.data_loader import load_dataset
from src.common.evaluation import evaluate_classification, save_metrics_json
from src.common.visualization import plot_confusion_matrix, plot_model_comparison

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAME: str = "species"
CLASS_NAMES: List[str] = ["setosa", "versicolor", "virginica"]
LABEL_MAPPING: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INV_LABEL_MAPPING: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def prepare_iris_data(
    random_state: int = 42, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load, validate, and split Iris dataset into stratified train and test partitions."""
    df, meta = load_dataset("task1_iris")

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df[TARGET_NAME].map(LABEL_MAPPING).to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Iris data split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples (stratified)."
    )
    return X_train, X_test, y_train, y_test, df


def build_logistic_regression_pipeline(
    random_state: int = 42, max_iter: int = 1000, C: float = 1.0
) -> Pipeline:
    """Build Logistic Regression pipeline with StandardScaler.

    StandardScaler is fitted solely on the training data within the pipeline,
    preventing data leakage.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=random_state, max_iter=max_iter, C=C)),
        ]
    )


def build_decision_tree_pipeline(
    random_state: int = 42, max_depth: Optional[int] = 4, min_samples_split: int = 2
) -> Pipeline:
    """Build Decision Tree classification pipeline."""
    return Pipeline(
        [
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=random_state,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                ),
            )
        ]
    )


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """Train both Logistic Regression and Decision Tree pipelines and evaluate on test set."""
    models: Dict[str, Pipeline] = {
        "logistic_regression": build_logistic_regression_pipeline(),
        "decision_tree": build_decision_tree_pipeline(),
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        logger.info(f"Training {name} pipeline on {X_train.shape[0]} samples...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = evaluate_classification(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
        )
        results[name] = metrics
        logger.info(
            f"{name} Results -> Accuracy: {metrics['accuracy']:.4f}, "
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )

    return models, results


def predict_species(
    model: Pipeline,
    features: Union[List[float], Tuple[float, float, float, float], Dict[str, float]],
) -> Dict[str, Any]:
    """Reusable prediction function for Iris species.

    Accepts 4 features [sepal_length, sepal_width, petal_length, petal_width] or dict.
    Returns predicted species, class integer, and probability distribution.
    """
    if isinstance(features, dict):
        missing = [f for f in FEATURE_NAMES if f not in features]
        if missing:
            raise ValueError(f"Missing required feature keys: {missing}")
        arr = np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float64)
    else:
        if len(features) != 4:
            raise ValueError(
                f"Expected 4 feature values [sepal_length, sepal_width, petal_length, petal_width], got {len(features)}"
            )
        arr = np.array([features], dtype=np.float64)

    pred_idx = int(model.predict(arr)[0])
    species_name = INV_LABEL_MAPPING[pred_idx]

    probs: Dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(arr)[0]
        probs = {name: float(prob_arr[i]) for i, name in enumerate(CLASS_NAMES)}

    return {
        "predicted_class_index": pred_idx,
        "predicted_species": species_name,
        "probabilities": probs,
    }
