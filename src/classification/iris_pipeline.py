"""Iris Multi-Class Benchmark Classification Pipeline.

Demonstrates rigorous data splitting, leakage prevention via Scikit-learn pipelines,
model comparison between Logistic Regression and Decision Tree, and 2D boundary projection.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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
