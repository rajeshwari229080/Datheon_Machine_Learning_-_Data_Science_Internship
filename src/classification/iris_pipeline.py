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


def plot_two_feature_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logistic_regression",
    feature_indices: Tuple[int, int] = (0, 2),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Train a legitimate 2-feature model and visualize its 2D decision boundary.

    Limitation Note: This boundary is trained and plotted strictly in the 2D subspace of
    (feature_indices[0], feature_indices[1]). It demonstrates the model's geometric decision
    regions in 2D and is NOT a projection of the full 4D decision hyperspace.
    """
    f1_idx, f2_idx = feature_indices
    f1_name = FEATURE_NAMES[f1_idx]
    f2_name = FEATURE_NAMES[f2_idx]

    X_2d = X[:, [f1_idx, f2_idx]]

    # Train dedicated 2D model
    if model_type == "logistic_regression":
        model_2d = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        display_name = "Logistic Regression"
    else:
        model_2d = DecisionTreeClassifier(random_state=42, max_depth=4)
        display_name = "Decision Tree"

    model_2d.fit(X_2d, y)

    # Generate dense 2D meshgrid
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_2d.predict(grid_points)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Decision regions
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis", levels=[-0.5, 0.5, 1.5, 2.5])
    ax.contour(xx, yy, Z, colors="#1F2937", linewidths=0.8, alpha=0.5, levels=[0.5, 1.5])

    # Scatter actual observations
    colors = ["#4338CA", "#059669", "#D97706"]
    markers = ["o", "s", "^"]
    for idx, cname in enumerate(CLASS_NAMES):
        mask = y == idx
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=colors[idx],
            marker=markers[idx],
            edgecolor="#111827",
            linewidth=0.7,
            s=55,
            label=f"{cname.capitalize()} (Actual)",
        )

    ax.set_title(
        f"{display_name}: 2-Feature Decision Surface\n"
        f"Features: {f1_name.replace('_', ' ').title()} vs {f2_name.replace('_', ' ').title()}",
        pad=12,
        weight="bold",
        color="#111827",
    )
    ax.set_xlabel(f"{f1_name.replace('_', ' ').title()} (cm)", labelpad=8)
    ax.set_ylabel(f"{f2_name.replace('_', ' ').title()} (cm)", labelpad=8)
    ax.legend(loc="upper left", frameon=True, facecolor="#F9FAFB")

    # Explanatory limitation footnote
    fig.text(
        0.5,
        -0.02,
        "Limitation: Decision surface is produced by a legitimate 2-feature model for geometric visualization.\n"
        "It does not represent the full 4-dimensional hyper-surface of the production 4-feature classifier.",
        ha="center",
        fontsize=8,
        color="#6B7280",
        style="italic",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


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


def run_task1_pipeline(output_dir: str = "reports/task1") -> Dict[str, Any]:
    """Execute complete Task 1 pipeline, generate figures, and export metrics.json."""
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, df = prepare_iris_data()

    models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # 1. Confusion Matrix Plots
    for name, metrics in results.items():
        cm_path = os.path.join(output_dir, f"confusion_matrix_{name}.png")
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=CLASS_NAMES,
            title=f"Iris Confusion Matrix: {name.replace('_', ' ').title()}",
            save_path=cm_path,
        )
        plt.close("all")

    # 2. Model Comparison Bar Chart
    comp_path = os.path.join(output_dir, "model_comparison.png")
    plot_model_comparison(
        models_dict={
            "Logistic Regression": results["logistic_regression"]["accuracy"],
            "Decision Tree": results["decision_tree"]["accuracy"],
        },
        metric_name="Accuracy",
        title="Model Accuracy Comparison: Iris Multi-Class Benchmark",
        save_path=comp_path,
    )
    plt.close("all")

    # 3. 2-Feature Decision Boundaries
    for mtype in ["logistic_regression", "decision_tree"]:
        bd_path = os.path.join(output_dir, f"decision_boundary_{mtype}.png")
        plot_two_feature_decision_boundary(
            X=X_train,
            y=y_train,
            model_type=mtype,
            feature_indices=(0, 2),  # sepal_length vs petal_length
            save_path=bd_path,
        )
        plt.close("all")

    # 4. Save metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_metrics_json(results, metrics_path)

    # 5. Prediction verification
    sample_feat = [5.1, 3.5, 1.4, 0.2]
    pred_res = predict_species(models["logistic_regression"], sample_feat)
    logger.info(f"Sample prediction for {sample_feat}: {pred_res['predicted_species']}")

    return {
        "models": models,
        "results": results,
        "output_dir": output_dir,
    }
