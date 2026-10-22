"""Publication-quality visualization module for data science and machine learning.

Enforces analytical laboratory styling, accessible color palettes, and
reproducible high-resolution figure exports.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set professional analytical style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update(
    {
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot an annotated confusion matrix with counts and percentages."""
    cm_arr = np.array(cm, dtype=int)
    cm_norm = cm_arr.astype("float") / np.maximum(cm_arr.sum(axis=1)[:, np.newaxis], 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    annot = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            annot[i, j] = f"{cm_arr[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=1.0,
        linecolor="#E5E7EB",
    )
    ax.set_title(title, pad=14, weight="bold", color="#111827")
    ax.set_xlabel("Predicted Class", labelpad=10, weight="medium", color="#374151")
    ax.set_ylabel("True Class", labelpad=10, weight="medium", color="#374151")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_model_comparison(
    models_dict: Dict[str, float],
    metric_name: str = "Accuracy",
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparative bar chart across models with explicit value callouts."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(models_dict.keys())
    values = list(models_dict.values())

    colors = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B"][: len(names)]
    bars = ax.bar(names, values, color=colors, width=0.55, edgecolor="#1F2937", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
            color="#1F2937",
        )

    ax.set_title(title, pad=12, weight="bold", color="#111827")
    ax.set_ylabel(metric_name, labelpad=8, weight="medium", color="#374151")
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1.0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_roc_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "Receiver Operating Characteristic (ROC)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot overlaid ROC curves with AUC annotations."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    palette = ["#4F46E5", "#06B6D4", "#F59E0B", "#10B981"]

    for idx, (name, (fpr, tpr, auc_val)) in enumerate(curves.items()):
        color = palette[idx % len(palette)]
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_val:.4f})",
            color=color,
            linewidth=2.2,
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", alpha=0.5, linewidth=1.2)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=8, weight="medium")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", labelpad=8, weight="medium")
    ax.set_title(title, pad=12, weight="bold", color="#111827")
    ax.legend(loc="lower right", frameon=True, facecolor="#F9FAFB")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate 3-panel regression diagnostics: Actual vs Predicted, Residuals, and Histogram."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.35, color="#4F46E5", edgecolors="none", s=20)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal Fit")
    axes[0].set_title(f"{model_name}: Actual vs Predicted", weight="bold")
    axes[0].set_xlabel("Actual Value")
    axes[0].set_ylabel("Predicted Value")
    axes[0].legend(loc="upper left")

    # Panel 2: Residual vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.35, color="#06B6D4", edgecolors="none", s=20)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted", weight="bold")
    axes[1].set_xlabel("Predicted Value")
    axes[1].set_ylabel("Residual (Actual - Pred)")

    # Panel 3: Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[2], color="#10B981", bins=30)
    axes[2].axvline(0, color="r", linestyle="--", linewidth=1.5)
    axes[2].set_title("Residual Error Distribution", weight="bold")
    axes[2].set_xlabel("Residual Error")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_feature_importances(
    features: List[str],
    importances: np.ndarray,
    top_n: int = 10,
    title: str = "Top Feature Importances",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot sorted horizontal bar chart for feature importances."""
    imp = np.array(importances)
    indices = np.argsort(imp)[::-1][:top_n]
    sorted_features = [features[i] for i in reversed(indices)]
    sorted_values = [imp[i] for i in reversed(indices)]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    bars = ax.barh(sorted_features, sorted_values, color="#4F46E5", edgecolor="#1F2937", height=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.4f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            weight="bold",
            color="#1F2937",
        )

    ax.set_title(title, pad=12, weight="bold", color="#111827")
    ax.set_xlabel("Importance Score", labelpad=8, weight="medium")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    return fig
