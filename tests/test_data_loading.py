"""Unit tests for resilient data loading and evaluation utilities."""

import os
import tempfile
import numpy as np
import pytest

from src.common.data_loader import DATA_REGISTRY, download_file, load_dataset
from src.common.evaluation import evaluate_classification, evaluate_regression, save_metrics_json


def test_registry_contains_all_tasks():
    """Verify that all four required internship tasks are registered."""
    expected_tasks = ["task1_iris", "task2_housing", "task3_churn", "task4_sentiment"]
    for task in expected_tasks:
        assert task in DATA_REGISTRY, f"Task {task} missing from DATA_REGISTRY"
        meta = DATA_REGISTRY[task]
        assert "primary_url" in meta
        assert "filename" in meta
        assert "expected_columns" in meta
        assert meta["min_rows"] > 0


def test_download_file_nonexistent_url():
    """Verify graceful handling when a URL fails."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        success = download_file("https://httpbin.org/status/404", tmp_path, timeout=5)
        assert success is False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_iris_dataset():
    """Verify live loading, schema validation, and row count of Iris benchmark."""
    df, meta = load_dataset("task1_iris")
    assert len(df) == 150
    assert "species" in df.columns
    assert set(df["species"].unique()) == {"setosa", "versicolor", "virginica"}
    assert meta["total_rows"] == 150


def test_evaluate_classification_metrics():
    """Verify multi-class classification metric calculations."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])  # 5/6 correct
    metrics = evaluate_classification(y_true, y_pred)

    assert pytest.approx(metrics["accuracy"], 0.01) == 5 / 6
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics
    assert len(metrics["confusion_matrix"]) == 3


def test_evaluate_regression_metrics():
    """Verify regression metric calculations."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 32.0])  # errors: +1, -1, +2 -> MSE = 2.0
    metrics = evaluate_regression(y_true, y_pred)

    assert pytest.approx(metrics["mse"], 0.01) == (1 + 1 + 4) / 3
    assert pytest.approx(metrics["rmse"], 0.01) == np.sqrt(2.0)
    assert pytest.approx(metrics["mae"], 0.01) == 4 / 3
    assert metrics["r2"] > 0.9


def test_save_metrics_json():
    """Verify JSON export of metrics with numpy types."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "metrics.json")
        sample_metrics = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14159),
            "array_val": np.array([1, 2, 3]),
            "nested": {"nested_float": np.float32(0.95)},
        }
        save_metrics_json(sample_metrics, json_path)
        assert os.path.exists(json_path)
        assert os.path.getsize(json_path) > 10
