"""Automated unit and integration tests for Task 1: Iris Classification Pipeline."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.classification.iris_pipeline import (
    CLASS_NAMES,
    FEATURE_NAMES,
    build_decision_tree_pipeline,
    build_logistic_regression_pipeline,
    plot_two_feature_decision_boundary,
    predict_species,
    prepare_iris_data,
    train_and_evaluate_models,
)


def test_prepare_iris_data_split_and_stratification():
    """Verify 80:20 split and exact class stratification."""
    X_train, X_test, y_train, y_test, df = prepare_iris_data(random_state=42, test_size=0.2)

    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert len(y_train) == 120
    assert len(y_test) == 30

    # Stratification check: 10 of each class in test set
    counts = np.bincount(y_test)
    assert list(counts) == [10, 10, 10]


def test_leakage_prevention_in_pipeline():
    """Verify that StandardScaler is embedded within the pipeline and unfitted beforehand."""
    pipe = build_logistic_regression_pipeline()
    scaler = pipe.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert not hasattr(scaler, "mean_"), "StandardScaler must not be fitted before pipeline training"


def test_train_and_evaluate_models():
    """Verify that both Logistic Regression and Decision Tree train and achieve >0.85 accuracy."""
    X_train, X_test, y_train, y_test, _ = prepare_iris_data(random_state=42, test_size=0.2)
    models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    assert "logistic_regression" in results
    assert "decision_tree" in results

    lr_res = results["logistic_regression"]
    dt_res = results["decision_tree"]

    assert lr_res["accuracy"] >= 0.90
    assert dt_res["accuracy"] >= 0.85
    assert len(lr_res["confusion_matrix"]) == 3
    assert "classification_report" in lr_res


def test_predict_species_helper():
    """Verify inference helper accepts list, dict, and handles invalid inputs gracefully."""
    X_train, X_test, y_train, y_test, _ = prepare_iris_data()
    pipe = build_logistic_regression_pipeline()
    pipe.fit(X_train, y_train)

    # List input
    res_list = predict_species(pipe, [5.1, 3.5, 1.4, 0.2])
    assert res_list["predicted_species"] == "setosa"
    assert "probabilities" in res_list
    assert pytest.approx(sum(res_list["probabilities"].values()), 0.01) == 1.0

    # Dict input
    sample_dict = {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3,
    }
    res_dict = predict_species(pipe, sample_dict)
    assert res_dict["predicted_species"] in CLASS_NAMES

    # Invalid input (missing dimensions)
    with pytest.raises(ValueError):
        predict_species(pipe, [5.1, 3.5])


def test_plot_two_feature_decision_boundary():
    """Verify 2D decision boundary visualization creates a valid figure."""
    X_train, _, y_train, _, _ = prepare_iris_data()
    fig = plot_two_feature_decision_boundary(
        X=X_train, y=y_train, model_type="logistic_regression", feature_indices=(0, 2)
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
