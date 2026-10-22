"""Automated tests for Task 2: California Housing Price Regression Pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.regression.housing_pipeline import (
    ALL_NUMERIC_COLS,
    RAW_NUMERIC_COLS,
    TARGET_COL,
    add_interaction_features,
    build_preprocessor,
    compute_vif_dataframe,
    predict_house_value,
    prepare_housing_data,
    train_and_tune_models,
)


def test_add_interaction_features():
    """Verify calculation of domain interaction features and division safety."""
    df_sample = pd.DataFrame(
        {
            "total_rooms": [100.0, 50.0],
            "total_bedrooms": [20.0, 10.0],
            "population": [200.0, 0.0],
            "households": [50.0, 0.0],  # test zero household safety
        }
    )
    res = add_interaction_features(df_sample)
    assert "rooms_per_household" in res.columns
    assert "bedrooms_per_room" in res.columns
    assert "population_per_household" in res.columns

    # Check division by zero was avoided
    assert np.isfinite(res["rooms_per_household"].iloc[1])
    assert res["rooms_per_household"].iloc[0] == 2.0


def test_prepare_housing_data_split():
    """Verify 80:20 split and feature presence."""
    X_train, X_test, y_train, y_test, df = prepare_housing_data(random_state=42, test_size=0.2)

    total_rows = len(df)
    assert len(X_train) == int(total_rows * 0.8)
    assert len(X_test) == total_rows - len(X_train)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert TARGET_COL not in X_train.columns


def test_preprocessor_transformation():
    """Verify ColumnTransformer transforms training and testing folds correctly."""
    X_train, X_test, y_train, y_test, _ = prepare_housing_data(random_state=42)
    preprocessor = build_preprocessor()

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert not np.isnan(X_train_trans).any(), "No missing values should remain after imputer"


def test_vif_computation():
    """Verify VIF computation produces numeric values for features."""
    _, _, _, _, df = prepare_housing_data()
    vif_df = compute_vif_dataframe(df.head(500), RAW_NUMERIC_COLS)

    assert len(vif_df) == len(RAW_NUMERIC_COLS)
    assert "feature" in vif_df.columns
    assert "vif" in vif_df.columns
    assert vif_df["vif"].notnull().all()


def test_predict_house_value_output():
    """Verify prediction helper on realistic property dictionary."""
    X_train, X_test, y_train, y_test, _ = prepare_housing_data()
    # Fast model for test
    from sklearn.linear_model import Ridge
    from src.regression.housing_pipeline import build_regression_pipeline

    pipe = build_regression_pipeline(Ridge(alpha=10.0))
    pipe.fit(X_train, y_train)

    sample_house = {
        "longitude": -122.25,
        "latitude": 37.85,
        "housing_median_age": 52.0,
        "total_rooms": 1274.0,
        "total_bedrooms": 235.0,
        "population": 558.0,
        "households": 219.0,
        "median_income": 8.0,
        "ocean_proximity": "NEAR BAY",
    }
    pred = predict_house_value(pipe, sample_house)

    assert "predicted_median_house_value_usd" in pred
    assert pred["predicted_median_house_value_usd"] > 10000.0
    assert "$" in pred["predicted_value_formatted"]
