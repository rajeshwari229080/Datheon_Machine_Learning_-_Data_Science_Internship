"""Automated unit and integration tests for Task 4: NLP Sentiment Classification Pipeline."""

import numpy as np
import pytest

from src.sentiment.sentiment_pipeline import (
    build_tfidf_pipeline,
    clean_text,
    predict_sentiment,
    prepare_sentiment_data,
    train_and_evaluate_sentiment_models,
)


def test_clean_text_html_and_punctuation():
    """Verify HTML stripping, lowercasing, punctuation normalization, and whitespace collapsing."""
    raw = "  <b>Fantastic</b> product!!! Arrived in <br/> 2 days. :)  "
    cleaned = clean_text(raw)
    assert "fantastic" in cleaned
    assert "<b>" not in cleaned
    assert "<br/>" not in cleaned
    assert "!" not in cleaned
    assert ":" not in cleaned
    assert "  " not in cleaned


def test_clean_text_null_and_edge_cases():
    """Verify clean handling of None, NaN, and empty strings."""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("    ") == ""


def test_prepare_sentiment_data_split():
    """Verify sampling and stratified train/test split."""
    X_train, X_test, y_train, y_test, df = prepare_sentiment_data(
        sample_size=1000, random_state=42, test_size=0.2
    )
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert len(y_train) == 800
    assert len(y_test) == 200
    assert set(np.unique(y_test)).issubset({0, 1})


def test_predict_sentiment_inference():
    """Verify prediction helper on clear positive and negative sentences."""
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test, _ = prepare_sentiment_data(
        sample_size=1000, random_state=42
    )
    pipe = build_tfidf_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipe.fit(X_train, y_train)

    # Positive sample
    pos_res = predict_sentiment(pipe, "Absolutely fantastic, loved everything about it!")
    assert pos_res["predicted_sentiment"] in ["Positive", "Negative"]
    assert "probabilities" in pos_res
    assert "calibration_note" in pos_res
    assert pytest.approx(pos_res["probabilities"]["positive"] + pos_res["probabilities"]["negative"], 0.01) == 1.0

    # Empty sample
    empty_res = predict_sentiment(pipe, "   ")
    assert empty_res["predicted_sentiment"] == "Neutral/Indeterminate"
