"""NLP Sentiment Classification Pipeline using TF-IDF.

Implements text cleaning, leakage-free TF-IDF vectorization inside Scikit-learn pipelines,
model comparison between Multinomial Naive Bayes and Logistic Regression,
and custom inference with probability calibration notes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.common.data_loader import load_dataset
from src.common.evaluation import evaluate_classification, save_metrics_json
from src.common.visualization import plot_confusion_matrix, plot_model_comparison

logger = logging.getLogger(__name__)

TEXT_COL: str = "reviewText"
LABEL_COL: str = "Positive"
CLASS_NAMES: List[str] = ["Negative (0)", "Positive (1)"]


def clean_text(
    text: Union[str, float, None],
    remove_stopwords: bool = False,
    stopwords_set: Optional[set] = None,
) -> str:
    """Sanitize and normalize raw review text.

    Steps:
    1. Handle None / float NaNs cleanly.
    2. Strip HTML tags (e.g. <br />, <p>).
    3. Lowercase text.
    4. Replace punctuation and non-alphanumeric chars with whitespace.
    5. Collapse multiple whitespace and strip edges.

    Note on Stopwords:
    Standard stopword lists remove negation particles like 'not', 'no', 'never',
    which can completely invert sentiment polarity. Hence, stopword removal is
    made optional with detailed documentation.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    text_str = str(text)

    # 1. Strip HTML tags
    text_clean = re.sub(r"<[^>]+>", " ", text_str)

    # 2. Lowercase
    text_clean = text_clean.lower()

    # 3. Retain alphanumeric characters and spaces
    text_clean = re.sub(r"[^a-z0-9\s]", " ", text_clean)

    # 4. Collapse whitespace
    tokens = text_clean.split()

    # 5. Optional stopword removal
    if remove_stopwords and stopwords_set:
        tokens = [t for t in tokens if t not in stopwords_set]

    return " ".join(tokens)


def prepare_sentiment_data(
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, pd.DataFrame]:
    """Load, inspect, sanitize, and partition customer review dataset.

    Splits the raw text *before* fitting TF-IDF vocabulary to strictly prevent data leakage.
    """
    df_raw, meta = load_dataset("task4_sentiment")
    logger.info(
        f"Sentiment data loaded from {meta['source_used']}: {len(df_raw)} total reviews."
    )

    if TEXT_COL not in df_raw.columns or LABEL_COL not in df_raw.columns:
        raise ValueError(
            f"Expected columns ['{TEXT_COL}', '{LABEL_COL}'], but found: {list(df_raw.columns)}"
        )

    df = df_raw[[TEXT_COL, LABEL_COL]].dropna().copy()
    df[LABEL_COL] = df[LABEL_COL].astype(np.int64)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info(f"Subsampled to {sample_size} reviews for efficient, CPU-friendly evaluation.")

    logger.info("Sanitizing review text (removing HTML tags, normalizing whitespace)...")
    cleaned_reviews = [clean_text(r) for r in df[TEXT_COL]]
    df["cleaned_text"] = cleaned_reviews

    valid_mask = df["cleaned_text"].str.len() > 0
    df = df[valid_mask].reset_index(drop=True)

    X_texts = df["cleaned_text"].tolist()
    y_labels = df[LABEL_COL].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X_texts, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels
    )

    logger.info(
        f"Sentiment data partitioned: {len(X_train)} train texts, {len(X_test)} test texts (stratified)."
    )
    return X_train, X_test, y_train, y_test, df


def build_tfidf_pipeline(classifier: Any, max_features: int = 5000) -> Pipeline:
    """Enclose TfidfVectorizer and classifier into a single Scikit-learn Pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate_sentiment_models(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    max_features: int = 5000,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """Train Multinomial Naive Bayes and Logistic Regression on text corpus and evaluate on test set."""
    models: Dict[str, Pipeline] = {
        "multinomial_nb": build_tfidf_pipeline(MultinomialNB(alpha=1.0), max_features=max_features),
        "logistic_regression": build_tfidf_pipeline(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            max_features=max_features,
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, pipe in models.items():
        logger.info(f"Training {name} pipeline on {len(X_train)} text documents...")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)

        metrics = evaluate_classification(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            labels=[0, 1],
            target_names=CLASS_NAMES,
        )
        results[name] = metrics
        logger.info(
            f"{name} -> Test Accuracy: {metrics['accuracy']:.4f}, "
            f"Macro F1: {metrics['macro_f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 0):.4f}"
        )

    return models, results


def predict_sentiment(pipeline: Pipeline, raw_text: str) -> Dict[str, Any]:
    """Classify sentiment of custom text string.

    Returns predicted sentiment, numeric score, and a probability calibration caveat.
    """
    cleaned = clean_text(raw_text)
    if not cleaned:
        return {
            "input_text": raw_text,
            "cleaned_text": "",
            "predicted_sentiment": "Neutral/Indeterminate",
            "predicted_class": -1,
            "probabilities": {"negative": 0.5, "positive": 0.5},
            "calibration_note": "Input text was empty after sanitization.",
        }

    probs = pipeline.predict_proba([cleaned])[0]
    pred_class = int(pipeline.predict([cleaned])[0])
    label = "Positive" if pred_class == 1 else "Negative"

    return {
        "input_text": raw_text,
        "cleaned_text": cleaned,
        "predicted_sentiment": label,
        "predicted_class": pred_class,
        "probabilities": {
            "negative": round(float(probs[0]), 4),
            "positive": round(float(probs[1]), 4),
        },
        "calibration_note": (
            "Caution: Raw probabilities reflect heuristic decision function scores; "
            "they are not formally calibrated Bayesian posterior probabilities."
        ),
    }


def run_task4_pipeline(output_dir: str = "reports/task4") -> Dict[str, Any]:
    """Execute complete Task 4 NLP pipeline, generate charts, evaluate samples, and save metrics."""
    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, df = prepare_sentiment_data(sample_size=10000)

    models, results = train_and_evaluate_sentiment_models(X_train, y_train, X_test, y_test)

    # 1. Confusion Matrix Plots
    for name, metrics in results.items():
        cm_path = os.path.join(output_dir, f"confusion_matrix_{name}.png")
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=["Negative", "Positive"],
            title=f"Sentiment Confusion Matrix: {name.replace('_', ' ').title()}",
            save_path=cm_path,
        )
        plt.close("all")

    # 2. Model Comparison Bar Chart
    comp_path = os.path.join(output_dir, "model_comparison.png")
    plot_model_comparison(
        models_dict={
            "Naive Bayes": results["multinomial_nb"]["accuracy"],
            "Logistic Regression": results["logistic_regression"]["accuracy"],
        },
        metric_name="Accuracy",
        title="NLP Sentiment Classification: Model Accuracy Comparison",
        save_path=comp_path,
    )
    plt.close("all")

    # 3. Save metrics.json
    save_metrics_json(results, os.path.join(output_dir, "metrics.json"))

    # 4. Five labeled example inputs for interactive verification
    test_examples = [
        "This product is absolutely wonderful! It exceeded all my expectations and arrived quickly.",
        "Terrible quality, broke after two days of light use. Customer service was unhelpful.",
        "It is decent for the price, but nothing to get overly excited about.",
        "Not bad at all, works much better than the previous brand I purchased.",
        "I wanted to love this, but it completely failed to perform as advertised. Waste of money.",
    ]

    example_results = []
    for text in test_examples:
        res_lr = predict_sentiment(models["logistic_regression"], text)
        res_nb = predict_sentiment(models["multinomial_nb"], text)
        example_results.append(
            {
                "input_text": text,
                "logistic_regression": {
                    "sentiment": res_lr["predicted_sentiment"],
                    "probability_positive": res_lr["probabilities"]["positive"],
                },
                "naive_bayes": {
                    "sentiment": res_nb["predicted_sentiment"],
                    "probability_positive": res_nb["probabilities"]["positive"],
                },
            }
        )

    example_path = os.path.join(output_dir, "example_predictions.json")
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(example_results, f, indent=2)

    logger.info(f"Saved {len(test_examples)} example predictions to {example_path}")
    return {
        "models": models,
        "results": results,
        "examples": example_results,
        "output_dir": output_dir,
    }
