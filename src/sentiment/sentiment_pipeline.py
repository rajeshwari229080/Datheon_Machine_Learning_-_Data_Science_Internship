"""NLP Sentiment Classification Pipeline using TF-IDF.

Implements text cleaning, leakage-free TF-IDF vectorization inside Scikit-learn pipelines,
model comparison between Multinomial Naive Bayes and Logistic Regression,
and custom inference with probability calibration notes.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.common.data_loader import load_dataset

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

    # Validate columns
    if TEXT_COL not in df_raw.columns or LABEL_COL not in df_raw.columns:
        raise ValueError(
            f"Expected columns ['{TEXT_COL}', '{LABEL_COL}'], but found: {list(df_raw.columns)}"
        )

    df = df_raw[[TEXT_COL, LABEL_COL]].dropna().copy()
    # Normalize labels to int (0 or 1)
    df[LABEL_COL] = df[LABEL_COL].astype(np.int64)

    # Subsample if requested for CPU friendliness while retaining statistical power
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info(f"Subsampled to {sample_size} reviews for efficient, CPU-friendly evaluation.")

    # Clean reviews
    logger.info("Sanitizing review text (removing HTML tags, normalizing whitespace)...")
    cleaned_reviews = [clean_text(r) for r in df[TEXT_COL]]
    df["cleaned_text"] = cleaned_reviews

    # Remove empty reviews after cleaning
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
    """Enclose TfidfVectorizer and classifier into a single Scikit-learn Pipeline.

    Ensures that vocabulary extraction and IDF weights are computed strictly from
    the training corpus, preventing test-set leakage.
    """
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
