"""Resilient dataset loading and caching module.

Provides local caching, schema validation, transparent network error handling,
and documented fallback management across all four internship tasks.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(DATA_RAW_DIR, exist_ok=True)

DATA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "task1_iris": {
        "title": "Iris Multi-Class Benchmark Dataset",
        "primary_url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "fallback_url": None,
        "filename": "iris.csv",
        "expected_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        "min_rows": 100,
        "license": "BSD-3-Clause",
        "provenance": "Ronald Fisher (1936), seaborn benchmark archive",
    },
    "task2_housing": {
        "title": "California Housing Prices",
        "primary_url": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        "fallback_url": None,
        "filename": "housing.csv",
        "expected_columns": [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "median_house_value", "ocean_proximity",
        ],
        "min_rows": 15000,
        "license": "Public Domain / CC0",
        "provenance": "1990 California Census, Hands-On Machine Learning (Aurélien Géron)",
    },
    "task3_churn": {
        "title": "IBM Telco Customer Churn",
        "primary_url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "fallback_url": None,
        "filename": "telco_churn.csv",
        "expected_columns": ["customerID", "gender", "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"],
        "min_rows": 5000,
        "license": "Apache-2.0 / IBM Community Data",
        "provenance": "IBM Cognos Analytics Sample Dataset",
    },
    "task4_sentiment": {
        "title": "Customer Reviews Sentiment Dataset",
        "primary_url": "https://raw.githubusercontent.com/datasets/sample-reviews/main/sentiment_reviews.csv",
        "fallback_url": "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv",
        "filename": "customer_reviews.csv",
        "expected_columns": ["reviewText", "Positive"],
        "min_rows": 1000,
        "license": "Public Open Source (Amazon product review corpus, PyCaret benchmark)",
        "provenance": "Curated Amazon customer reviews corpus with binary sentiment labels (1=Positive, 0=Negative)",
    },
}


def download_file(url: str, destination: str, timeout: int = 15) -> bool:
    """Download a file via HTTP/HTTPS with proper user-agent headers and error trapping."""
    headers = {"User-Agent": "Datheon-ML-Suite/1.0 (Python urllib)"}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                with open(destination, "wb") as f_out:
                    f_out.write(response.read())
                logger.info(f"Successfully downloaded: {url} -> {destination}")
                return True
            logger.warning(f"Unexpected HTTP status {response.status} for {url}")
            return False
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP Error {e.code} ({e.reason}) while downloading: {url}")
        return False
    except Exception as e:
        logger.warning(f"Network error downloading {url}: {e}")
        return False


def load_dataset(task_name: str, force_download: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate dataset by task key.

    Uses local cache if available; otherwise attempts primary URL followed by
    documented fallback if primary is inaccessible.
    """
    if task_name not in DATA_REGISTRY:
        raise ValueError(f"Unknown task key '{task_name}'. Available: {list(DATA_REGISTRY.keys())}")

    meta = DATA_REGISTRY[task_name].copy()
    cache_path = os.path.join(DATA_RAW_DIR, meta["filename"])
    meta["cache_path"] = cache_path
    meta["source_used"] = "cache"

    # Step 1: Use cache if present and valid
    if os.path.exists(cache_path) and not force_download and os.path.getsize(cache_path) > 100:
        logger.info(f"Loading cached dataset for '{task_name}' from {cache_path}")
        df = pd.read_csv(cache_path)
        meta["source_used"] = f"local_cache:{cache_path}"
    else:
        # Step 2: Attempt primary download
        primary_url = meta["primary_url"]
        logger.info(f"Fetching primary dataset for '{task_name}' from {primary_url}...")
        download_success = download_file(primary_url, cache_path)

        if download_success:
            meta["source_used"] = f"primary_url:{primary_url}"
        else:
            # Step 3: Handle documented fallback
            fallback_url = meta.get("fallback_url")
            if fallback_url:
                logger.warning(
                    f"Primary dataset source for '{task_name}' was unavailable. "
                    f"Switching to documented fallback: {fallback_url}"
                )
                fallback_success = download_file(fallback_url, cache_path)
                if not fallback_success:
                    raise RuntimeError(
                        f"Both primary URL ({primary_url}) and fallback URL ({fallback_url}) "
                        f"failed for task '{task_name}'."
                    )
                meta["source_used"] = f"fallback_url:{fallback_url}"
            else:
                raise RuntimeError(
                    f"Primary dataset URL failed for '{task_name}' and no fallback was specified: {primary_url}"
                )

        df = pd.read_csv(cache_path)

    # Step 4: Validate schema and minimum row constraints
    if len(df) < meta["min_rows"]:
        raise ValueError(
            f"Dataset '{task_name}' contains {len(df)} rows, which is below the minimum threshold of {meta['min_rows']}."
        )

    for col in meta["expected_columns"]:
        if col not in df.columns:
            raise ValueError(
                f"Dataset '{task_name}' is missing expected column: '{col}'. Present columns: {list(df.columns)}"
            )

    meta["total_rows"] = int(len(df))
    meta["total_columns"] = int(len(df.columns))
    meta["column_list"] = list(df.columns)
    logger.info(
        f"Dataset '{task_name}' validated: {meta['total_rows']} rows, {meta['total_columns']} cols."
    )
    return df, meta


def get_metadata(task_name: str) -> Dict[str, Any]:
    """Retrieve metadata registry entry for a specific task."""
    if task_name not in DATA_REGISTRY:
        raise ValueError(f"Task '{task_name}' not found.")
    return DATA_REGISTRY[task_name].copy()
