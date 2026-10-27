"""
Baseline evaluation module for Lab 4.
Trains Logistic Regression models with and without derived features.
Records accuracy and F1 metrics to quantify the impact of feature engineering.
"""

import json
import logging
import time
from typing import Any, Dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from lab04.src.transformation import AdultPreprocessor

logger = logging.getLogger(__name__)


def evaluate_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "income",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate LogisticRegression baselines:
    1. Without derived features
    2. With derived features (capital_net, hours_per_age)
    """
    # Prepare binary target vector (1 for >50K, 0 for <=50K)
    y_train = (train_df[target_col].astype(str).str.strip() == ">50K").astype(int)
    y_test = (test_df[target_col].astype(str).str.strip() == ">50K").astype(int)

    results: Dict[str, Any] = {}

    # 1. Baseline without derived features
    logger.info("Evaluating baseline model WITHOUT derived features...")
    prep_no_derived = AdultPreprocessor(include_derived=False)
    prep_no_derived.fit(train_df)
    X_train_no = prep_no_derived.transform(train_df)
    X_test_no = prep_no_derived.transform(test_df)

    clf_no = LogisticRegression(max_iter=1000, random_state=random_state)
    t0 = time.perf_counter()
    clf_no.fit(X_train_no, y_train)
    t_train_no = time.perf_counter() - t0

    preds_no = clf_no.predict(X_test_no)
    acc_no = float(accuracy_score(y_test, preds_no))
    f1_binary_no = float(f1_score(y_test, preds_no, pos_label=1))
    f1_macro_no = float(f1_score(y_test, preds_no, average="macro"))

    results["without_derived_features"] = {
        "accuracy": round(acc_no, 4),
        "f1_binary": round(f1_binary_no, 4),
        "f1_macro": round(f1_macro_no, 4),
        "features_count": int(X_train_no.shape[1]),
        "train_time_sec": round(t_train_no, 3),
    }

    # 2. Baseline with derived features
    logger.info("Evaluating baseline model WITH derived features (capital_net, hours_per_age)...")
    prep_derived = AdultPreprocessor(include_derived=True)
    prep_derived.fit(train_df)
    X_train_derived = prep_derived.transform(train_df)
    X_test_derived = prep_derived.transform(test_df)

    clf_derived = LogisticRegression(max_iter=1000, random_state=random_state)
    t0 = time.perf_counter()
    clf_derived.fit(X_train_derived, y_train)
    t_train_derived = time.perf_counter() - t0

    preds_derived = clf_derived.predict(X_test_derived)
    acc_derived = float(accuracy_score(y_test, preds_derived))
    f1_binary_derived = float(f1_score(y_test, preds_derived, pos_label=1))
    f1_macro_derived = float(f1_score(y_test, preds_derived, average="macro"))

    results["with_derived_features"] = {
        "accuracy": round(acc_derived, 4),
        "f1_binary": round(f1_binary_derived, 4),
        "f1_macro": round(f1_macro_derived, 4),
        "features_count": int(X_train_derived.shape[1]),
        "train_time_sec": round(t_train_derived, 3),
    }

    # Calculate delta
    results["delta"] = {
        "accuracy_gain": round(acc_derived - acc_no, 4),
        "f1_binary_gain": round(f1_binary_derived - f1_binary_no, 4),
        "f1_macro_gain": round(f1_macro_derived - f1_macro_no, 4),
    }

    return results


def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save baseline evaluation metrics to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Baseline metrics saved to: %s", filepath)
