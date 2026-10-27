"""
Main end-to-end reproducible pipeline for Lab 4.
Orchestrates:
1. Ingestion of raw OpenML Adult v2 dataset (48842 rows, 15 columns).
2. Schema, range, null-policy validation (reports violations without dropping rows).
3. Leakage-free 70/30 stratified train/test split.
4. Cleaning: train-only imputation (mode/median), train-only outlier clipping (1st/99th percentiles),
   and train-only duplicate removal.
5. Transformer serialization to results/preprocessor.joblib.
6. Feature engineering (capital_net, hours_per_age) and Logistic Regression baseline evaluation.
7. Label-quality audit (duplicate-conflict audit and 50-row data dictionary check).
8. Processed dataset SHA256 hashing and config YAML generation.

Runnable with one command:
    python lab04/src/pipeline.py
"""

import hashlib
import json
import logging
import os
import sys
from typing import Any, Dict
import pandas as pd
import yaml

# Set up project root in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lab04.src.ingestion import load_raw_data
from lab04.src.validation import validate_dataset, save_validation_report
from lab04.src.transformation import (
    split_data,
    drop_train_duplicates,
    AdultPreprocessor,
    save_preprocessor,
)
from lab04.src.baseline import evaluate_baselines, save_metrics
from lab04.src.label_quality import audit_duplicate_conflicts, audit_50_row_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lab04_pipeline")


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_pipeline() -> Dict[str, Any]:
    """Execute the full end-to-end data pipeline."""
    base_dir = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("================================================================")
    logger.info("STARTING LAB 4 DATA PIPELINE EXECUTION")
    logger.info("================================================================")

    # -------------------------------------------------------------------------
    # STEP 1 & 2: Ingestion
    # -------------------------------------------------------------------------
    logger.info("[Step 2/7] Ingesting raw UCI Adult dataset without modification...")
    raw_df = load_raw_data()
    row_count, col_count = raw_df.shape
    logger.info("Raw dataset shape: %d rows, %d columns", row_count, col_count)

    # Standardize target name for internal processing
    df = raw_df.copy()
    if "class" in df.columns and "income" not in df.columns:
        df = df.rename(columns={"class": "income"})

    # -------------------------------------------------------------------------
    # STEP 3: Validation
    # -------------------------------------------------------------------------
    logger.info("[Step 3/7] Validating schema, ranges, and null policy...")
    validation_report = validate_dataset(df)
    validation_path = os.path.join(results_dir, "validation_report.json")
    save_validation_report(validation_report, validation_path)

    # -------------------------------------------------------------------------
    # STEP 4: Leakage-free Train/Test Split & Cleaning
    # -------------------------------------------------------------------------
    logger.info("[Step 4/7] Splitting train/test 70/30 stratified before cleaning...")
    train_df, test_df = split_data(df, target_col="income", test_size=0.30, random_state=42)
    logger.info("Initial split: Train=%d rows, Test=%d rows", len(train_df), len(test_df))

    # Drop exact duplicates from train only (keep test untouched)
    train_clean, dropped_dups = drop_train_duplicates(train_df)
    logger.info("Dropped %d exact duplicate rows from train only (new train count: %d)", dropped_dups, len(train_clean))
    logger.info("Test set kept untouched: %d rows", len(test_df))

    # Fit preprocessor on train only
    logger.info("Fitting transformer on train set only...")
    preprocessor = AdultPreprocessor(include_derived=True)
    preprocessor.fit(train_clean)

    # Save fitted preprocessor
    preprocessor_path = os.path.join(results_dir, "preprocessor.joblib")
    save_preprocessor(preprocessor, preprocessor_path)
    logger.info("Saved fitted preprocessor to: %s", preprocessor_path)

    # Clean and transform DataFrames for export
    train_processed_df = preprocessor.transform_to_df(train_clean)
    train_processed_df["income"] = train_clean["income"]

    test_processed_df = preprocessor.transform_to_df(test_df)
    test_processed_df["income"] = test_df["income"]

    train_csv_path = os.path.join(results_dir, "train_processed.csv")
    test_csv_path = os.path.join(results_dir, "test_processed.csv")
    train_processed_df.to_csv(train_csv_path, index=False)
    test_processed_df.to_csv(test_csv_path, index=False)
    logger.info("Saved processed train set (%d rows) and test set (%d rows)", len(train_processed_df), len(test_processed_df))

    # -------------------------------------------------------------------------
    # STEP 5: Feature Engineering & Baseline Logistic Regression Evaluation
    # -------------------------------------------------------------------------
    logger.info("[Step 5/7] Evaluating LogisticRegression baselines (with and without derived features)...")
    metrics = evaluate_baselines(train_clean, test_df, target_col="income", random_state=42)
    metrics_path = os.path.join(results_dir, "metrics.json")
    save_metrics(metrics, metrics_path)

    logger.info("Baseline without derived: Accuracy=%.4f, F1(binary)=%.4f, F1(macro)=%.4f",
                metrics["without_derived_features"]["accuracy"],
                metrics["without_derived_features"]["f1_binary"],
                metrics["without_derived_features"]["f1_macro"])
    logger.info("Baseline WITH derived:    Accuracy=%.4f, F1(binary)=%.4f, F1(macro)=%.4f",
                metrics["with_derived_features"]["accuracy"],
                metrics["with_derived_features"]["f1_binary"],
                metrics["with_derived_features"]["f1_macro"])
    logger.info("Gain from derived features: Delta Accuracy=%+.4f, Delta F1(binary)=%+.4f",
                metrics["delta"]["accuracy_gain"],
                metrics["delta"]["f1_binary_gain"])

    # -------------------------------------------------------------------------
    # STEP 6: Label-Quality Check
    # -------------------------------------------------------------------------
    logger.info("[Step 6/7] Performing label quality audits...")
    # Duplicate-conflict audit
    conflicts_df, conflicts_summary = audit_duplicate_conflicts(df, target_col="income")
    conflicts_path = os.path.join(results_dir, "duplicate_conflicts.csv")
    conflicts_df.to_csv(conflicts_path, index=False)
    logger.info("Duplicate-conflict audit: %d rows with conflicting income labels saved to %s",
                len(conflicts_df), conflicts_path)

    # 50-row random sample manual check
    sample_50_df, sample_50_summary = audit_50_row_sample(df, sample_size=50, random_state=42, target_col="income")
    sample_50_path = os.path.join(results_dir, "sample_50_check.csv")
    sample_50_df.to_csv(sample_50_path, index=False)
    logger.info("50-row manual audit saved to: %s", sample_50_path)

    # -------------------------------------------------------------------------
    # STEP 7: SHA256 Hashes and Config YAML
    # -------------------------------------------------------------------------
    logger.info("[Step 7/7] Computing SHA256 hashes and storing configuration...")
    sha256_hashes = {
        "train_processed_csv": compute_sha256(train_csv_path),
        "test_processed_csv": compute_sha256(test_csv_path),
        "preprocessor_joblib": compute_sha256(preprocessor_path),
        "validation_report_json": compute_sha256(validation_path),
        "metrics_json": compute_sha256(metrics_path),
        "duplicate_conflicts_csv": compute_sha256(conflicts_path),
        "sample_50_check_csv": compute_sha256(sample_50_path),
    }

    # Save hashes text file
    hashes_txt_path = os.path.join(results_dir, "processed_hashes.txt")
    with open(hashes_txt_path, "w", encoding="utf-8") as f:
        for fname, hval in sha256_hashes.items():
            f.write(f"{hval}  {fname}\n")
    logger.info("Saved hashes to: %s", hashes_txt_path)

    # Construct complete config YAML
    config_dict = {
        "pipeline_metadata": {
            "name": "Lab 4 Data Pipeline and Quality",
            "author": "Student",
            "dataset": "UCI Adult (OpenML name='adult', version=2)",
            "random_seed": 42,
        },
        "raw_data": {
            "source": "OpenML",
            "name": "adult",
            "version": 2,
            "row_count": row_count,
            "column_count": col_count,
            "target_column": "income",
            "positive_label": ">50K",
            "negative_label": "<=50K",
        },
        "validation_schema": {
            "numeric_ranges": {
                "age": [17, 90],
                "hours_per_week": [1, 99],
                "capital_gain": [0, 99999],
                "capital_loss": [0, 4356],
            },
            "null_policy": "Question mark '?' or NaN denotes missing value; missing values are imputed, not silently dropped",
        },
        "split_and_cleaning": {
            "train_ratio": 0.70,
            "test_ratio": 0.30,
            "stratified": True,
            "initial_train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_duplicates_dropped": dropped_dups,
            "final_train_rows": len(train_clean),
            "imputation": {
                "categorical_strategy": "mode (fitted on train only)",
                "numerical_strategy": "median (fitted on train only)",
            },
            "outlier_clipping": {
                "lower_quantile": 0.01,
                "upper_quantile": 0.99,
                "computed_on": "train set only",
            },
        },
        "feature_engineering": {
            "derived_features": [
                {
                    "name": "capital_net",
                    "formula": "capital-gain - capital-loss",
                    "description": "Net capital gain or loss from investments",
                },
                {
                    "name": "hours_per_age",
                    "formula": "hours-per-week / (age - 16)",
                    "description": "Weekly working hours scaled by years elapsed since minimum legal working age (16)",
                },
            ]
        },
        "baseline_evaluation": metrics,
        "label_quality_audit": {
            "conflicting_feature_patterns": conflicts_summary["distinct_conflicting_feature_patterns"],
            "conflicting_total_rows": conflicts_summary["total_conflicting_rows"],
            "sample_50_impossible_values": sample_50_summary["rows_with_impossible_values"],
        },
        "sha256_checksums": sha256_hashes,
    }

    # Store config YAML in results/ and in lab04 root
    results_yaml_path = os.path.join(results_dir, "config.yaml")
    root_yaml_path = os.path.join(base_dir, "config.yaml")

    with open(results_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    with open(root_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved configuration YAML to: %s and %s", results_yaml_path, root_yaml_path)
    logger.info("================================================================")
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("================================================================")

    return config_dict


if __name__ == "__main__":
    run_pipeline()
