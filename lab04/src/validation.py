"""
Schema and data validation module for Lab 4.
Validates raw data types, expected ranges, and missing values.
Reports all violations without dropping rows.
"""

import json
import logging
from typing import Any, Dict
import pandas as pd

logger = logging.getLogger(__name__)

# Expected schema definition
SCHEMA_TYPES = {
    "age": "int",
    "workclass": "category",
    "fnlwgt": "int",
    "education": "category",
    "education-num": "int",
    "marital-status": "category",
    "occupation": "category",
    "relationship": "category",
    "race": "category",
    "sex": "category",
    "capital-gain": "int",
    "capital-loss": "int",
    "hours-per-week": "int",
    "native-country": "category",
    "income": "category",
}

# Expected value ranges for numeric features
NUMERIC_RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run comprehensive schema, range, and missing value checks.
    Does not drop any rows; returns a complete validation report.
    """
    report: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "total_cols": int(len(df.columns)),
        "schema_checks": {},
        "range_checks": {},
        "missing_value_checks": {},
        "violations_found": False,
        "violation_details": [],
    }

    # Standardize target column name if needed ('class' -> 'income')
    check_df = df.copy()
    if "class" in check_df.columns and "income" not in check_df.columns:
        check_df = check_df.rename(columns={"class": "income"})

    # 1. Column existence check
    missing_cols = [c for c in SCHEMA_TYPES if c not in check_df.columns]
    if missing_cols:
        report["violations_found"] = True
        msg = f"Missing required columns: {missing_cols}"
        report["violation_details"].append(msg)
        logger.error(msg)

    # 2. Type validation
    for col, expected_type in SCHEMA_TYPES.items():
        if col not in check_df.columns:
            continue
        actual_type = str(check_df[col].dtype)
        is_type_valid = True

        if expected_type == "int":
            # Check if values are integer-compatible
            is_type_valid = pd.api.types.is_integer_dtype(check_df[col]) or pd.api.types.is_float_dtype(check_df[col])
        elif expected_type == "category":
            # Allowed as category or object string
            is_type_valid = isinstance(check_df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(check_df[col])

        report["schema_checks"][col] = {
            "expected": expected_type,
            "actual": actual_type,
            "valid": bool(is_type_valid),
        }
        if not is_type_valid:
            report["violations_found"] = True
            msg = f"Type mismatch for '{col}': expected {expected_type}, got {actual_type}"
            report["violation_details"].append(msg)
            logger.warning(msg)

    # 3. Numeric range validation
    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col not in check_df.columns:
            continue
        series = pd.to_numeric(check_df[col], errors="coerce")
        out_of_bounds = (series < min_val) | (series > max_val)
        violations_count = int(out_of_bounds.sum())

        report["range_checks"][col] = {
            "min_allowed": min_val,
            "max_allowed": max_val,
            "observed_min": float(series.min()) if not series.empty else None,
            "observed_max": float(series.max()) if not series.empty else None,
            "violations_count": violations_count,
            "valid": (violations_count == 0),
        }
        if violations_count > 0:
            report["violations_found"] = True
            msg = f"Range violation in '{col}': {violations_count} values outside [{min_val}, {max_val}]"
            report["violation_details"].append(msg)
            logger.warning(msg)

    # 4. Missing value checks (including '?' policy)
    for col in check_df.columns:
        series = check_df[col]
        # Check standard NaNs
        nan_count = int(series.isna().sum())
        # Check '?' values
        qmark_count = 0
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            qmark_count = int((series.astype(str).str.strip() == "?").sum())

        total_missing = nan_count + qmark_count
        missing_pct = round((total_missing / len(check_df)) * 100, 3)

        report["missing_value_checks"][col] = {
            "nan_count": nan_count,
            "question_mark_count": qmark_count,
            "total_missing": total_missing,
            "missing_percentage": missing_pct,
        }

        if total_missing > 0:
            logger.info(
                "Missing values detected in '%s': %d (%.2f%%) [NaN: %d, '?': %d]",
                col,
                total_missing,
                missing_pct,
                nan_count,
                qmark_count,
            )

    logger.info("Validation completed. Total violations found: %d", len(report["violation_details"]))
    return report


def save_validation_report(report: Dict[str, Any], output_path: str) -> None:
    """Save validation report to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved to: %s", output_path)
