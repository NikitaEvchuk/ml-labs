"""
Label quality audit module for Lab 4.
Performs:
1. Duplicate-conflict audit: finds identical feature rows with conflicting income labels.
2. 50-row sample inspection against data dictionary for impossible or anomalous values.
"""

import json
import logging
from typing import Any, Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# Standard education to education-num mapping from UCI Adult documentation
EDUCATION_NUM_MAP = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


def audit_duplicate_conflicts(
    df: pd.DataFrame,
    target_col: str = "income",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Find duplicate feature rows with conflicting income labels.
    Returns the DataFrame of conflicting rows and a summary dictionary.
    """
    data = df.copy()
    if target_col not in data.columns and "class" in data.columns:
        data = data.rename(columns={"class": target_col})

    feature_cols = [c for c in data.columns if c != target_col]

    # Find rows with identical feature values
    dup_feature_mask = data.duplicated(subset=feature_cols, keep=False)
    dup_feature_rows = data[dup_feature_mask]

    # Group by features and identify groups with multiple distinct income labels
    conflicts_df = dup_feature_rows.groupby(feature_cols, observed=True).filter(
        lambda group: group[target_col].nunique() > 1
    )

    # Sort so duplicate pairs appear adjacent
    conflicts_df = conflicts_df.sort_values(by=feature_cols)

    distinct_conflict_patterns = len(
        conflicts_df.drop_duplicates(subset=feature_cols)
    )

    summary = {
        "total_rows_with_duplicate_features": int(dup_feature_mask.sum()),
        "distinct_conflicting_feature_patterns": int(distinct_conflict_patterns),
        "total_conflicting_rows": int(len(conflicts_df)),
    }

    logger.info(
        "Duplicate conflict audit: found %d distinct conflicting feature patterns across %d rows.",
        summary["distinct_conflicting_feature_patterns"],
        summary["total_conflicting_rows"],
    )

    return conflicts_df, summary


def audit_50_row_sample(
    df: pd.DataFrame,
    sample_size: int = 50,
    random_state: int = 42,
    target_col: str = "income",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Draw 50 random rows and audit them against the data dictionary
    for impossible, out-of-range, or inconsistent values.
    """
    data = df.copy()
    if target_col not in data.columns and "class" in data.columns:
        data = data.rename(columns={"class": target_col})

    sample = data.sample(n=sample_size, random_state=random_state).copy()
    sample = sample.reset_index(names="original_index")

    checks: List[str] = []
    impossible_counts = 0

    for idx, row in sample.iterrows():
        issues = []

        # Age check
        if row["age"] < 17 or row["age"] > 90:
            issues.append(f"Age out of bounds: {row['age']}")

        # Hours check
        if row["hours-per-week"] < 1 or row["hours-per-week"] > 99:
            issues.append(f"Hours out of bounds: {row['hours-per-week']}")

        # Capital gain check
        if row["capital-gain"] < 0 or row["capital-gain"] > 99999:
            issues.append(f"Capital gain invalid: {row['capital-gain']}")

        # Capital loss check
        if row["capital-loss"] < 0 or row["capital-loss"] > 4356:
            issues.append(f"Capital loss invalid: {row['capital-loss']}")

        # Education vs Education-num consistency
        edu_str = str(row["education"]).strip()
        edu_num = int(row["education-num"])
        expected_num = EDUCATION_NUM_MAP.get(edu_str)
        if expected_num is not None and expected_num != edu_num:
            issues.append(f"Education mismatch: {edu_str} has num {edu_num}, expected {expected_num}")

        # Marital status vs relationship sanity check
        rel_str = str(row["relationship"]).strip()
        mar_str = str(row["marital-status"]).strip()
        if rel_str in ["Husband", "Wife"] and not mar_str.startswith("Married"):
            issues.append(f"Relationship '{rel_str}' inconsistent with marital status '{mar_str}'")

        if issues:
            impossible_counts += 1
            checks.append("; ".join(issues))
        else:
            checks.append("Valid: All values match data dictionary constraints")

    sample["audit_status"] = checks

    summary = {
        "sample_size": sample_size,
        "random_state": random_state,
        "rows_with_impossible_values": impossible_counts,
        "missing_values_in_sample": {
            col: int(sample[col].isna().sum() + (sample[col].astype(str).str.strip() == "?").sum())
            for col in sample.columns
            if col not in ["original_index", "audit_status"]
            and (sample[col].isna().sum() + (sample[col].astype(str).str.strip() == "?").sum()) > 0
        },
    }

    logger.info(
        "50-row manual audit completed: %d rows with impossible values found.",
        impossible_counts,
    )
    return sample, summary
