"""
Lab 4: Data pipeline and quality
Single-file pipeline that handles:
- Ingestion of UCI Adult (OpenML name='adult', version=2)
- Schema, range, and missing value validation (no silent dropping)
- Leakage-free 70/30 stratified train/test split and cleaning
- Feature engineering (capital_net, hours_per_age)
- Baseline Logistic Regression evaluation
- Label quality audit (duplicate conflicts and 50-row manual check)
- SHA-256 hashing and config YAML output
"""

import hashlib
import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# Column definitions
NUMERIC_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]
NUMERIC_RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}


# -----------------------------------------------------------------------------
# 1. Ingestion Step
# -----------------------------------------------------------------------------
def ingest_data() -> pd.DataFrame:
    """Read raw UCI Adult data from OpenML without modification."""
    logger.info("Step 1: Reading raw data from OpenML (name='adult', version=2)...")
    data = fetch_openml(name="adult", version=2, as_frame=True)
    raw_df = data.frame.copy()

    # Log row count and column count
    num_rows, num_cols = raw_df.shape
    logger.info("Raw data loaded: %d rows and %d columns", num_rows, num_cols)
    return raw_df


# -----------------------------------------------------------------------------
# 2. Validation Step
# -----------------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> dict:
    """Validate schema, ranges, and nulls. Never silently drop rows."""
    logger.info("Step 2: Running schema, range, and missing value checks...")
    target_col = "class" if "class" in df.columns else "income"

    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "range_checks": {},
        "missing_counts": {},
        "violations": []
    }

    # Range checks for numeric columns
    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_bounds = ((series < min_val) | (series > max_val)).sum()
        report["range_checks"][col] = {
            "min_allowed": min_val,
            "max_allowed": max_val,
            "violations": int(out_of_bounds)
        }
        if out_of_bounds > 0:
            report["violations"].append(f"{col}: {out_of_bounds} values out of bounds")

    # Missing value checks ('?' or NaN means missing)
    for col in df.columns:
        series = df[col]
        nan_count = int(series.isna().sum())
        qmark_count = 0
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            qmark_count = int((series.astype(str).str.strip() == "?").sum())
        total_missing = nan_count + qmark_count
        if total_missing > 0:
            report["missing_counts"][col] = total_missing

    logger.info("Validation completed. Missing columns: %s", list(report["missing_counts"].keys()))
    return report


# -----------------------------------------------------------------------------
# 3. Transformer Class (Leakage-free)
# -----------------------------------------------------------------------------
class AdultPreprocessor(BaseEstimator, TransformerMixin):
    """Fitted on train only: imputes, clips outliers, adds derived features, encodes."""

    def __init__(self, include_derived: bool = True):
        self.include_derived = include_derived
        self.cat_modes_ = {}
        self.num_medians_ = {}
        self.percentiles_ = {}
        self.scaler_ = StandardScaler()
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X: pd.DataFrame, y=None):
        """Fit all statistics strictly on training set."""
        df = X.copy()

        # Categorical modes on train only
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                cleaned = df[col].astype(str).str.strip().replace("?", np.nan)
                self.cat_modes_[col] = cleaned.dropna().mode()[0]

        # Numeric medians and 1st/99th percentiles on train only
        for col in NUMERIC_COLS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                med = float(series.median())
                self.num_medians_[col] = med
                filled = series.fillna(med)
                q01 = float(filled.quantile(0.01))
                q99 = float(filled.quantile(0.99))
                self.percentiles_[col] = (q01, q99)

        # Fit derived feature percentiles on train if enabled
        clean_df = self.clean_dataframe(df)
        active_nums = list(NUMERIC_COLS)
        if self.include_derived:
            active_nums += ["capital_net", "hours_per_age"]
            for d_col in ["capital_net", "hours_per_age"]:
                q01 = float(clean_df[d_col].quantile(0.01))
                q99 = float(clean_df[d_col].quantile(0.99))
                self.percentiles_[d_col] = (q01, q99)
                clean_df[d_col] = clean_df[d_col].clip(lower=q01, upper=q99)

        # Fit scaler and one-hot encoder on train
        self.scaler_.fit(clean_df[active_nums])
        self.encoder_.fit(clean_df[CATEGORICAL_COLS])
        return self

    def clean_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation, clipping, and derived features using fitted train stats."""
        df = X.copy()

        # Impute categoricals with train mode
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                cleaned = df[col].astype(str).str.strip().replace("?", np.nan)
                df[col] = cleaned.fillna(self.cat_modes_[col])

        # Impute and clip numerics with train medians and percentiles
        for col in NUMERIC_COLS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").fillna(self.num_medians_[col])
                q01, q99 = self.percentiles_[col]
                df[col] = series.clip(lower=q01, upper=q99)

        # Add derived features
        if self.include_derived:
            # capital_net = capital-gain - capital-loss
            df["capital_net"] = df["capital-gain"] - df["capital-loss"]
            # hours_per_age = hours-per-week / (age - 16)
            denom = (df["age"] - 16).clip(lower=1)
            df["hours_per_age"] = df["hours-per-week"] / denom
            for d_col in ["capital_net", "hours_per_age"]:
                if d_col in self.percentiles_:
                    q01, q99 = self.percentiles_[d_col]
                    df[d_col] = df[d_col].clip(lower=q01, upper=q99)

        return df

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform into encoded and scaled array."""
        clean_df = self.clean_dataframe(X)
        active_nums = list(NUMERIC_COLS)
        if self.include_derived:
            active_nums += ["capital_net", "hours_per_age"]

        num_part = self.scaler_.transform(clean_df[active_nums])
        cat_part = self.encoder_.transform(clean_df[CATEGORICAL_COLS])
        return np.hstack([num_part, cat_part])


# -----------------------------------------------------------------------------
# 4. Label Quality Audits
# -----------------------------------------------------------------------------
def audit_labels(df: pd.DataFrame, target_col: str = "income") -> dict:
    """Find duplicate feature rows with conflicting income labels, and check 50 rows."""
    logger.info("Step 4: Running label quality audit...")
    features = [c for c in df.columns if c != target_col]

    # Duplicate conflict audit
    dup_mask = df.duplicated(subset=features, keep=False)
    conflicts = df[dup_mask].groupby(features, observed=True).filter(lambda g: g[target_col].nunique() > 1)
    distinct_patterns = len(conflicts.drop_duplicates(subset=features))
    total_conflicting_rows = len(conflicts)

    # 50 random rows check against data dictionary
    sample_50 = df.sample(n=50, random_state=42)
    impossible_rows = 0
    for _, row in sample_50.iterrows():
        if row["age"] < 17 or row["age"] > 90:
            impossible_rows += 1
        if row["hours-per-week"] < 1 or row["hours-per-week"] > 99:
            impossible_rows += 1

    return {
        "distinct_conflicting_patterns": int(distinct_patterns),
        "total_conflicting_rows": int(total_conflicting_rows),
        "sample_50_impossible_count": int(impossible_rows)
    }


# -----------------------------------------------------------------------------
# 5. Main Pipeline Execution
# -----------------------------------------------------------------------------
def run_pipeline():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info("==================================================")
    logger.info("STARTING DATA PIPELINE RUN")
    logger.info("==================================================")

    # 1. Ingestion
    raw_df = ingest_data()
    target_col = "income"
    df = raw_df.rename(columns={"class": target_col}) if "class" in raw_df.columns else raw_df.copy()

    # 2. Validation
    val_report = validate_data(df)
    val_path = os.path.join(RESULTS_DIR, "validation_report.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    # 3. Stratified 70/30 split before cleaning
    logger.info("Step 3: Stratified 70/30 train/test split before cleaning...")
    train_df, test_df = train_test_split(
        df, test_size=0.30, stratify=df[target_col], random_state=42
    )

    # Drop exact duplicates from train only (keep test untouched)
    train_clean = train_df.drop_duplicates().copy()
    dropped_dups = len(train_df) - len(train_clean)
    logger.info("Dropped %d duplicate rows from train only. Test rows preserved: %d", dropped_dups, len(test_df))

    # Fit preprocessor on train only and save
    preprocessor = AdultPreprocessor(include_derived=True)
    preprocessor.fit(train_clean)
    prep_path = os.path.join(RESULTS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, prep_path)
    logger.info("Saved fitted preprocessor to %s", prep_path)

    # Save cleaned processed datasets
    train_proc = preprocessor.clean_dataframe(train_clean)
    test_proc = preprocessor.clean_dataframe(test_df)
    train_csv = os.path.join(RESULTS_DIR, "train_processed.csv")
    test_csv = os.path.join(RESULTS_DIR, "test_processed.csv")
    train_proc.to_csv(train_csv, index=False)
    test_proc.to_csv(test_csv, index=False)

    # 4. Feature Engineering & Logistic Regression Baseline Evaluation
    logger.info("Step 5: Testing Logistic Regression with & without derived features...")
    y_train = (train_clean[target_col] == ">50K").astype(int)
    y_test = (test_df[target_col] == ">50K").astype(int)

    # Without derived
    prep_base = AdultPreprocessor(include_derived=False)
    prep_base.fit(train_clean)
    X_train_base = prep_base.transform(train_clean)
    X_test_base = prep_base.transform(test_df)
    clf_base = LogisticRegression(max_iter=1000, random_state=42)
    clf_base.fit(X_train_base, y_train)
    preds_base = clf_base.predict(X_test_base)
    acc_base = float(accuracy_score(y_test, preds_base))
    f1_base = float(f1_score(y_test, preds_base))

    # With derived (capital_net, hours_per_age)
    X_train_eng = preprocessor.transform(train_clean)
    X_test_eng = preprocessor.transform(test_df)
    clf_eng = LogisticRegression(max_iter=1000, random_state=42)
    clf_eng.fit(X_train_eng, y_train)
    preds_eng = clf_eng.predict(X_test_eng)
    acc_eng = float(accuracy_score(y_test, preds_eng))
    f1_eng = float(f1_score(y_test, preds_eng))

    logger.info("Without derived: Accuracy=%.4f, F1=%.4f", acc_base, f1_base)
    logger.info("With derived:    Accuracy=%.4f, F1=%.4f", acc_eng, f1_eng)

    # Save metrics
    metrics = {
        "without_derived": {"accuracy": round(acc_base, 4), "f1": round(f1_base, 4)},
        "with_derived": {"accuracy": round(acc_eng, 4), "f1": round(f1_eng, 4)},
        "delta": {"accuracy_gain": round(acc_eng - acc_base, 4), "f1_gain": round(f1_eng - f1_base, 4)}
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 5. Label Quality Audit
    label_audit = audit_labels(df, target_col=target_col)
    logger.info("Label audit: %d conflicting feature groups across %d rows",
                label_audit["distinct_conflicting_patterns"], label_audit["total_conflicting_rows"])

    # 6. SHA256 Hashes & Config YAML
    def file_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(65536):
                h.update(chunk)
        return h.hexdigest()

    hashes = {
        "train_processed_csv": file_sha256(train_csv),
        "test_processed_csv": file_sha256(test_csv),
        "preprocessor_joblib": file_sha256(prep_path),
        "validation_report_json": file_sha256(val_path),
    }

    config = {
        "dataset": {"name": "adult", "version": 2, "rows": len(raw_df), "cols": len(raw_df.columns)},
        "split": {"train_ratio": 0.70, "test_ratio": 0.30, "stratified": True, "seed": 42},
        "cleaning": {
            "dropped_train_duplicates": dropped_dups,
            "categorical_imputer": "mode (fitted on train only)",
            "numeric_imputer": "median (fitted on train only)",
            "outlier_clipping": "1st/99th percentiles (fitted on train only)"
        },
        "derived_features": ["capital_net", "hours_per_age"],
        "evaluation": metrics,
        "label_audit": label_audit,
        "sha256_checksums": hashes
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Config and hashes saved to %s", CONFIG_PATH)
    logger.info("==================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    run_pipeline()
