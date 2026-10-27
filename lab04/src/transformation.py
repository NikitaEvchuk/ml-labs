"""
Data transformation module for Lab 4.
Implements a leakage-free cleaning and preprocessing pipeline.
Fits all imputers, percentiles, and encoders strictly on the training set.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default column definitions
DEFAULT_CAT_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

DEFAULT_NUM_COLS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


def split_data(
    df: pd.DataFrame,
    target_col: str = "income",
    test_size: float = 0.30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split raw data into 70% train and 30% test sets with stratification.
    Must be performed BEFORE any cleaning to prevent data leakage.
    """
    # Normalize target column name if needed
    data = df.copy()
    if target_col not in data.columns and "class" in data.columns:
        data = data.rename(columns={"class": target_col})

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        stratify=data[target_col],
        random_state=random_state,
    )

    # Return deep copies to prevent side effects
    return train_df.copy(), test_df.copy()


def drop_train_duplicates(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Drop exact duplicate rows from the training set only.
    Leaves the test set untouched to preserve real-world evaluation distribution.
    """
    initial_count = len(train_df)
    cleaned_train = train_df.drop_duplicates().copy()
    dropped_count = initial_count - len(cleaned_train)
    return cleaned_train, dropped_count


class AdultPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free transformer for the Adult census dataset.
    - Fits categorical modes and numeric medians on train only.
    - Fits 1st and 99th percentiles for numeric clipping on train only.
    - Computes derived features: capital_net and hours_per_age.
    - Scales numeric features and encodes categorical features.
    """

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        include_derived: bool = True,
    ):
        self.categorical_cols = categorical_cols or list(DEFAULT_CAT_COLS)
        self.numeric_cols = numeric_cols or list(DEFAULT_NUM_COLS)
        self.include_derived = include_derived

        # Fitted attributes
        self.cat_modes_: Dict[str, Any] = {}
        self.num_medians_: Dict[str, float] = {}
        self.percentiles_: Dict[str, Tuple[float, float]] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.encoder_: Optional[OneHotEncoder] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit statistics strictly on the training set."""
        df = X.copy()

        # 1. Compute categorical modes on train only
        self.cat_modes_ = {}
        for col in self.categorical_cols:
            if col in df.columns:
                series = df[col].astype(str).str.strip().replace("?", np.nan)
                mode_val = series.dropna().mode()[0]
                self.cat_modes_[col] = mode_val

        # 2. Compute numeric medians on train only
        self.num_medians_ = {}
        for col in self.numeric_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                self.num_medians_[col] = float(series.median())

        # 3. Compute 1st and 99th percentiles on train only
        self.percentiles_ = {}
        for col in self.numeric_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").fillna(self.num_medians_[col])
                q01 = float(series.quantile(0.01))
                q99 = float(series.quantile(0.99))
                self.percentiles_[col] = (q01, q99)

        # 4. Clean train copy to fit scaler and encoder
        clean_df = self._clean_and_impute(df)
        active_num_cols = list(self.numeric_cols)

        if self.include_derived:
            clean_df = self._add_derived_features(clean_df)
            active_num_cols = active_num_cols + ["capital_net", "hours_per_age"]
            # Compute percentiles for derived features on train
            for d_col in ["capital_net", "hours_per_age"]:
                q01 = float(clean_df[d_col].quantile(0.01))
                q99 = float(clean_df[d_col].quantile(0.99))
                self.percentiles_[d_col] = (q01, q99)
                clean_df[d_col] = clean_df[d_col].clip(lower=q01, upper=q99)

        # 5. Fit StandardScaler on numeric features
        self.scaler_ = StandardScaler()
        self.scaler_.fit(clean_df[active_num_cols])

        # 6. Fit OneHotEncoder on categorical features
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(clean_df[self.categorical_cols])

        # Track feature names
        cat_features = list(self.encoder_.get_feature_names_out(self.categorical_cols))
        self.feature_names_ = active_num_cols + cat_features

        return self

    def _clean_and_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values and clip numeric outliers using fitted stats."""
        df = X.copy()

        # Categorical imputation with train modes
        for col in self.categorical_cols:
            if col in df.columns:
                series = df[col].astype(str).str.strip().replace("?", np.nan)
                df[col] = series.fillna(self.cat_modes_[col])

        # Numeric imputation with train medians and clipping with train percentiles
        for col in self.numeric_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                imputed = series.fillna(self.num_medians_[col])
                q01, q99 = self.percentiles_[col]
                df[col] = imputed.clip(lower=q01, upper=q99)

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features without data leakage:
        - capital_net = capital-gain - capital-loss
        - hours_per_age = hours-per-week / (age - 16)
        """
        data = df.copy()
        # capital_net: Net investment gain or loss
        data["capital_net"] = data["capital-gain"] - data["capital-loss"]

        # hours_per_age: Work hours relative to working career length (age >= 17, so denominator >= 1)
        safe_age_denom = (data["age"] - 16).clip(lower=1)
        data["hours_per_age"] = data["hours-per-week"] / safe_age_denom

        return data

    def transform_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return cleaned DataFrame with imputed values, clipped values,
        and derived features (unscaled/unencoded, useful for inspection and saving).
        """
        df = self._clean_and_impute(X)
        if self.include_derived:
            df = self._add_derived_features(df)
            for d_col in ["capital_net", "hours_per_age"]:
                if d_col in self.percentiles_:
                    q01, q99 = self.percentiles_[d_col]
                    df[d_col] = df[d_col].clip(lower=q01, upper=q99)
        return df

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform raw data into a fully processed numeric array:
        imputed, clipped, derived, scaled, and one-hot encoded.
        """
        clean_df = self.transform_to_df(X)
        active_num_cols = list(self.numeric_cols)
        if self.include_derived:
            active_num_cols = active_num_cols + ["capital_net", "hours_per_age"]

        num_scaled = self.scaler_.transform(clean_df[active_num_cols])
        cat_encoded = self.encoder_.transform(clean_df[self.categorical_cols])

        return np.hstack([num_scaled, cat_encoded])


def save_preprocessor(preprocessor: AdultPreprocessor, filepath: str) -> None:
    """Save fitted preprocessor to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)
