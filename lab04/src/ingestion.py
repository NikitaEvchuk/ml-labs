"""
Data ingestion module for Lab 4.
Loads the raw UCI Adult dataset from OpenML without modification.
"""

import logging
import pandas as pd
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """
    Fetch the raw UCI Adult dataset from OpenML.
    Returns the unmodified raw DataFrame.
    """
    # Fetch dataset from OpenML (version 2 as specified in prompt)
    logger.info("Fetching raw Adult dataset from OpenML (name='adult', version=2)...")
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    raw_df = dataset.frame.copy()

    # Log row count and column count
    num_rows, num_cols = raw_df.shape
    logger.info("Ingestion completed: loaded %d rows and %d columns", num_rows, num_cols)

    # Sanity check against expected dimensions
    expected_rows = 48842
    expected_cols = 15
    if num_rows != expected_rows or num_cols != expected_cols:
        logger.warning(
            "Dimension mismatch! Expected (%d, %d), but got (%d, %d)",
            expected_rows,
            expected_cols,
            num_rows,
            num_cols,
        )
    else:
        logger.info("Dimension check passed: exactly %d rows and %d columns", expected_rows, expected_cols)

    return raw_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = load_raw_data()
    print(f"Loaded DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
