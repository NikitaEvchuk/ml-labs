# Lab 4: Data Pipeline and Quality

## Overview
This laboratory builds an end-to-end, reproducible, leakage-free data pipeline for mixed numerical and categorical data using the UCI Adult Census Income dataset via `fetch_openml(name='adult', version=2, as_frame=True)`.

The pipeline defines and executes strict schema and range checks, prevents data leakage via train-only fitted transformers, evaluates derived features (`capital_net`, `hours_per_age`), audits label quality, and generates cryptographic integrity hashes and a comprehensive data card.

## Directory Structure
```
lab04/
├── data_card.md              # Complete data card (source, license, size, biases, privacy)
├── report.md                 # Full laboratory report with methodology and results
├── README.md                 # Lab documentation and execution guide
├── config.yaml               # Pipeline configuration and SHA-256 hashes
├── src/
│   ├── __init__.py           # Package marker
│   ├── ingestion.py          # Reads raw OpenML Adult v2 without modification
│   ├── validation.py         # Schema, range, and missing-value checks (no silent dropping)
│   ├── transformation.py     # Leakage-free train/test split, imputer, percentile clipping
│   ├── baseline.py           # Logistic Regression evaluation with & without derived features
│   ├── label_quality.py      # Duplicate-conflict audit and 50-row manual check
│   └── pipeline.py           # Main entry point (one-command execution)
└── results/
    ├── preprocessor.joblib   # Saved fitted transformer
    ├── config.yaml           # Pipeline config and artifact checksums
    ├── validation_report.json# Schema and range validation report
    ├── metrics.json          # Logistic Regression test accuracy & F1 comparison
    ├── duplicate_conflicts.csv # Audit of duplicate feature rows with conflicting labels
    ├── sample_50_check.csv   # 50 random rows audited against data dictionary
    ├── train_processed.csv   # Cleaned training dataset
    ├── test_processed.csv    # Cleaned test dataset (untouched)
    └── processed_hashes.txt  # SHA-256 checksums of all generated artifacts
```

## How to Run with One Command
From the repository root directory, run:
```bash
python lab04/src/pipeline.py
```
Or using the virtual environment:
```powershell
& .venv\Scripts\python.exe lab04\src\pipeline.py
```

## Graded Evidence Summary
1. **Validation Report**: Generated at `lab04/results/validation_report.json` covering schema types, bounds, and null policies without silently dropping any rows.
2. **Leakage-Free Pipeline**:
   - 70/30 stratified split performed before cleaning.
   - Categorical mode and numerical median imputers fitted strictly on training data.
   - 1st and 99th percentile clipping boundaries computed strictly on training data.
   - Duplicates dropped from train set only (test set kept untouched).
   - Serialized transformer saved at `lab04/results/preprocessor.joblib`.
3. **Data Card**: Authored at `lab04/data_card.md` detailing dataset provenance, licensing (CC BY 4.0), dimensions, schema, known biases (gender, race, geography, inflation), and privacy concerns (re-identification, census reconstruction).
4. **One-Command Reproducibility**: Fully automated execution via `python lab04/src/pipeline.py`.
