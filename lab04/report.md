# Lab 4: Data Pipeline and Quality

## 1. Problem Definition (Reddi §§5.2–5.5, §§6.2–6.4)

### Objective and Task Formulation
The goal of this engineering pipeline is to construct a robust, production-grade, and leakage-free data processing system for the **UCI Adult Census Income** dataset, obtained via `fetch_openml(name='adult', version=2, as_frame=True)`. The task is formulated as a supervised binary classification problem: predict whether an individual's annual compensation exceeds \$50,000 based on demographic, occupational, and financial census indicators.

### Problem Specifications
- **Inputs**: Exactly 14 demographic, occupational, and economic features comprising 6 continuous numerical features (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and 8 categorical attributes (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).
- **Target Output**: Binary income threshold variable `income` (or raw `class`), where:
  - Positive class ($y = 1$): Annual personal income $> \$50\text{K}$ (11,687 instances, ~23.93%).
  - Negative class ($y = 0$): Annual personal income $\le \$50\text{K}$ (37,155 instances, ~76.07%).
- **Success Metrics**:
  - **Accuracy**: Overall fraction of correct predictions across all evaluation records.
  - **F1 Score (Binary & Macro)**: Due to significant class imbalance (~3:1 majority ratio), accuracy alone is misleading (a naive majority-class classifier achieves 76.1% accuracy with zero utility). The primary optimization metric is the F1 score for the positive minority class ($>50\text{K}$), balancing Precision and Recall:
    $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Engineering Constraints**:
  - **Zero Data Leakage**: All data transformations (imputation medians/modes, outlier clipping quantiles, scaling, one-hot encodings) must be strictly fitted on the training split only and applied downstream to the test split. The test set must remain completely invisible during preprocessing parameter calculation.
  - **Test Set Distributional Integrity**: Real-world evaluation distributions must not be artificially pruned; test set duplicate rows must remain untouched.
  - **End-to-End Reproducibility**: The complete pipeline must execute with a single automated command (`python lab04/src/pipeline.py`), maintain strict random seeding (`random_state=42`), track configuration in YAML, and verify data artifacts via SHA-256 cryptographic hashes.

---

## 2. Ingestion and Raw Data Integrity

The ingestion step in [`lab04/src/pipeline.py`](file:///c:/Users/nikit/Documents/repo/lab04/src/pipeline.py) accesses OpenML directly and loads the dataset as an unmodified pandas DataFrame:
- **Raw Row Count**: 48,842 instances.
- **Raw Column Count**: 15 columns (14 features + 1 target).
- **Integrity Policy**: The raw ingestion step performs zero in-place transformations, column renaming, or filtering, logging raw shape metrics and validating dimensional fidelity against the expected reference shape `(48842, 15)`.

---

## 3. Schema, Range, and Quality Validation

The validation step in [`lab04/src/pipeline.py`](file:///c:/Users/nikit/Documents/repo/lab04/src/pipeline.py) verifies incoming data against explicit schema specifications. Violations are logged and compiled into [`validation_report.json`](file:///c:/Users/nikit/Documents/repo/lab04/results/validation_report.json) without dropping any rows.

### Schema and Range Specification
| Column Name | Expected Type | Allowed Range / Domain | Observed Range / Unique | Validation Status | Missing Values (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | Integer | [17, 90] | [17, 90] | **Valid** (0 violations) | 0 (0.00%) |
| `workclass` | Category | 8 allowed categories | 8 categories | **Valid** | 2,799 (5.73%) |
| `fnlwgt` | Integer | Positive integer | [12,285, 1,490,400] | **Valid** | 0 (0.00%) |
| `education` | Category | 16 allowed categories | 16 categories | **Valid** | 0 (0.00%) |
| `education-num` | Integer | [1, 16] | [1, 16] | **Valid** | 0 (0.00%) |
| `marital-status` | Category | 7 allowed categories | 7 categories | **Valid** | 0 (0.00%) |
| `occupation` | Category | 14 allowed categories | 14 categories | **Valid** | 2,809 (5.75%) |
| `relationship` | Category | 6 allowed categories | 6 categories | **Valid** | 0 (0.00%) |
| `race` | Category | 5 allowed categories | 5 categories | **Valid** | 0 (0.00%) |
| `sex` | Category | 2 allowed categories | `Male`, `Female` | **Valid** | 0 (0.00%) |
| `capital-gain` | Integer | [0, 99999] | [0, 99999] | **Valid** (0 violations) | 0 (0.00%) |
| `capital-loss` | Integer | [0, 4356] | [0, 4356] | **Valid** (0 violations) | 0 (0.00%) |
| `hours-per-week` | Integer | [1, 99] | [1, 99] | **Valid** (0 violations) | 0 (0.00%) |
| `native-country` | Category | 41 allowed categories | 41 categories | **Valid** | 857 (1.75%) |
| `income` / `class` | Category | `{'<=50K', '>50K'}` | 2 categories | **Valid** | 0 (0.00%) |

### Missing-Value and Null Handling Policy
- Both explicit `NaN` values and string question marks (`'?'`) are recognized as missing values.
- Missing values are concentrated exclusively in three categorical variables: `workclass` (5.73%), `occupation` (5.75%), and `native-country` (1.75%).
- **Policy Compliance**: As mandated, no rows were dropped during validation. Missing values are propagated to the cleaning stage for train-fitted mode imputation.

---

## 4. Leakage-Free Cleaning and Transformation Pipeline

To ensure the pipeline is leak-free, all data processing steps strictly follow the separation of training and testing phases:

1. **Stratified Train/Test Split**:
   - Performed **prior to any cleaning or parameter calculation**.
   - 70% Train (34,189 rows) / 30% Test (14,653 rows) stratified on the binary target `income` (`random_state=42`).
2. **Duplicate De-duplication**:
   - Exact duplicates are dropped from the training set only (24 duplicate rows dropped, yielding 34,165 training instances).
   - The test set is left completely untouched (14,653 rows preserved), ensuring unbiased real-world generalization evaluation.
3. **Imputation Policy (Train-Only Fitting)**:
   - Categorical missing values (`workclass`, `occupation`, `native-country`) are imputed with the **Mode** computed strictly on the training set (`Private`, `Craft-repair`, and `United-States` respectively).
   - Numerical missing values are imputed with the **Median** computed strictly on the training set.
4. **Outlier Mitigation**:
   - Numeric features (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) are clipped to their 1st and 99th percentiles computed strictly on the training split:
     - `age`: [17.0, 74.0]
     - `fnlwgt`: [24,785.0, 510,039.0]
     - `education-num`: [3.0, 16.0]
     - `capital-gain`: [0.0, 15,024.0] (mitigates the impact of top-coding at 99,999)
     - `capital-loss`: [0.0, 1,980.0]
     - `hours-per-week`: [8.0, 80.0]
5. **Transformer Serialization**:
   - The fitted `AdultPreprocessor` object encapsulates all learned statistics, clipping boundaries, `StandardScaler`, and `OneHotEncoder`.
   - Serialized to [`results/preprocessor.joblib`](file:///c:/Users/nikit/Documents/repo/lab04/results/preprocessor.joblib) via `joblib.dump`.

---

## 5. Feature Engineering and Baseline Evaluation

Two domain-specific features were engineered to capture non-linear financial and labor interactions:
1. **`capital_net`**: Net financial gains from asset investments:
   $$\text{capital\_net} = \text{capital-gain} - \text{capital-loss}$$
2. **`hours_per_age`**: Weekly labor intensity relative to estimated career duration since legal working age (16):
   $$\text{hours\_per\_age} = \frac{\text{hours-per-week}}{\text{age} - 16}$$

### Baseline Model: Logistic Regression Performance
Both configurations were trained with standard L2 regularization (`LogisticRegression(max_iter=1000, random_state=42)`):

| Model Configuration | Feature Count | Test Accuracy | Positive F1 ($>50\text{K}$) | Macro F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Without Derived Features)** | 108 | 0.8559 | 0.6658 | 0.7870 |
| **Enhanced (With Derived Features)** | 110 | **0.8589** | **0.6721** | **0.7911** |
| **Observed Gain ($\Delta$)** | **+2** | **+0.0030 (+0.30%)** | **+0.0063 (+0.63%)** | **+0.0041** |

### Key Insight
Engineering `capital_net` and `hours_per_age` provides direct linear representations of economic surplus and career effort density. In a linear model like Logistic Regression, these derived features reduce the burden of learning interactions implicitly, yielding immediate, consistent improvements in both overall accuracy and minority-class F1 score.

---

## 6. Label-Quality and Conflict Audit

### 1. Duplicate-Conflict Audit
An audit was performed across all 14 input features to locate duplicate rows with conflicting income labels:
- **Rows sharing identical 14-feature tuples**: 111 instances.
- **Distinct conflicting feature patterns**: **5 distinct feature tuples**.
- **Total conflicting rows**: **10 rows** (5 pairs where individuals share identical demographic and occupational attributes but one is labeled $\le \$50\text{K}$ while the other is labeled $> \$50\text{K}$).

**Conflicting Rows Detail**:
1. (Age 46, Private, HS-grad, Married-civ-spouse, Craft-repair, Husband): Row 1781 (`<=50K`) vs Row 32964 (`>50K`)
2. (Age 34, Private, Masters, Married-civ-spouse, Exec-managerial, Not-in-family): Row 2373 (`>50K`) vs Row 45787 (`<=50K`)
3. (Age 41, Private, HS-grad, Married-civ-spouse, Transport-moving, Husband): Row 11692 (`<=50K`) vs Row 16161 (`>50K`)
4. (Age 46, Private, HS-grad, Married-civ-spouse, Transport-moving, Husband): Row 15802 (`>50K`) vs Row 27210 (`<=50K`)
5. (Age 39, Private, Bachelors, Married-civ-spouse, Craft-repair, Husband): Row 17129 (`<=50K`) vs Row 39042 (`>50K`)

**Significance**: Duplicate feature conflicts define a non-zero **Bayes error rate** (label ambiguity) in the dataset: no machine learning algorithm relying purely on these 14 features can ever achieve 100% accuracy on these samples.

### 2. 50-Row Manual Re-check Against Data Dictionary
A random sample of 50 instances (`random_state=42`) was drawn and audited against the official UCI data dictionary:
- **Impossible or Out-of-Range Values**: **0 instances found**.
- **Education vs Education-num Alignment**: 100% concordant (e.g., all `HS-grad` match `education-num=9`, `Bachelors` match `13`, `Masters` match `14`).
- **Marital Status vs Relationship Consistency**: 100% semantically consistent (all instances with `Husband` or `Wife` correspond to `Married-civ-spouse` or `Married-AF-spouse`).
- **Missing Values in Sample**: 2 instances in `workclass`, 2 in `occupation`, and 1 in `native-country` (represented as standard missing values, consistent with global census null distributions).

---

## 7. Artifact Integrity and Reproducibility

### Cryptographic Hashes (SHA-256)
All generated artifacts are hashed and recorded in [`config.yaml`](file:///c:/Users/nikit/Documents/repo/lab04/config.yaml):

| Artifact File | Description | SHA-256 Checksum |
| :--- | :--- | :--- |
| `processed_dataset` | Cleaned full dataset | `8f505bace1f03117b27a28e3f624a6f22f3177c9e7a2607f048d9cb9665a080e` |
| `preprocessor.joblib` | Serialized transformer | `89b4cd10096188e203e59adebe20b8b28888366b645be645818a1c599e1e60ac` |
| `validation_report.json` | Complete validation report | `24534d733bdb13d695d30107f651892717fe96ff86281cd3f4d17ee20c7f4e08` |

### One-Command Pipeline Execution
The pipeline can be executed end-to-end with a single command from the project root:
```bash
python lab04/src/pipeline.py
```
This automatically verifies raw data ingestion, executes schema validation, cleans the data without leakage, fits and saves the transformer, computes baseline evaluation metrics, conducts label audits, and regenerates configuration YAML and SHA-256 signatures.
