# Data Card: UCI Adult Income Dataset

## 1. Dataset Overview and Source
- **Dataset Name**: UCI Adult Census Income Dataset (Census Income / Adult)
- **Source**: Retrieved from OpenML (`name='adult'`, `version=2`, ID: 1590) via scikit-learn `fetch_openml`.
- **Original Donor**: Ronny Kohavi and Barry Becker, Data Mining and Visualization, Silicon Graphics (1996).
- **Original Source**: 1994 United States Census Bureau database (Current Population Survey).
- **Canonical Repository Link**: [UCI Machine Learning Repository: Adult](https://archive.ics.uci.edu/dataset/2/adult)

## 2. License and Terms of Use
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Permitted Uses**: Academic research, educational benchmarks, public machine learning benchmarking, commercial evaluation with appropriate attribution.
- **Restrictions**: Commercial and non-commercial users must cite the original creators and the UCI Machine Learning Repository.

## 3. Dataset Size and Dimensions
- **Raw Ingestion Shape**: 48,842 instances across 15 columns (14 predictive features + 1 target column).
- **Data Partitions (70/30 Stratified Split)**:
  - **Raw Training Set**: 34,189 rows x 15 columns.
  - **Cleaned Training Set**: 34,165 rows x 16 columns (after dropping 24 exact duplicates and adding 2 derived features).
  - **Test Set**: 14,653 rows x 16 columns (test set kept untouched; duplicates preserved; 2 derived features added).
- **Storage Format**: CSV for tabular processed splits, YAML for pipeline configuration, Joblib for serialized transformer weights.

## 4. Feature Dictionary and Schema

| Feature Name | Type | Physical / Statistical Meaning | Expected Range / Domain | Null Count in Raw (%) |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Integer | Chronological age of the individual in years | [17, 90] | 0 (0.00%) |
| `workclass` | Category | General employment sector / employer category | 8 categories (Private, Self-emp, Gov...) | 2,799 (5.73%) |
| `fnlwgt` | Integer | Final sampling weight assigned by Census Bureau | [12,285, 1,490,400] | 0 (0.00%) |
| `education` | Category | Highest completed level of formal schooling | 16 categories (Preschool to Doctorate) | 0 (0.00%) |
| `education-num` | Integer | Continuous numerical representation of schooling years | [1, 16] | 0 (0.00%) |
| `marital-status` | Category | Current legal marital status | 7 categories (Married, Divorced, Never-married...) | 0 (0.00%) |
| `occupation` | Category | Specific professional occupational category | 14 categories (Exec, Prof-specialty, Craft...) | 2,809 (5.75%) |
| `relationship` | Category | Familial relationship to the household head | 6 categories (Husband, Wife, Own-child...) | 0 (0.00%) |
| `race` | Category | Self-reported racial classification | 5 categories (White, Black, Asian-Pac-Islander...) | 0 (0.00%) |
| `sex` | Category | Biological sex as recorded in the 1994 census | 2 categories (`Male`, `Female`) | 0 (0.00%) |
| `capital-gain` | Integer | Income generated from capital asset appreciation | [0, 99999] (artificially top-coded at 99999) | 0 (0.00%) |
| `capital-loss` | Integer | Financial loss incurred from capital investment sales | [0, 4356] | 0 (0.00%) |
| `hours-per-week` | Integer | Stated work hours performed per week | [1, 99] | 0 (0.00%) |
| `native-country` | Category | Birth country of origin | 41 country categories | 857 (1.75%) |
| **`income`** (target) | Category | Annual income classification threshold | `{<=50K, >50K}` (37,155 `<=50K`, 11,687 `>50K`) | 0 (0.00%) |
| `capital_net` (derived) | Integer | Engineered feature: `capital-gain - capital-loss` | Net capital balance | 0 (0.00%) |
| `hours_per_age` (derived) | Float | Engineered feature: `hours-per-week / (age - 16)` | Working intensity relative to career span | 0 (0.00%) |

## 5. Known Biases and Limitations
1. **Severe Demographic Imbalance**:
   - **Gender Bias**: Approximately 66.8% of individuals are recorded as `Male` and 33.2% as `Female`. High earners (`>50K`) are disproportionately male (84.6% of `>50K` individuals are male). Models trained naively on this data reflect historical wage disparity.
   - **Racial Bias**: Over 85.5% of the records identify as `White`, while `Black` constitutes 9.6%, `Asian-Pac-Islander` 3.1%, and `Amer-Indian-Eskimo` 0.9%. Predictions for minority sub-populations exhibit higher variance and potential performance disparity.
2. **Geographic Skew**:
   - Approximately 89.7% of surveyed individuals report `United-States` as their native country, limiting generalizability to international demographic cohorts.
3. **Temporal Drift and Inflation Obsolescence**:
   - The survey represents 1994 macroeconomic conditions. Due to cumulative economic inflation over three decades, \$50,000 in 1994 corresponds to approximately \$105,000 in modern purchasing power. Treating \$50K as an absolute benchmark today is obsolete without recalibration.
4. **Target Variable Top-Coding and Truncation**:
   - `capital-gain` is artificially capped at \$99,999 by the US Census Bureau to prevent extreme individual disclosure, distorting true high-end investment distributions.
   - The income label is strictly binary (`<=50K` vs `>50K`), collapsing complex, continuous income distributions into a single arbitrary threshold.

## 6. Privacy and Ethical Concerns
- **Quasi-Identifiers and Re-identification Risk**: Although direct personal identifiers (e.g., full names, Social Security Numbers, exact home addresses) were removed by Kohavi and Becker, the combination of multiple high-cardinality quasi-identifiers (`age`, `education-num`, `occupation`, `native-country`, `fnlwgt`) poses potential linkage risks against external historical public voter or credit records (census reconstruction attacks).
- **Sampling Weight (`fnlwgt`) Disclosures**: `fnlwgt` represents the number of individuals in the population that a given record represents, derived from stratified demographic controls. Using it as a raw predictive feature can inadvertently leak demographic population estimates.
- **Fairness and Protected Attributes**: Because `sex`, `race`, and `age` are explicit features, models trained on this dataset must be evaluated with fairness-aware metrics (e.g., disparate impact, equalized odds) to avoid amplifying historical discrimination.

## 7. Data Quality and Cryptographic Verification
- **Ingestion Validation**: 48,842 rows x 15 columns verified.
- **Range Audit**: Zero range violations across numeric features (`age`, `hours-per-week`, `capital-gain`, `capital-loss`).
- **Duplicate-Conflict Audit**: Identified 5 distinct feature tuples (10 total rows) where identical individuals have contradictory income labels (`<=50K` vs `>50K`), establishing a non-zero Bayes error floor.
- **SHA-256 Checksums for Data Artifacts**:
  - `train_processed.csv`: `6029a8c6ff9c1721edf7289ab027097b1743ffa46ad598a97bb561cec0315e97`
  - `test_processed.csv`: `f5fb9e6b1e70cdbaddbff1050b664fc989e887d355ca737ea07fd11409473c30`
  - `preprocessor.joblib`: `4b47bb052b5dc83a95c798a50debe7377d68fc42e5e3efa1d711639df42ed9ed`
  - `config.yaml`: `0e82c5ff0a36e95bfefee050d276b5d985223ae3bcf41cf0bc973c9fba75f73c`
