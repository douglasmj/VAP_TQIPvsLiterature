# VAP_TQIPvsLiterature

Comparison of ventilator-associated pneumonia (VAP) outcomes from the
Trauma Quality Improvement Program (TQIP) against published literature
estimates.

---

## Repository layout

```
VAP_TQIPvsLiterature/
├── pyproject.toml                         # package configuration — makes vap_tqip installable
├── TQIP_data_exploration_cleaning.py      # thin end-to-end script entry point
├── src/
│   └── vap_tqip/                          # installable package
│       ├── __init__.py                    # re-exports all user-facing names
│       ├── cleaning.py                    # cleaning + feature engineering functions
│       ├── plotting.py                    # exploratory plotting functions
│       └── constants.py                   # AIS columns, feature lists, rename maps
├── notebooks/
│   └── tqip_preprocessing_example.ipynb   # step-by-step notebook walkthrough
├── tests/
│   └── test_tqip_preprocessing.py         # unit tests for preprocessing logic
├── requirements.txt                       # Python dependencies
├── data/
│   ├── raw/                               # place raw input files here (not tracked)
│   └── processed/                         # cleaned outputs written here (not tracked)
└── figures/                               # exploratory figures written here (not tracked)
```

---

## Installation

### Install the package locally (required before running scripts or notebooks)

```bash
pip install -e .
```

This installs `vap_tqip` in editable mode so it is importable from anywhere in
the same Python environment, including from notebooks outside the repo directory.

### Install with development extras (tests and notebook support)

```bash
pip install -e ".[dev]"
```

### Legacy dependency install

```bash
pip install -r requirements.txt
```

---

## Using the package in a notebook

Once installed, import directly from `vap_tqip`:

```python
from vap_tqip.cleaning import (
    load_tqip_data,
    validate_required_columns,
    summarize_raw_cohort,
    apply_cohort_exclusions,
    harmonize_binary_variables,
    normalize_temperature_to_celsius,
    process_ais_features,
    one_hot_encode_features,
    add_verification_level_columns,
    drop_original_encoded_columns,
)
from vap_tqip.constants import AIS_COLUMNS, OHE_RENAME_MAP
from vap_tqip.plotting import plot_patient_count_by_year
```

Or import everything from the top-level package:

```python
from vap_tqip import load_tqip_data, apply_cohort_exclusions
```

Because the package is installed in your environment, these imports work from
any notebook — even notebooks stored outside the repository folder — as long as
the same environment has `vap_tqip` installed.

---

## Expected input data

Place the raw TQIP VAP extract at:

```
data/raw/VAP_TQIP_2017-2022.xlsx
```

The file is a standard TQIP export covering years 2017–2022.

---

## Recommended function order (notebook-friendly)

The preprocessing utilities are designed for stepwise notebook use. Typical call order:

1. `load_tqip_data(path)`
2. `validate_required_columns(df)`
3. `summarize_raw_cohort(df)`
4. `apply_cohort_exclusions(df, adult_age=18, min_riss=16)`
   - missing `YODISCH`
   - missing `AGEyears`
   - `AGEyears < 18`
   - `riss < 16`
5. `harmonize_binary_variables(df)`
   - `ETHNICITY`: `2 -> 0`
   - `SUPPLEMENTALOXYGEN`: `1 -> 0`, `2 -> 1`
   - `RESPIRATORYASSISTANCE`: `1 -> 0`, `2 -> 1`
   - `WITHDRAWALLST`: `2 -> 0`
   - `PREHOSPITALCARDIACARREST`: `2 -> 0`
   - `HC_RESPIRATORY`: unchanged
6. `normalize_temperature_to_celsius(df)`
7. `process_ais_features(df, ais_cols, ais_thresh=3)`
   - keep AIS body-region columns ordinal
   - recode AIS `9 -> NaN`
   - add `inj_*` for AIS `> 0`
   - add `severe_*` for AIS `>= 3`
8. `one_hot_encode_features(df, ["SEX", "HMRRHGCTRLSURGTYPE"], rename_map=OHE_RENAME_MAP)`
9. `add_verification_level_columns(df)`
10. `drop_original_encoded_columns(df, ["SEX", "HMRRHGCTRLSURGTYPE", "VERIFICATIONLEVEL"])`

`build_vap_ohe_dataset(df, ...)` is also available as a convenience wrapper.

---

## Script workflow

`TQIP_data_exploration_cleaning.py` keeps a thin orchestration layer and runs the same preprocessing and plotting steps end-to-end, including:

- clear required-column validation (`HC_PNEUMONIA` included for downstream modeling use)
- ordered exclusions with remaining patient counts printed after each step
- binary harmonization audits
- temperature sanity-check conversion to Celsius
- AIS unknown handling (`9 -> NaN`) + injury/severity derived columns
- one-hot encoding of `SEX` and `HMRRHGCTRLSURGTYPE` with readable column names
- trauma center verification derived columns (`L1`, `L2`, `L3`, `L_unkn`)
- dropping raw encoded source columns from export
- exploratory figure generation
- final export to `data/processed/vap_ohe.csv`

---

## Notebook usage

See `notebooks/tqip_preprocessing_example.ipynb` for a notebook-oriented workflow that demonstrates:

- loading and validating raw data
- summarizing cohort size and year counts
- applying exclusions and reviewing remaining counts
- harmonizing variables and checking value counts
- engineering AIS and encoded features
- generating exploratory figures
- inspecting/saving the final dataframe

---

## Outputs

| Path                                         | Description                                    |
|----------------------------------------------|------------------------------------------------|
| `figures/patient_count_by_year.png`          | Patient count by discharge year                |
| `figures/withdrawal_lst_by_year.png`         | WITHDRAWALLST histograms split by year         |
| `figures/temperature_distribution.png`       | Temperature distribution exploration            |
| `figures/numeric_feature_distributions.png`  | Numeric feature distribution grid               |
| `figures/categorical_feature_distributions.png` | Categorical feature distribution grid        |
| `data/processed/vap_ohe.csv`                 | Harmonized + engineered analysis-ready dataset |

---

## How to run

### 1. Install the package

```bash
pip install -e .
```

### 2. Place raw data

Copy the raw TQIP extract to `data/raw/VAP_TQIP_2017-2022.xlsx`.

### 3. Run the script

```bash
python TQIP_data_exploration_cleaning.py
```

### 4. Run tests

```bash
pytest -q
```

