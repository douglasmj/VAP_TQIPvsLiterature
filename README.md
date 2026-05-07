# VAP_TQIPvsLiterature

Comparison of ventilator-associated pneumonia (VAP) outcomes from the
Trauma Quality Improvement Program (TQIP) against published literature
estimates.

---

## Repository layout

```
VAP_TQIPvsLiterature/
├── TQIP_data_exploration_cleaning.py   # data-cleaning and exploration script
├── src/
│   └── tqip_preprocessing.py           # reusable preprocessing helpers
├── tests/
│   └── test_tqip_preprocessing.py      # unit tests for preprocessing logic
├── requirements.txt                    # Python dependencies
├── data/
│   ├── raw/                            # place raw input files here (not tracked)
│   └── processed/                      # cleaned outputs written here (not tracked)
└── figures/                            # exploratory figures written here (not tracked)
```

---

## Expected input data

Place the raw TQIP VAP extract at:

```
data/raw/VAP_TQIP_2017-2022.xlsx
```

The file is a standard TQIP export covering years 2017–2022.

---

## Workflow

`TQIP_data_exploration_cleaning.py` runs the following preprocessing workflow:

1. **Validate required columns** and fail clearly if key columns are missing.
2. **Print cohort counts** (overall and by `YODISCH` discharge year).
3. **Apply exclusions in order**, printing remaining patients after each:
   - missing `YODISCH`
   - missing `AGEyears`
   - `AGEyears < 18`
   - `riss < 16`
4. **Harmonize binary variables** for cross-year consistency:
   - `ETHNICITY`: `2 -> 0` (`1 = Hispanic`, `2 = Not Hispanic`)
   - `SUPPLEMENTALOXYGEN`: `1 -> 0`, `2 -> 1` (`1 = No`, `2 = Yes`)
   - `RESPIRATORYASSISTANCE`: `1 -> 0`, `2 -> 1` (`1 = No`, `2 = Yes`)
   - `WITHDRAWALLST`: `2 -> 0` (harmonize shifted year-specific coding to binary)
   - `PREHOSPITALCARDIACARREST`: `2 -> 0` (same year-specific harmonization pattern)
   - `HC_RESPIRATORY`: unchanged (already 0/1)
   Value counts are printed before/after recoding for auditability.
5. **Normalize temperature units** using a sanity check:
   likely Fahrenheit values (`>45` and `<=120`) are converted to Celsius.
6. **Handle AIS fields**:
   - keep AIS scores numeric (not one-hot encoded)
   - recode `9` (unknown) to `NaN`
   - create region injury indicators (`inj_*`) where AIS `> 0`
   - create severe injury indicators (`severe_*`) where AIS `>= 3`
7. **Feature engineering / one-hot encoding**:
   - one-hot encode `SEX` and `HMRRHGCTRLSURGTYPE`
   - rename OHE columns using readable names from the TQIP mapping
   - derive trauma center verification OHE columns: `L1`, `L2`, `L3`, `L_unkn`
   - drop original encoded columns (`SEX`, `HMRRHGCTRLSURGTYPE`, `VERIFICATIONLEVEL`)
8. **Generate exploratory figures** in `figures/`:
   - patient count by discharge year
   - `WITHDRAWALLST` histograms split by year (2017–2022)
   - temperature distribution figure
   - numeric feature distribution grid
   - categorical feature distribution grid
9. **Write final model-ready output** to `data/processed/vap_ohe.csv`.

---

## Outputs

| Path                             | Description                                    |
|----------------------------------|------------------------------------------------|
| `figures/patient_count_by_year.png` | Patient count by discharge year             |
| `figures/withdrawal_lst_by_year.png` | WITHDRAWALLST histograms split by year      |
| `figures/temperature_distribution.png` | Temperature distribution exploration       |
| `figures/numeric_feature_distributions.png` | Numeric feature distribution grid      |
| `figures/categorical_feature_distributions.png` | Categorical feature distribution grid |
| `data/processed/vap_ohe.csv`     | Harmonized + engineered analysis-ready dataset |

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
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

The script prints labelled cohort/exclusion summaries, recoding audits, and
writes processed outputs and figures to the locations listed above.
