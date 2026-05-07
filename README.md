# VAP_TQIPvsLiterature

Comparison of ventilator-associated pneumonia (VAP) outcomes from the
Trauma Quality Improvement Program (TQIP) against published literature
estimates.

---

## Repository layout

```
VAP_TQIPvsLiterature/
├── TQIP_data_exploration_cleaning.py   # data-cleaning and exploration script
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

The file is a standard TQIP export covering years 2017–2022 and must contain
at least the following columns:

| Column     | Description                         |
|------------|-------------------------------------|
| `YODISCH`  | Year of discharge                   |
| `AGEyears` | Patient age in years                |
| `riss`     | Revised Injury Severity Score (ISS) |

---

## Workflow

`TQIP_data_exploration_cleaning.py` runs the following steps:

1. **Load** the raw Excel file and print the total patient count.
2. **Summarise** patient counts by discharge year (`YODISCH`).
3. **Exclude** patients missing `YODISCH`, reporting count and percentage.
4. **Apply cohort exclusions** in order, printing patients remaining after each:
   - Exclude patients with missing `AGEyears`.
   - Exclude patients with `AGEyears < 18` (pediatric).
   - Exclude patients with `riss < 16` (ISS < 16).
5. **Generate** an exploratory histogram of patient count by discharge year,
   saved to `figures/patient_count_by_year.png`.
6. **Export** the cleaned dataset to `data/processed/vap_ohe.csv` for
   downstream analysis.

---

## Outputs

| Path                             | Description                                    |
|----------------------------------|------------------------------------------------|
| `figures/patient_count_by_year.png` | Histogram: patient count by discharge year  |
| `data/processed/vap_ohe.csv`     | Cleaned, analysis-ready dataset                |

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

The script will print a labelled summary of cohort counts at each exclusion
step and write the figure and CSV to the locations listed above.
