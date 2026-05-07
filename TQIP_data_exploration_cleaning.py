"""
TQIP_data_exploration_cleaning.py

Purpose
-------
Load raw TQIP VAP data, summarize the starting cohort, apply initial cohort
restrictions, generate exploratory figures, and prepare a cleaned dataset for
downstream analysis.

Workflow
--------
1. Load the raw Excel input file from data/raw/.
2. Print the total number of patients in the raw dataset.
3. Print patient counts by discharge year (YODISCH), clearly labeled.
4. Report the number and percentage of patients missing YODISCH, then exclude
   those rows and print the remaining count.
5. Apply cohort exclusions in order, printing total patients remaining after
   each step:
     a. Exclude patients with missing AGEyears.
     b. Exclude patients with AGEyears < 18 (pediatric patients).
     c. Exclude patients with riss < 16 (ISS < 16).
6. Generate an exploratory histogram of patient count by discharge year and
   save it to figures/.
7. Write the cleaned, analysis-ready dataset to data/processed/vap_ohe.csv.

Inputs
------
  data/raw/VAP_TQIP_2017-2022.xlsx  -- raw TQIP VAP extract

Outputs
-------
  figures/patient_count_by_year.png  -- exploratory figure
  data/processed/vap_ohe.csv         -- cleaned dataset for downstream analysis
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:,.3f}".format)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "VAP_TQIP_2017-2022.xlsx"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "vap_ohe.csv"
OUTPUT_FIG = BASE_DIR / "figures" / "patient_count_by_year.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    """Print a clearly labelled section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _count(label: str, n: int, denom: int | None = None) -> None:
    """Print a labelled count, optionally with a percentage of *denom*."""
    if denom is not None and denom > 0:
        pct = n / denom * 100
        print(f"  {label}: {n:,} ({pct:.2f}%)")
    else:
        print(f"  {label}: {n:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # -- 1. Load raw data ----------------------------------------------------
    _section("LOAD RAW DATA")
    vap = pd.read_excel(INPUT_FILE)
    raw_n = len(vap)
    _count("Total patients in raw dataset", raw_n)

    # -- 2. Patient counts by discharge year (before exclusions) -------------
    _section("PATIENT COUNTS BY DISCHARGE YEAR (raw)")
    year_counts = vap["YODISCH"].value_counts(dropna=False).sort_index()
    for year, count in year_counts.items():
        label = f"Year {int(year)}" if pd.notna(year) else "Year missing"
        _count(label, int(count))

    # -- 3. Missing YODISCH --------------------------------------------------
    _section("MISSING DISCHARGE YEAR (YODISCH)")
    missing_yodisch = vap["YODISCH"].isna().sum()
    _count("Patients missing YODISCH", int(missing_yodisch), raw_n)

    vap = vap.loc[~vap["YODISCH"].isna()].copy()
    _count("Patients remaining after excluding missing YODISCH", len(vap), raw_n)

    # -- 4. Cohort exclusions ------------------------------------------------
    _section("COHORT EXCLUSIONS")

    # 4a. Missing AGEyears
    missing_age = vap["AGEyears"].isna().sum()
    _count("Patients with missing AGEyears (excluded)", int(missing_age), len(vap))
    vap = vap.loc[vap["AGEyears"].notna()].copy()
    _count("Patients remaining after excluding missing AGEyears", len(vap), raw_n)

    # 4b. Age < 18 (pediatric)
    under_18 = (vap["AGEyears"] < 18).sum()
    _count("Patients with AGEyears < 18 (excluded, pediatric)", int(under_18), len(vap))
    vap = vap.loc[vap["AGEyears"] >= 18].copy()
    _count("Patients remaining after excluding age < 18", len(vap), raw_n)

    # 4c. rISS < 16
    low_iss = (vap["riss"] < 16).sum()
    _count("Patients with riss < 16 (excluded)", int(low_iss), len(vap))
    vap = vap.loc[vap["riss"] >= 16].copy()
    _count("Patients remaining after excluding riss < 16", len(vap), raw_n)

    _section("FINAL COHORT AFTER EXCLUSIONS")
    _count("Final cohort size", len(vap), raw_n)

    # -- 5. Exploratory figure -----------------------------------------------
    _section("GENERATE EXPLORATORY FIGURE")
    fig, ax = plt.subplots(figsize=(8, 5))
    vap["YODISCH"].hist(ax=ax, bins=len(vap["YODISCH"].unique()))
    ax.set_title("TQIP Patient Count by Discharge Year")
    ax.set_xlabel("Discharge Year")
    ax.set_ylabel("Patient Count")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    fig.tight_layout()

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    print(f"  Saved figure: {OUTPUT_FIG}")
    plt.close(fig)

    # -- 6. Write cleaned CSV ------------------------------------------------
    _section("WRITE OUTPUT CSV")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    vap.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved cleaned dataset: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
