"""Load raw TQIP data, run preprocessing + feature engineering, generate figures, and export vap_ohe.csv."""

from pathlib import Path
import pandas as pd

from src.tqip_preprocessing import (
    add_one_hot_features,
    apply_ordered_exclusions,
    generate_exploratory_figures,
    normalize_temperature_celsius,
    prepare_final_export,
    recode_ais_and_derive,
    recode_binary_variables,
    validate_required_columns,
)


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:,.3f}".format)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "VAP_TQIP_2017-2022.xlsx"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "vap_ohe.csv"
OUTPUT_FIG_DIR = BASE_DIR / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _count(label: str, n: int, denom: int | None = None) -> None:
    if denom is not None and denom > 0:
        pct = n / denom * 100
        print(f"  {label}: {n:,} ({pct:.2f}%)")
    else:
        print(f"  {label}: {n:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    vap = pd.read_excel(INPUT_FILE)
    raw_n = len(vap)

    _section("LOAD RAW DATA")
    _count("Total patients in raw dataset", raw_n)

    _section("VALIDATE REQUIRED COLUMNS")
    validate_required_columns(vap)
    print("  Required columns are present.")

    _section("PATIENT COUNTS BY DISCHARGE YEAR (raw)")
    year_counts = vap["YODISCH"].value_counts(dropna=False).sort_index()
    for year, count in year_counts.items():
        label = f"Year {int(year)}" if pd.notna(year) else "Year missing"
        _count(label, int(count))

    _section("ORDERED EXCLUSIONS")
    vap, exclusion_steps = apply_ordered_exclusions(vap)
    for step in exclusion_steps:
        _count(f"Excluded ({step['step']})", step["removed"])
        _count(f"Remaining after {step['step']}", step["remaining"], raw_n)

    _section("FINAL COHORT AFTER EXCLUSIONS")
    _count("Final cohort size", len(vap), raw_n)

    _section("BINARY HARMONIZATION")
    vap, recode_audit = recode_binary_variables(vap)
    for col, audit in recode_audit.items():
        print(f"\n  {col} before: {audit['before']}")
        print(f"  {col} after : {audit['after']}")

    _section("TEMPERATURE NORMALIZATION")
    vap, converted_n = normalize_temperature_celsius(vap)
    _count("Likely Fahrenheit temperatures converted to Celsius", converted_n, len(vap))

    _section("AIS HANDLING AND DERIVED INJURY FEATURES")
    vap = recode_ais_and_derive(vap)
    print("  AIS unknown code 9 recoded to NaN; injury/severity indicators generated.")

    _section("ONE-HOT ENCODING")
    vap_engineered = add_one_hot_features(vap)

    _section("GENERATE EXPLORATORY FIGURES")
    figure_audit = generate_exploratory_figures(vap_engineered, OUTPUT_FIG_DIR)
    print(f"  Saved figures to: {OUTPUT_FIG_DIR}")
    if figure_audit["numeric_missing"]:
        print(f"  Numeric features missing from this run (skipped in plots): {figure_audit['numeric_missing']}")
    if figure_audit["categorical_missing"]:
        print(f"  Categorical features missing from this run (skipped in plots): {figure_audit['categorical_missing']}")

    vap_ohe = prepare_final_export(vap_engineered)
    print(f"  Final feature count for export: {vap_ohe.shape[1]}")

    _section("WRITE OUTPUT CSV")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    vap_ohe.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved cleaned dataset: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
