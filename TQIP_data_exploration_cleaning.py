"""Load raw TQIP data, run notebook-friendly preprocessing steps, generate figures, and export vap_ohe.csv."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.cleaning_utils import (
    AIS_COLUMNS,
    OHE_RENAME_MAP,
    add_verification_level_columns,
    apply_cohort_exclusions,
    drop_original_encoded_columns,
    harmonize_binary_variables,
    load_tqip_data,
    normalize_temperature_to_celsius,
    one_hot_encode_features,
    print_value_counts,
    process_ais_features,
    summarize_raw_cohort,
    validate_required_columns,
)
from src.plotting_utils import (
    plot_categorical_feature_distributions,
    plot_numeric_feature_distributions,
    plot_patient_count_by_year,
    plot_temperature_distribution,
    plot_withdrawal_lst_by_year,
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

CAT_FEATURES = [
    "HC_VAPNEUMONIA",
    "WHITE",
    "BLACK",
    "ETHNICITY",
    "PACIFICISLANDER",
    "AMERICANINDIAN",
    "ASIAN",
    "RACEOTHER",
    "WITHDRAWALLST",
    "INTERFACILITYTRANSFER",
    "RESPIRATORYASSISTANCE",
    "SUPPLEMENTALOXYGEN",
    "PREHOSPITALCARDIACARREST",
    "VERIFICATIONLEVEL",
    "Mortality",
    "CC_CHEMO",
    "CC_CIRRHOSIS",
    "CC_COPD",
    "CC_CVA",
    "CC_DIABETES",
    "CC_DISCANCER",
    "CC_FUNCTIONAL",
    "CC_CHF",
    "CC_RENAL",
    "CC_SMOKING",
    "ICP_Monitor",
    "sex_male",
    "sex_female",
    "sex_non-binary",
    "surg_none",
    "surg_laparotomy",
    "surg_thoracotomy",
    "surg_sternotomy",
    "surg_extremity",
    "surg_neck",
    "surg_amputation",
    "surg_skin_softtissue",
    "surg_pelvic_packing",
    "inj_HeadNeck",
    "inj_Face",
    "inj_Chest",
    "inj_Abdomen",
    "inj_Extremities",
    "L1",
    "L2",
    "L3",
]

NUM_FEATURES = [
    "TOTALVENTDAYS",
    "AGEyears",
    "HMRRHGCTRLSURGMins",
    "HMRRHGCTRLSURGDays",
    "WITHDRAWALLSTMins",
    "WITHDRAWALLSTDays",
    "TOTALICULOS",
    "Hospital_LOS_Hr",
    "Hospital_LOS_Days",
    "riss",
    "mxaisbr_HeadNeck",
    "mxaisbr_Face",
    "mxaisbr_Chest",
    "mxaisbr_Abdomen",
    "mxaisbr_Extremities",
    "SBP",
    "PULSERATE",
    "TEMPERATURE",
    "RESPIRATORYRATE",
    "PULSEOXIMETRY",
    "PRBC_4",
    "FFP_4",
    "PLT_4",
    "WB",
    "WB_time_mins",
]


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
    vap = load_tqip_data(str(INPUT_FILE))
    cohort_summary = summarize_raw_cohort(vap)
    raw_n = int(cohort_summary["total_patients"])

    _section("LOAD RAW DATA")
    _count("Total patients in raw dataset", raw_n)

    _section("VALIDATE REQUIRED COLUMNS")
    validate_required_columns(vap)
    print("  Required columns are present.")

    _section("PATIENT COUNTS BY DISCHARGE YEAR (raw)")
    year_counts = cohort_summary["year_counts"]
    for year, count in year_counts.items():
        label = f"Year {int(year)}" if pd.notna(year) else "Year missing"
        _count(label, int(count))

    _section("ORDERED EXCLUSIONS")
    vap, exclusion_steps = apply_cohort_exclusions(vap)
    for step in exclusion_steps:
        _count(f"Excluded ({step['step']})", step["removed"])
        _count(f"Remaining after {step['step']}", step["remaining"], raw_n)

    _section("FINAL COHORT AFTER EXCLUSIONS")
    _count("Final cohort size", len(vap), raw_n)

    _section("BINARY HARMONIZATION")
    print_value_counts(
        vap,
        [
            "ETHNICITY",
            "SUPPLEMENTALOXYGEN",
            "RESPIRATORYASSISTANCE",
            "WITHDRAWALLST",
            "PREHOSPITALCARDIACARREST",
            "HC_RESPIRATORY",
        ],
        title="  Before recoding:",
    )
    vap, recode_audit = harmonize_binary_variables(vap)
    for col, audit in recode_audit.items():
        print(f"\n  {col} before: {audit['before']}")
        print(f"  {col} after : {audit['after']}")

    _section("TEMPERATURE NORMALIZATION")
    vap, converted_n = normalize_temperature_to_celsius(vap)
    _count("Likely Fahrenheit temperatures converted to Celsius", converted_n, len(vap))

    _section("AIS HANDLING AND DERIVED INJURY FEATURES")
    vap = process_ais_features(vap, ais_cols=AIS_COLUMNS, ais_thresh=3)
    print("  AIS unknown code 9 recoded to NaN; injury/severity indicators generated.")

    _section("ONE-HOT ENCODING")
    vap_engineered = one_hot_encode_features(
        vap,
        columns_to_encode=["SEX", "HMRRHGCTRLSURGTYPE"],
        rename_map=OHE_RENAME_MAP,
    )
    vap_engineered = add_verification_level_columns(vap_engineered)
    vap_ohe = drop_original_encoded_columns(vap_engineered, columns=["SEX", "HMRRHGCTRLSURGTYPE", "VERIFICATIONLEVEL"])
    print(f"  Final feature count for export: {vap_ohe.shape[1]}")

    _section("GENERATE EXPLORATORY FIGURES")
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_patient_count_by_year(vap_engineered)
    fig.savefig(OUTPUT_FIG_DIR / "patient_count_by_year.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_withdrawal_lst_by_year(vap_engineered)
    fig.savefig(OUTPUT_FIG_DIR / "withdrawal_lst_by_year.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_temperature_distribution(vap_engineered)
    fig.savefig(OUTPUT_FIG_DIR / "temperature_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_numeric_feature_distributions(vap_engineered, NUM_FEATURES)
    fig.savefig(OUTPUT_FIG_DIR / "numeric_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_categorical_feature_distributions(vap_engineered, CAT_FEATURES)
    fig.savefig(OUTPUT_FIG_DIR / "categorical_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved figures to: {OUTPUT_FIG_DIR}")

    _section("WRITE OUTPUT CSV")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    vap_ohe.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved cleaned dataset: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
