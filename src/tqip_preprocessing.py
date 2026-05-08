from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vap_tqip.cleaning import (
    add_verification_level_columns,
    apply_cohort_exclusions,
    drop_original_encoded_columns,
    harmonize_binary_variables,
    normalize_temperature_to_celsius,
    one_hot_encode_features,
    process_ais_features,
    validate_required_columns,
)
from vap_tqip.constants import AIS_COLUMNS, OHE_RENAME_MAP
from vap_tqip.plotting import (
    plot_categorical_feature_distributions,
    plot_numeric_feature_distributions,
    plot_patient_count_by_year,
    plot_temperature_distribution,
    plot_withdrawal_lst_by_year,
)

REQUESTED_CATEGORICAL_FEATURES = [
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

REQUESTED_NUMERIC_FEATURES = [
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


def apply_ordered_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    return apply_cohort_exclusions(df)


def recode_binary_variables(df: pd.DataFrame):
    return harmonize_binary_variables(df)


def normalize_temperature_celsius(df: pd.DataFrame, feature: str = "TEMPERATURE", fahrenheit_threshold: float = 45.0):
    return normalize_temperature_to_celsius(df, column=feature, fahrenheit_threshold=fahrenheit_threshold)


def recode_ais_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    return process_ais_features(df, ais_cols=AIS_COLUMNS, ais_thresh=3)


def add_one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    work = one_hot_encode_features(df, columns_to_encode=["SEX", "HMRRHGCTRLSURGTYPE"], rename_map=OHE_RENAME_MAP)
    return add_verification_level_columns(work)


def prepare_final_export(df: pd.DataFrame) -> pd.DataFrame:
    return drop_original_encoded_columns(df)


def generate_exploratory_figures(df: pd.DataFrame, out_dir: Path) -> dict[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_patient_count_by_year(df)
    fig.savefig(out_dir / "patient_count_by_year.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, _ = plot_withdrawal_lst_by_year(df)
    fig.savefig(out_dir / "withdrawal_lst_by_year.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, _ = plot_temperature_distribution(df)
    fig.savefig(out_dir / "temperature_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    numeric_used = [f for f in REQUESTED_NUMERIC_FEATURES if f in df.columns]
    categorical_used = [f for f in REQUESTED_CATEGORICAL_FEATURES if f in df.columns]
    numeric_missing = [f for f in REQUESTED_NUMERIC_FEATURES if f not in df.columns]
    categorical_missing = [f for f in REQUESTED_CATEGORICAL_FEATURES if f not in df.columns]

    fig, _ = plot_numeric_feature_distributions(df, numeric_used)
    fig.savefig(out_dir / "numeric_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, _ = plot_categorical_feature_distributions(df, categorical_used)
    fig.savefig(out_dir / "categorical_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "numeric_missing": numeric_missing,
        "categorical_missing": categorical_missing,
        "numeric_used": numeric_used,
        "categorical_used": categorical_used,
    }
