from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

AIS_COLUMNS = [
    "mxaisbr_HeadNeck",
    "mxaisbr_Face",
    "mxaisbr_Chest",
    "mxaisbr_Abdomen",
    "mxaisbr_Extremities",
]

REQUIRED_COLUMNS = [
    "YODISCH",
    "AGEyears",
    "riss",
    "ETHNICITY",
    "SUPPLEMENTALOXYGEN",
    "RESPIRATORYASSISTANCE",
    "WITHDRAWALLST",
    "PREHOSPITALCARDIACARREST",
    "HC_RESPIRATORY",
    "TEMPERATURE",
    "SEX",
    "HMRRHGCTRLSURGTYPE",
    "VERIFICATIONLEVEL",
    *AIS_COLUMNS,
]

OHE_RENAME_MAP = {
    "SEX_1.0": "sex_male",
    "SEX_2.0": "sex_female",
    "SEX_3.0": "sex_non-binary",
    "HMRRHGCTRLSURGTYPE_1.0": "surg_none",
    "HMRRHGCTRLSURGTYPE_2.0": "surg_laparotomy",
    "HMRRHGCTRLSURGTYPE_3.0": "surg_thoracotomy",
    "HMRRHGCTRLSURGTYPE_4.0": "surg_sternotomy",
    "HMRRHGCTRLSURGTYPE_5.0": "surg_extremity",
    "HMRRHGCTRLSURGTYPE_6.0": "surg_neck",
    "HMRRHGCTRLSURGTYPE_7.0": "surg_amputation",
    "HMRRHGCTRLSURGTYPE_8.0": "surg_skin_softtissue",
    "HMRRHGCTRLSURGTYPE_9.0": "surg_pelvic_packing",
}

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

DERIVED_FEATURES = [
    "severe_HeadNeck",
    "severe_Face",
    "severe_Chest",
    "severe_Abdomen",
    "severe_Extremities",
    "L_unkn",
    "HC_RESPIRATORY",
]


def validate_required_columns(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    required = required_columns if required_columns is not None else REQUIRED_COLUMNS
    missing = sorted([col for col in required if col not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def apply_ordered_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    work = df.copy()
    steps: list[dict[str, int]] = []

    for label, mask in [
        ("missing YODISCH", work["YODISCH"].isna()),
        ("missing AGEyears", work["AGEyears"].isna()),
        ("AGEyears < 18", work["AGEyears"] < 18),
        ("riss < 16", work["riss"] < 16),
    ]:
        removed = int(mask.sum())
        work = work.loc[~mask].copy()
        steps.append({"step": label, "removed": removed, "remaining": int(len(work))})

    return work, steps


def recode_binary_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, dict[object, int]]]]:
    work = df.copy()
    cols = [
        "ETHNICITY",
        "SUPPLEMENTALOXYGEN",
        "RESPIRATORYASSISTANCE",
        "WITHDRAWALLST",
        "PREHOSPITALCARDIACARREST",
        "HC_RESPIRATORY",
    ]
    audit: dict[str, dict[str, dict[object, int]]] = {}

    for col in cols:
        audit[col] = {"before": work[col].value_counts(dropna=False).to_dict()}

    work.loc[work["ETHNICITY"] == 2, "ETHNICITY"] = 0
    work.loc[work["SUPPLEMENTALOXYGEN"] == 1, "SUPPLEMENTALOXYGEN"] = 0
    work.loc[work["SUPPLEMENTALOXYGEN"] == 2, "SUPPLEMENTALOXYGEN"] = 1
    work.loc[work["RESPIRATORYASSISTANCE"] == 1, "RESPIRATORYASSISTANCE"] = 0
    work.loc[work["RESPIRATORYASSISTANCE"] == 2, "RESPIRATORYASSISTANCE"] = 1
    work.loc[work["WITHDRAWALLST"] == 2, "WITHDRAWALLST"] = 0
    work.loc[work["PREHOSPITALCARDIACARREST"] == 2, "PREHOSPITALCARDIACARREST"] = 0

    for col in cols:
        audit[col]["after"] = work[col].value_counts(dropna=False).to_dict()

    return work, audit


def normalize_temperature_celsius(
    df: pd.DataFrame, feature: str = "TEMPERATURE", fahrenheit_threshold: float = 45.0, max_fahrenheit: float = 120.0
) -> tuple[pd.DataFrame, int]:
    work = df.copy()
    mask = work[feature].notna() & (work[feature] > fahrenheit_threshold) & (work[feature] <= max_fahrenheit)
    converted_n = int(mask.sum())
    work.loc[mask, feature] = (work.loc[mask, feature] - 32) * (5.0 / 9.0)
    return work, converted_n


def recode_ais_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in AIS_COLUMNS:
        work.loc[work[col] == 9, col] = np.nan
        region = col.replace("mxaisbr_", "")
        work[f"inj_{region}"] = ((work[col] > 0) & work[col].notna()).astype(int)
        work[f"severe_{region}"] = ((work[col] >= 3) & work[col].notna()).astype(int)
    return work


def add_one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    to_encode = ["SEX", "HMRRHGCTRLSURGTYPE"]
    for col in to_encode:
        work[col] = work[col].astype(float)

    vap_ohe = pd.get_dummies(data=work, prefix=to_encode, columns=to_encode, dtype=int)
    vap_ohe = vap_ohe.rename(columns=OHE_RENAME_MAP)

    vap_ohe["L1"] = (vap_ohe["VERIFICATIONLEVEL"] == 1).astype(int)
    vap_ohe["L2"] = (vap_ohe["VERIFICATIONLEVEL"] == 2).astype(int)
    vap_ohe["L3"] = (vap_ohe["VERIFICATIONLEVEL"] == 3).astype(int)
    vap_ohe["L_unkn"] = vap_ohe["VERIFICATIONLEVEL"].isna().astype(int)
    return vap_ohe


def prepare_final_export(df: pd.DataFrame) -> pd.DataFrame:
    requested = REQUESTED_CATEGORICAL_FEATURES + REQUESTED_NUMERIC_FEATURES + DERIVED_FEATURES
    keep_columns = [col for col in requested if col in df.columns and col not in {"SEX", "HMRRHGCTRLSURGTYPE", "VERIFICATIONLEVEL"}]
    return df[keep_columns].copy()


def _select_plot_features(df: pd.DataFrame, requested: list[str]) -> tuple[list[str], list[str]]:
    present = [col for col in requested if col in df.columns]
    missing = [col for col in requested if col not in df.columns]
    return present, missing


def plot_patient_count_by_year(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    year_counts = df["YODISCH"].dropna().astype(int).value_counts().sort_index()
    year_counts.plot(kind="bar", ax=ax)
    ax.set_title("TQIP Patient Count by Discharge Year")
    ax.set_xlabel("Discharge Year")
    ax.set_ylabel("Patient Count")
    fig.tight_layout()
    fig.savefig(out_dir / "patient_count_by_year.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_withdrawal_lst_by_year(df: pd.DataFrame, out_dir: Path) -> None:
    years = [2017, 2018, 2019, 2020, 2021, 2022]
    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(6, 8), dpi=120)
    ax = ax.ravel()
    for i, yr in enumerate(years):
        year_slice = df.loc[df["YODISCH"] == yr, "WITHDRAWALLST"]
        sns.histplot(x=year_slice, bins=3, ax=ax[i], discrete=True)
        ax[i].set_title(str(yr), fontsize=8)
        ax[i].set_xticks([0, 1, 2])
    fig.suptitle("TQIP Withdrawal of LST Values by Year")
    fig.tight_layout()
    fig.savefig(out_dir / "withdrawal_lst_by_year.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=120)
    sns.histplot(data=df, x="TEMPERATURE", bins=200, ax=ax[0])
    sns.histplot(data=df, x="TEMPERATURE", bins=200, ax=ax[1])
    ax[1].set_ylim(0, 100)
    ax[0].set_title("Temperature Distribution")
    ax[1].set_title("Temperature (Zoomed y-axis)")
    fig.tight_layout()
    fig.savefig(out_dir / "temperature_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_grid(df: pd.DataFrame, features: list[str], out_path: Path, title: str, categorical: bool = False) -> None:
    if not features:
        return

    n_cols = 6
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(3, n_rows * 2.5)))
    axes = np.array(axes).ravel()

    for i, feat in enumerate(features):
        if categorical:
            vc = df[feat].value_counts(dropna=False).sort_index()
            vc.plot(kind="bar", ax=axes[i])
        else:
            sns.histplot(data=df, x=feat, ax=axes[i], bins=40)
        axes[i].tick_params(axis="both", which="both", labelsize=7)
        axes[i].set_xlabel(feat, fontsize=8)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_exploratory_figures(df: pd.DataFrame, out_dir: Path) -> dict[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_patient_count_by_year(df, out_dir)
    plot_withdrawal_lst_by_year(df, out_dir)
    plot_temperature_distribution(df, out_dir)

    numeric_present, numeric_missing = _select_plot_features(df, REQUESTED_NUMERIC_FEATURES)
    categorical_present, categorical_missing = _select_plot_features(df, REQUESTED_CATEGORICAL_FEATURES)

    _plot_feature_grid(
        df=df,
        features=numeric_present,
        out_path=out_dir / "numeric_feature_distributions.png",
        title="Numeric Feature Distributions",
        categorical=False,
    )
    _plot_feature_grid(
        df=df,
        features=categorical_present,
        out_path=out_dir / "categorical_feature_distributions.png",
        title="Categorical Feature Distributions",
        categorical=True,
    )
    return {
        "numeric_missing": numeric_missing,
        "categorical_missing": categorical_missing,
        "numeric_used": numeric_present,
        "categorical_used": categorical_present,
    }
