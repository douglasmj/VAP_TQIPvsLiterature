from __future__ import annotations

import numpy as np
import pandas as pd

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
    "HC_VAPNEUMONIA",
    "HC_PNEUMONIA",
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

RAW_ENCODED_COLUMNS_TO_DROP = ("SEX", "HMRRHGCTRLSURGTYPE", "VERIFICATIONLEVEL")


def load_tqip_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def validate_required_columns(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    required = REQUIRED_COLUMNS if required_columns is None else required_columns
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def summarize_raw_cohort(df: pd.DataFrame) -> dict[str, object]:
    return {
        "total_patients": int(len(df)),
        "year_counts": df["YODISCH"].value_counts(dropna=False).sort_index(),
    }


def exclude_missing_discharge_year(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = df["YODISCH"].isna()
    return df.loc[~mask].copy(), int(mask.sum())


def exclude_missing_age(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = df["AGEyears"].isna()
    return df.loc[~mask].copy(), int(mask.sum())


def exclude_underage_patients(df: pd.DataFrame, adult_age: int = 18) -> tuple[pd.DataFrame, int]:
    mask = df["AGEyears"] < adult_age
    return df.loc[~mask].copy(), int(mask.sum())


def exclude_low_riss(df: pd.DataFrame, min_riss: int = 16) -> tuple[pd.DataFrame, int]:
    mask = df["riss"] < min_riss
    return df.loc[~mask].copy(), int(mask.sum())


def apply_cohort_exclusions(df: pd.DataFrame, adult_age: int = 18, min_riss: int = 16) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    work = df.copy()
    steps: list[dict[str, int]] = []
    ordered_steps = [
        ("missing YODISCH", exclude_missing_discharge_year, {}),
        ("missing AGEyears", exclude_missing_age, {}),
        (f"AGEyears < {adult_age}", exclude_underage_patients, {"adult_age": adult_age}),
        (f"riss < {min_riss}", exclude_low_riss, {"min_riss": min_riss}),
    ]
    for label, fn, kwargs in ordered_steps:
        work, removed = fn(work, **kwargs)
        steps.append({"step": label, "removed": removed, "remaining": int(len(work))})
    return work, steps


def print_value_counts(df: pd.DataFrame, columns: list[str], title: str | None = None) -> dict[str, dict[object, int]]:
    if title:
        print(title)
    summary: dict[str, dict[object, int]] = {}
    for col in columns:
        counts = df[col].value_counts(dropna=False).to_dict()
        summary[col] = counts
        print(f"{col}: {counts}")
    return summary


def harmonize_binary_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, dict[object, int]]]]:
    work = df.copy()
    cols = [
        "ETHNICITY",
        "SUPPLEMENTALOXYGEN",
        "RESPIRATORYASSISTANCE",
        "WITHDRAWALLST",
        "PREHOSPITALCARDIACARREST",
        "HC_RESPIRATORY",
    ]
    audit = {col: {"before": work[col].value_counts(dropna=False).to_dict()} for col in cols}

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


def normalize_temperature_to_celsius(
    df: pd.DataFrame, column: str = "TEMPERATURE", fahrenheit_threshold: float = 45.0, max_fahrenheit: float = 120.0
) -> tuple[pd.DataFrame, int]:
    work = df.copy()
    mask = work[column].notna() & (work[column] > fahrenheit_threshold) & (work[column] <= max_fahrenheit)
    converted_n = int(mask.sum())
    work.loc[mask, column] = (work.loc[mask, column] - 32.0) * (5.0 / 9.0)
    return work, converted_n


def one_hot_encode_features(
    df: pd.DataFrame, columns_to_encode: list[str], rename_map: dict[str, str] | None = None
) -> pd.DataFrame:
    work = df.copy()
    for col in columns_to_encode:
        work[col] = work[col].astype(float)
    encoded_df = pd.get_dummies(work, columns=columns_to_encode, prefix=columns_to_encode, dtype=int)
    return encoded_df.rename(columns=rename_map or {})


def add_verification_level_columns(df: pd.DataFrame, source_col: str = "VERIFICATIONLEVEL") -> pd.DataFrame:
    work = df.copy()
    work["L1"] = (work[source_col] == 1).astype(int)
    work["L2"] = (work[source_col] == 2).astype(int)
    work["L3"] = (work[source_col] == 3).astype(int)
    work["L_unkn"] = work[source_col].isna().astype(int)
    return work


def recode_ais_unknowns(df: pd.DataFrame, ais_cols: list[str]) -> pd.DataFrame:
    work = df.copy()
    for col in ais_cols:
        work.loc[work[col] == 9, col] = np.nan
    return work


def add_any_injury_indicators(df: pd.DataFrame, ais_cols: list[str], region_map: dict[str, str] | None = None) -> pd.DataFrame:
    work = df.copy()
    regions = region_map or {col: col.replace("mxaisbr_", "") for col in ais_cols}
    for col in ais_cols:
        region = regions[col]
        work[f"inj_{region}"] = ((work[col] > 0) & work[col].notna()).astype(int)
    return work


def add_severe_injury_indicators(
    df: pd.DataFrame, ais_cols: list[str], ais_thresh: int = 3, region_map: dict[str, str] | None = None
) -> pd.DataFrame:
    work = df.copy()
    regions = region_map or {col: col.replace("mxaisbr_", "") for col in ais_cols}
    for col in ais_cols:
        region = regions[col]
        work[f"severe_{region}"] = ((work[col] >= ais_thresh) & work[col].notna()).astype(int)
    return work


def process_ais_features(df: pd.DataFrame, ais_cols: list[str], ais_thresh: int = 3) -> pd.DataFrame:
    work = recode_ais_unknowns(df, ais_cols)
    work = add_any_injury_indicators(work, ais_cols)
    work = add_severe_injury_indicators(work, ais_cols, ais_thresh=ais_thresh)
    return work


def drop_original_encoded_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols_to_drop = RAW_ENCODED_COLUMNS_TO_DROP if columns is None else tuple(columns)
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns]).copy()


def build_vap_ohe_dataset(
    df: pd.DataFrame,
    adult_age: int = 18,
    min_riss: int = 16,
    columns_to_encode: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
    ais_cols: list[str] | None = None,
    ais_thresh: int = 3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    validate_required_columns(df)
    summary = summarize_raw_cohort(df)
    work, exclusions = apply_cohort_exclusions(df, adult_age=adult_age, min_riss=min_riss)
    work, binary_audit = harmonize_binary_variables(work)
    work, converted_n = normalize_temperature_to_celsius(work)
    work = process_ais_features(work, ais_cols=ais_cols or AIS_COLUMNS, ais_thresh=ais_thresh)
    work = one_hot_encode_features(
        work,
        columns_to_encode=columns_to_encode or ["SEX", "HMRRHGCTRLSURGTYPE"],
        rename_map=rename_map or OHE_RENAME_MAP,
    )
    work = add_verification_level_columns(work)
    work = drop_original_encoded_columns(work)
    metadata = {
        "summary": summary,
        "exclusions": exclusions,
        "binary_audit": binary_audit,
        "temperature_converted_n": converted_n,
    }
    return work, metadata
