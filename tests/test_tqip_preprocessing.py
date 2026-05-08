import numpy as np
import pandas as pd
import pytest

from src.cleaning_utils import (
    AIS_COLUMNS,
    OHE_RENAME_MAP,
    add_verification_level_columns,
    apply_cohort_exclusions,
    harmonize_binary_variables,
    normalize_temperature_to_celsius,
    one_hot_encode_features,
    process_ais_features,
    validate_required_columns,
)


def test_validate_required_columns_raises_for_missing_columns() -> None:
    df = pd.DataFrame({"YODISCH": [2020]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df)


def test_apply_cohort_exclusions_removes_invalid_rows_in_order() -> None:
    df = pd.DataFrame(
        {
            "YODISCH": [2019, np.nan, 2020, 2021, 2022],
            "AGEyears": [35, 44, np.nan, 17, 22],
            "riss": [20, 20, 20, 20, 12],
        }
    )
    out, steps = apply_cohort_exclusions(df)
    assert len(out) == 1
    assert out.iloc[0]["YODISCH"] == 2019
    assert out.iloc[0]["AGEyears"] == 35
    assert [step["removed"] for step in steps] == [1, 1, 1, 1]


def test_harmonize_binary_variables_harmonizes_values() -> None:
    df = pd.DataFrame(
        {
            "ETHNICITY": [1, 2],
            "SUPPLEMENTALOXYGEN": [1, 2],
            "RESPIRATORYASSISTANCE": [2, 1],
            "WITHDRAWALLST": [2, 1],
            "PREHOSPITALCARDIACARREST": [2, 1],
            "HC_RESPIRATORY": [0, 1],
        }
    )
    out, _ = harmonize_binary_variables(df)
    assert out["ETHNICITY"].tolist() == [1, 0]
    assert out["SUPPLEMENTALOXYGEN"].tolist() == [0, 1]
    assert out["RESPIRATORYASSISTANCE"].tolist() == [1, 0]
    assert out["WITHDRAWALLST"].tolist() == [0, 1]
    assert out["PREHOSPITALCARDIACARREST"].tolist() == [0, 1]
    assert out["HC_RESPIRATORY"].tolist() == [0, 1]


def test_normalize_temperature_to_celsius_converts_fahrenheit() -> None:
    df = pd.DataFrame({"TEMPERATURE": [37.0, 98.6, 45.0, 121.0, np.nan]})
    out, converted_n = normalize_temperature_to_celsius(df)
    assert converted_n == 1
    assert np.isclose(out.loc[1, "TEMPERATURE"], 37.0, atol=0.2)
    assert out.loc[0, "TEMPERATURE"] == 37.0
    assert out.loc[2, "TEMPERATURE"] == 45.0
    assert out.loc[3, "TEMPERATURE"] == 121.0
    assert np.isnan(out.loc[4, "TEMPERATURE"])


def test_process_ais_features_handles_unknown_and_creates_indicators() -> None:
    df = pd.DataFrame(
        {
            "mxaisbr_HeadNeck": [9, 2, 3],
            "mxaisbr_Face": [0, 1, 9],
            "mxaisbr_Chest": [4, 0, 1],
            "mxaisbr_Abdomen": [2, 3, 0],
            "mxaisbr_Extremities": [0, 9, 5],
        }
    )
    out = process_ais_features(df, ais_cols=AIS_COLUMNS, ais_thresh=3)
    first_row = out.iloc[0]
    second_row = out.iloc[1]
    third_row = out.iloc[2]
    assert np.isnan(first_row["mxaisbr_HeadNeck"])
    assert np.isnan(third_row["mxaisbr_Face"])
    assert np.isnan(second_row["mxaisbr_Extremities"])
    assert out["inj_HeadNeck"].tolist() == [0, 1, 1]
    assert out["severe_HeadNeck"].tolist() == [0, 0, 1]
    assert out["inj_Chest"].tolist() == [1, 0, 1]
    assert out["severe_Chest"].tolist() == [1, 0, 0]


def test_verification_level_columns_are_created_from_source_column() -> None:
    df = pd.DataFrame(
        {
            "SEX": [1, 2, 3, 1],
            "HMRRHGCTRLSURGTYPE": [1, 2, 9, 1],
            "VERIFICATIONLEVEL": [1, 2, 3, np.nan],
        }
    )
    out = one_hot_encode_features(df, columns_to_encode=["SEX", "HMRRHGCTRLSURGTYPE"], rename_map=OHE_RENAME_MAP)
    out = add_verification_level_columns(out)
    assert out["L1"].tolist() == [1, 0, 0, 0]
    assert out["L2"].tolist() == [0, 1, 0, 0]
    assert out["L3"].tolist() == [0, 0, 1, 0]
    assert out["L_unkn"].tolist() == [0, 0, 0, 1]
