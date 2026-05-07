import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.tqip_preprocessing import (
    add_one_hot_features,
    apply_ordered_exclusions,
    normalize_temperature_celsius,
    recode_ais_and_derive,
    recode_binary_variables,
)


def test_apply_ordered_exclusions() -> None:
    df = pd.DataFrame(
        {
            "YODISCH": [2019, np.nan, 2020, 2021, 2022],
            "AGEyears": [35, 44, np.nan, 17, 22],
            "riss": [20, 20, 20, 20, 12],
        }
    )
    out, steps = apply_ordered_exclusions(df)
    assert len(out) == 1
    assert out.iloc[0]["AGEyears"] == 35
    assert [step["removed"] for step in steps] == [1, 1, 1, 1]


def test_binary_recoding_logic() -> None:
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
    out, _ = recode_binary_variables(df)
    assert out["ETHNICITY"].tolist() == [1, 0]
    assert out["SUPPLEMENTALOXYGEN"].tolist() == [0, 1]
    assert out["RESPIRATORYASSISTANCE"].tolist() == [1, 0]
    assert out["WITHDRAWALLST"].tolist() == [0, 1]
    assert out["PREHOSPITALCARDIACARREST"].tolist() == [0, 1]
    assert out["HC_RESPIRATORY"].tolist() == [0, 1]


def test_temperature_conversion_logic() -> None:
    df = pd.DataFrame({"TEMPERATURE": [37.0, 98.6, 45.0, 121.0, np.nan]})
    out, converted_n = normalize_temperature_celsius(df)
    assert converted_n == 1
    assert np.isclose(out.loc[1, "TEMPERATURE"], 37.0, atol=0.2)
    assert out.loc[0, "TEMPERATURE"] == 37.0
    assert out.loc[2, "TEMPERATURE"] == 45.0
    assert out.loc[3, "TEMPERATURE"] == 121.0
    assert np.isnan(out.loc[4, "TEMPERATURE"])


def test_ais_unknown_recode_and_derived_columns() -> None:
    df = pd.DataFrame(
        {
            "mxaisbr_HeadNeck": [9, 2, 3],
            "mxaisbr_Face": [0, 1, 9],
            "mxaisbr_Chest": [4, 0, 1],
            "mxaisbr_Abdomen": [2, 3, 0],
            "mxaisbr_Extremities": [0, 9, 5],
        }
    )
    out = recode_ais_and_derive(df)
    assert np.isnan(out.loc[0, "mxaisbr_HeadNeck"])
    assert np.isnan(out.loc[2, "mxaisbr_Face"])
    assert np.isnan(out.loc[1, "mxaisbr_Extremities"])
    assert out["inj_HeadNeck"].tolist() == [0, 1, 1]
    assert out["severe_HeadNeck"].tolist() == [0, 0, 1]
    assert out["inj_Chest"].tolist() == [1, 0, 1]
    assert out["severe_Chest"].tolist() == [1, 0, 0]


def test_verification_level_columns() -> None:
    df = pd.DataFrame(
        {
            "SEX": [1, 2, 3, 1],
            "HMRRHGCTRLSURGTYPE": [1, 2, 9, 1],
            "VERIFICATIONLEVEL": [1, 2, 3, np.nan],
        }
    )
    out = add_one_hot_features(df)
    assert out["L1"].tolist() == [1, 0, 0, 0]
    assert out["L2"].tolist() == [0, 1, 0, 0]
    assert out["L3"].tolist() == [0, 0, 1, 0]
    assert out["L_unkn"].tolist() == [0, 0, 0, 1]
