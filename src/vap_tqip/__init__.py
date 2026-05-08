"""vap_tqip — TQIP VAP preprocessing and plotting utilities.

Install locally with:
    pip install -e .

Example usage::

    from vap_tqip.cleaning import load_tqip_data, apply_cohort_exclusions
    from vap_tqip.plotting import plot_patient_count_by_year

Or import frequently-used names directly from the top-level package::

    from vap_tqip import load_tqip_data, apply_cohort_exclusions
"""

from vap_tqip.constants import (
    AIS_COLUMNS,
    CAT_FEATURES,
    NUM_FEATURES,
    OHE_RENAME_MAP,
    RAW_ENCODED_COLUMNS_TO_DROP,
    REQUIRED_COLUMNS,
)
from vap_tqip.cleaning import (
    add_any_injury_indicators,
    add_severe_injury_indicators,
    add_verification_level_columns,
    apply_cohort_exclusions,
    build_vap_ohe_dataset,
    drop_original_encoded_columns,
    exclude_low_riss,
    exclude_missing_age,
    exclude_missing_discharge_year,
    exclude_underage_patients,
    harmonize_binary_variables,
    load_tqip_data,
    normalize_temperature_to_celsius,
    one_hot_encode_features,
    print_value_counts,
    process_ais_features,
    recode_ais_unknowns,
    summarize_raw_cohort,
    validate_required_columns,
)
from vap_tqip.plotting import (
    plot_categorical_feature_distributions,
    plot_numeric_feature_distributions,
    plot_patient_count_by_year,
    plot_temperature_distribution,
    plot_withdrawal_lst_by_year,
)

__all__ = [
    # constants
    "AIS_COLUMNS",
    "CAT_FEATURES",
    "NUM_FEATURES",
    "OHE_RENAME_MAP",
    "RAW_ENCODED_COLUMNS_TO_DROP",
    "REQUIRED_COLUMNS",
    # cleaning
    "add_any_injury_indicators",
    "add_severe_injury_indicators",
    "add_verification_level_columns",
    "apply_cohort_exclusions",
    "build_vap_ohe_dataset",
    "drop_original_encoded_columns",
    "exclude_low_riss",
    "exclude_missing_age",
    "exclude_missing_discharge_year",
    "exclude_underage_patients",
    "harmonize_binary_variables",
    "load_tqip_data",
    "normalize_temperature_to_celsius",
    "one_hot_encode_features",
    "print_value_counts",
    "process_ais_features",
    "recode_ais_unknowns",
    "summarize_raw_cohort",
    "validate_required_columns",
    # plotting
    "plot_categorical_feature_distributions",
    "plot_numeric_feature_distributions",
    "plot_patient_count_by_year",
    "plot_temperature_distribution",
    "plot_withdrawal_lst_by_year",
]
