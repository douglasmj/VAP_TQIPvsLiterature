"""Compatibility shim — imports re-exported from vap_tqip.cleaning and vap_tqip.constants."""

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
from vap_tqip.constants import (
    AIS_COLUMNS,
    OHE_RENAME_MAP,
    RAW_ENCODED_COLUMNS_TO_DROP,
    REQUIRED_COLUMNS,
)
