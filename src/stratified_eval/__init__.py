"""
Stratified Evaluation Package for German GEC Models.

This package provides a comprehensive evaluation framework for profiling
Grammar Error Correction models across different difficulty tiers.
"""

from .evaluation_data import (
    TIER1_EXAMPLES,
    TIER2_EXAMPLES,
    TIER3_EXAMPLES,
    FLAWLESS_EXAMPLES,
    STRESS_TEST_LENGTH,
    STRESS_TEST_HIGH_DENSITY,
    STRESS_TEST_DIALECT,
    get_all_tier_data,
    get_flawless_data,
    get_stress_test_data,
)

from .error_classifier import (
    ErrorType,
    ErrorClassification,
    classify_error_type,
    compute_error_matrix,
)

from .capabilities_matrix import (
    ModelCapabilities,
    generate_capabilities_matrix,
    save_capabilities_report,
    generate_comparison_table,
)

__all__ = [
    "TIER1_EXAMPLES",
    "TIER2_EXAMPLES",
    "TIER3_EXAMPLES",
    "FLAWLESS_EXAMPLES",
    "STRESS_TEST_LENGTH",
    "STRESS_TEST_HIGH_DENSITY",
    "STRESS_TEST_DIALECT",
    "get_all_tier_data",
    "get_flawless_data",
    "get_stress_test_data",
    "ErrorType",
    "ErrorClassification",
    "classify_error_type",
    "compute_error_matrix",
    "ModelCapabilities",
    "generate_capabilities_matrix",
    "save_capabilities_report",
    "generate_comparison_table",
]
