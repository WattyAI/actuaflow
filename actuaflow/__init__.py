"""
ActuaFlow: Modern Actuarial Pricing Library

A Python library for non-life insurance pricing using GLM-based frequency-severity modeling.

Purpose: Research and commercial use by companies in the insurance and actuarial industries.

Author: Michael Watson (michael@watsondataandrisksolutions.com)
License: MPL-2.0 (Mozilla Public License v2.0)

Key Features:
    - GLM wrappers for Poisson, Negative Binomial, Gamma, Tweedie distributions
    - Frequency-severity modeling workflow
    - Comprehensive model diagnostics (AIC, BIC, VIF, lift curves, residuals)
    - Exposure rating and trending tools
    - Portfolio impact analysis and elasticity curves
    - Actuarial-friendly summary tables

Example:
    >>> import actuaflow as af
    >>> from actuaflow.glm import FrequencyGLM
    >>> 
    >>> # Fit frequency model
    >>> freq_model = FrequencyGLM(family='poisson', link='log')
    >>> freq_model.fit(data, formula='claim_count ~ age_group + vehicle_type', 
    ...                offset='exposure')
    >>> 
    >>> # Get diagnostics
    >>> diag = freq_model.diagnostics()
    >>> print(f"AIC: {diag['aic']:.2f}, Dispersion: {diag['dispersion']:.3f}")
"""

__version__ = "0.1.0"
__author__ = "Michael Watson"
__email__ = "michael@watsondataandrisksolutions.com"
__license__ = "MPL-2.0"
__copyright__ = "Copyright (c) 2026–present Michael Watson"


# Exposure rating
from actuaflow.exposure.rating import (
    apply_relativities,
    compute_rate_per_exposure,
    create_class_plan,
)
from actuaflow.exposure.trending import (
    apply_inflation,
    apply_trend_factor,
    project_exposures,
)
from actuaflow.freqsev.aggregate import AggregateModel, combine_models

# Frequency-Severity workflow
from actuaflow.freqsev.frequency import FrequencyModel
from actuaflow.freqsev.severity import SeverityModel

# Diagnostics
from actuaflow.glm.diagnostics import (
    compute_diagnostics,
    compute_gini_index,
    compute_lift_curve,
    compute_vif,
)

# Core GLM models
from actuaflow.glm.models import BaseGLM, FrequencyGLM, SeverityGLM, TweedieGLM
from actuaflow.portfolio.elasticity import (
    compute_elasticity_curve,
    estimate_demand_elasticity,
)

# Portfolio analysis
from actuaflow.portfolio.impact import (
    compute_premium_impact,
    factor_sensitivity,
    mix_shift_analysis,
)

# Utilities
from actuaflow.utils.data import (
    load_data,
    validate_data,
    calculate_exposure,
)
from actuaflow.utils.validation import validate_family_link, validate_formula

__all__ = [
    # Version
    "__version__",
    "__author__",
    # GLM Models
    "FrequencyGLM",
    "SeverityGLM",
    "TweedieGLM",
    "BaseGLM",
    # Diagnostics
    "compute_diagnostics",
    "compute_vif",
    "compute_lift_curve",
    "compute_gini_index",
    # Frequency-Severity
    "FrequencyModel",
    "SeverityModel",
    "AggregateModel",
    "combine_models",
    # Exposure Rating
    "compute_rate_per_exposure",
    "create_class_plan",
    "apply_relativities",
    "apply_trend_factor",
    "apply_inflation",
    "project_exposures",
    # Portfolio Analysis
    "compute_premium_impact",
    "factor_sensitivity",
    "mix_shift_analysis",
    "compute_elasticity_curve",
    "estimate_demand_elasticity",
    # Utilities
    "load_data",
    "validate_data",
    "calculate_exposure",
    "validate_formula",
    "validate_family_link",
]