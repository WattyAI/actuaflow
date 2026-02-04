"""
Exposure Rating Tools

Comprehensive rating functions for computing rates and applying class plans.

Features:
- Rate per exposure computation with loadings
- Class plan creation from factor relativities
- Rating table generation for production systems
- Relativity application and aggregation

Author: Michael Watson
License: MPL-2.0
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rate_per_exposure(
    pure_premium: Union[pd.Series, np.ndarray, float],
    exposure: Union[pd.Series, np.ndarray, float],
    loadings: Optional[Dict[str, float]] = None
) -> Union[pd.Series, np.ndarray, float]:
    """
    Compute rate per unit exposure.
    
    Rate = (Pure Premium / Exposure) × Loading Factor
    
    Parameters
    ----------
    pure_premium : array-like or float
        Pure premium (total expected loss)
    exposure : array-like or float
        Exposure (e.g., policy years, sales, payroll)
    loadings : dict, optional
        Loading factors (if None, uses pure premium rate)
    
    Returns
    -------
    rate : array-like or float
        Rate per unit exposure
    
    Examples
    --------
    >>> rate = compute_rate_per_exposure(
    ...     pure_premium=1000,
    ...     exposure=10,
    ...     loadings={'profit': 0.05, 'expenses': 0.15}
    ... )
    """
    # Base rate
    rate = pure_premium / np.maximum(exposure, 1e-10)
    
    # Apply loadings if provided
    if loadings:
        loading_factor = 1.0
        for key, value in loadings.items():
            if 'ratio' in key or 'commission' in key or 'tax' in key:
                # These are divisors
                loading_factor /= max(1 - value, 0.01)
            elif 'margin' in key or 'inflation' in key or 'profit' in key:
                # These are multipliers
                loading_factor *= (1 + value)
        
        rate = rate * loading_factor
    
    return rate


def create_class_plan(
    data: pd.DataFrame,
    rating_factors: List[str],
    base_rate: float,
    relativities: Dict[str, Dict[str, float]],
    exposure_col: str = 'exposure',
    min_rate: Optional[float] = None,
    max_rate: Optional[float] = None
) -> pd.DataFrame:
    """
    Create a class plan rate table by applying factor relativities to base rate.
    
    Class Plan Formula:
    Rate = Base Rate × Factor_1_Relativity × Factor_2_Relativity × ...
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with rating factor values
    rating_factors : list of str
        Rating factor column names
    base_rate : float
        Base rate per unit exposure
    relativities : dict
        Nested dict of {factor: {level: relativity}}
    exposure_col : str
        Exposure column name
    min_rate : float, optional
        Minimum allowed rate
    max_rate : float, optional
        Maximum allowed rate
    
    Returns
    -------
    pd.DataFrame
        Data with added 'rate' and 'premium' columns
    
    Examples
    --------
    >>> relativities = {
    ...     'age_group': {'18-25': 1.5, '26-35': 1.0, '36+': 0.8},
    ...     'vehicle_type': {'sedan': 1.0, 'suv': 1.2, 'sports': 1.8}
    ... }
    >>> rates = create_class_plan(
    ...     data=policies,
    ...     rating_factors=['age_group', 'vehicle_type'],
    ...     base_rate=100.0,
    ...     relativities=relativities
    ... )
    """
    result = data.copy()
    result['rate'] = base_rate
    
    # Validate inputs
    try:
        from actuaflow.exceptions import MissingColumnError as MissingColumnError
    except ImportError:
        class MissingColumnError(ValueError): pass  # type: ignore
    
    for factor in rating_factors:
        if factor not in result.columns:
            raise MissingColumnError(
                column=factor,
                available_columns=result.columns.tolist()
            )
    
    # Apply each factor's relativities
    for factor in rating_factors:
        
        if factor not in relativities:
            continue
        
        # Map relativities
        factor_rel = relativities[factor]
        result['rate'] *= result[factor].map(factor_rel).fillna(1.0)
    
    # Apply min/max caps
    if min_rate is not None:
        result['rate'] = result['rate'].clip(lower=min_rate)
    if max_rate is not None:
        result['rate'] = result['rate'].clip(upper=max_rate)
    
    # Compute premium
    if exposure_col in result.columns:
        result['premium'] = result['rate'] * result[exposure_col]
    
    return result


def apply_relativities(
    base_value: float,
    factor_relativities: Dict[str, float]
) -> float:
    """
    Apply multiplicative relativities to a base value.
    
    Result = Base × Rel_1 × Rel_2 × ...
    
    Parameters
    ----------
    base_value : float
        Base value (rate, premium, etc.)
    factor_relativities : dict
        {factor_name: relativity_value}
    
    Returns
    -------
    float
        Adjusted value
    
    Examples
    --------
    >>> rate = apply_relativities(
    ...     base_value=100.0,
    ...     factor_relativities={'age': 1.2, 'territory': 0.9}
    ... )
    >>> # Result: 100 × 1.2 × 0.9 = 108.0
    """
    result = base_value
    for relativity in factor_relativities.values():
        result *= relativity
    return result


def create_rating_table(
    factor_combinations: pd.DataFrame,
    base_rate: float,
    relativities: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Create a complete rating table for all factor combinations.
    
    Useful for creating lookup tables for production rating systems.
    
    Parameters
    ----------
    factor_combinations : pd.DataFrame
        All combinations of rating factors
    base_rate : float
        Base rate
    relativities : dict
        Nested relativities dict
    
    Returns
    -------
    pd.DataFrame
        Rating table with rate for each combination
    
    Examples
    --------
    >>> import itertools
    >>> ages = ['18-25', '26-35', '36+']
    >>> vehicles = ['sedan', 'suv', 'sports']
    >>> combos = pd.DataFrame(
    ...     list(itertools.product(ages, vehicles)),
    ...     columns=['age_group', 'vehicle_type']
    ... )
    >>> rating_table = create_rating_table(combos, 100.0, relativities)
    """
    result = factor_combinations.copy()
    result['base_rate'] = base_rate
    result['combined_relativity'] = 1.0
    
    # Apply each factor
    for factor, levels in relativities.items():
        if factor not in result.columns:
            continue
        
        result[f'{factor}_rel'] = result[factor].map(levels).fillna(1.0)
        result['combined_relativity'] *= result[f'{factor}_rel']
    
    result['final_rate'] = result['base_rate'] * result['combined_relativity']
    
    return result


def compute_credibility_weighted_rate(
    manual_rate: float,
    experience_rate: float,
    credibility: float
) -> float:
    """
    Compute credibility-weighted rate.
    
    Blends manual (a priori) rate with experience (a posteriori) rate.
    
    Credibility Formula:
    Rate = Z × Experience_Rate + (1 - Z) × Manual_Rate
    
    where Z is the credibility factor (0 to 1).
    
    Parameters
    ----------
    manual_rate : float
        Manual (class plan) rate
    experience_rate : float
        Experience-based rate
    credibility : float
        Credibility factor (0 = full manual, 1 = full experience)
    
    Returns
    -------
    float
        Credibility-weighted rate
    
    Examples
    --------
    >>> rate = compute_credibility_weighted_rate(
    ...     manual_rate=100.0,
    ...     experience_rate=120.0,
    ...     credibility=0.3
    ... )
    >>> # Result: 0.3 × 120 + 0.7 × 100 = 106.0
    """
    if not 0 <= credibility <= 1:
        raise ValueError("Credibility must be between 0 and 1")
    
    return credibility * experience_rate + (1 - credibility) * manual_rate


def compute_experience_mod(
    actual_losses: float,
    expected_losses: float,
    credibility: Optional[float] = None,
    cap: Optional[float] = None
) -> float:
    """
    Compute experience modification factor.
    
    Experience Mod = [Z × (Actual / Expected) + (1 - Z) × 1.0]
    
    Parameters
    ----------
    actual_losses : float
        Actual incurred losses
    expected_losses : float
        Expected losses (from class rate)
    credibility : float, optional
        Credibility factor. If None, uses full credibility (Z=1)
    cap : float, optional
        Maximum allowed modification (e.g., 2.0 for 200% cap)
    
    Returns
    -------
    float
        Experience modification factor
    
    Examples
    --------
    >>> exp_mod = compute_experience_mod(
    ...     actual_losses=120000,
    ...     expected_losses=100000,
    ...     credibility=0.5,
    ...     cap=2.0
    ... )
    >>> # Result: 0.5 × (120/100) + 0.5 × 1.0 = 1.1
    """
    if credibility is None:
        credibility = 1.0
    
    if expected_losses <= 0:
        return 1.0
    
    loss_ratio = actual_losses / expected_losses
    mod = credibility * loss_ratio + (1 - credibility) * 1.0
    
    if cap is not None:
        mod = min(mod, cap)
    
    return mod