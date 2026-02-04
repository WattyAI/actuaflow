"""
Input Validation Functions - Version 3 (v1.0)

Complete validation functions with comprehensive error handling,
type hints, and custom exceptions.

Author: Michael Watson
License: MPL-2.0
"""

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import custom exceptions
try:
    from actuaflow.exceptions import (
        InvalidDataTypeError,
        InvalidFamilyError,
        InvalidFormulaError,
        InvalidLinkError,
        InvalidLoadingsError,
        InvalidValueError,
        MissingColumnError,
        MissingValueError,
    )
except ImportError:
    # Fallback - using different names to avoid redefinition
    class InvalidFormulaError(ValueError): pass  # type: ignore
    class InvalidFamilyError(ValueError): pass  # type: ignore
    class InvalidLinkError(ValueError): pass  # type: ignore
    class InvalidLoadingsError(ValueError): pass  # type: ignore
    class MissingColumnError(ValueError): pass  # type: ignore
    class InvalidValueError(ValueError): pass  # type: ignore
    class MissingValueError(ValueError): pass  # type: ignore
    class InvalidDataTypeError(TypeError): pass  # type: ignore


def validate_formula(formula: str) -> bool:
    """
    Validate R-style formula syntax.
    
    Parameters
    ----------
    formula : str
        Formula string (e.g., 'y ~ x1 + x2')
    
    Returns
    -------
    bool
        True if valid
    
    Raises
    ------
    InvalidDataTypeError
        If formula is not a string
    InvalidFormulaError
        If formula is invalid
    
    Examples
    --------
    >>> validate_formula('y ~ x1 + x2')
    True
    >>> validate_formula('y ~ x1 * x2')
    True
    >>> validate_formula('invalid')
    InvalidFormulaError: Formula must contain '~' separator
    """
    if not isinstance(formula, str):
        raise InvalidDataTypeError(
            f"Formula must be string, got {type(formula).__name__}"
        )
    
    if not formula.strip():
        raise InvalidFormulaError(
            formula=formula,
            reason="Formula cannot be empty"
        )
    
    if '~' not in formula:
        raise InvalidFormulaError(
            formula=formula,
            reason="Formula must contain '~' separator"
        )
    
    parts = formula.split('~')
    if len(parts) != 2:
        raise InvalidFormulaError(
            formula=formula,
            reason=f"Formula must have exactly one '~', found {len(parts)-1}"
        )
    
    response, predictors = parts
    
    if not response.strip():
        raise InvalidFormulaError(
            formula=formula,
            reason="Response variable (left side of ~) is empty"
        )
    
    if not predictors.strip():
        raise InvalidFormulaError(
            formula=formula,
            reason="Predictors (right side of ~) are empty"
        )
    
    # Check for invalid characters (basic check)
    invalid_chars = ['@', '#', '$', '%', '^', '&']
    for char in invalid_chars:
        if char in formula:
            raise InvalidFormulaError(
                formula=formula,
                reason=f"Formula contains invalid character '{char}'"
            )
    
    return True


def validate_family_link(family: str, link: str) -> bool:
    """
    Validate family-link combination.
    
    Parameters
    ----------
    family : str
        Distribution family
    link : str
        Link function
    
    Returns
    -------
    bool
        True if valid combination
    
    Raises
    ------
    InvalidDataTypeError
        If family or link is not a string
    InvalidFamilyError
        If family is not recognized
    InvalidLinkError
        If link is not valid for the family
    
    Examples
    --------
    >>> validate_family_link('poisson', 'log')
    True
    >>> validate_family_link('gamma', 'log')
    True
    >>> validate_family_link('poisson', 'inverse')
    InvalidLinkError: Invalid link 'inverse' for family 'poisson'
    """
    if not isinstance(family, str):
        raise InvalidDataTypeError(
            f"family must be string, got {type(family).__name__}"
        )
    
    if not isinstance(link, str):
        raise InvalidDataTypeError(
            f"link must be string, got {type(link).__name__}"
        )
    
    valid_combinations: Dict[str, List[str]] = {
        'poisson': ['log', 'identity'],
        'negative_binomial': ['log', 'identity'],
        'gamma': ['log', 'inverse', 'identity'],
        'inverse_gaussian': ['log', 'inverse', 'inverse_squared', 'identity'],
        'gaussian': ['identity', 'log'],
        'lognormal': ['identity'],
        'tweedie': ['log', 'identity'],
    }
    
    family = family.lower().strip()
    link = link.lower().strip()
    
    if not family:
        raise InvalidFamilyError(
            f"family cannot be empty"
        )
    
    if not link:
        raise InvalidLinkError(
            link="",
            family=family,
            valid_links=valid_combinations.get(family, [])
        )
    
    if family not in valid_combinations:
        raise InvalidFamilyError(
            family=family,
            valid_families=list(valid_combinations.keys())
        )
    
    if link not in valid_combinations[family]:
        raise InvalidLinkError(
            link=link,
            family=family,
            valid_links=valid_combinations[family]
        )
    
    return True


def validate_loadings(loadings: Dict[str, float]) -> bool:
    """
    Validate premium loading factors.
    
    Parameters
    ----------
    loadings : dict
        Loading factors (e.g., {'inflation': 0.03, 'expense_ratio': 0.15})
    
    Returns
    -------
    bool
        True if valid
    
    Raises
    ------
    InvalidDataTypeError
        If loadings is not a dict or values are not numeric
    InvalidLoadingsError
        If loadings are invalid
    
    Examples
    --------
    >>> validate_loadings({'inflation': 0.03, 'expense_ratio': 0.15})
    True
    >>> validate_loadings({'expense_ratio': 1.5})
    InvalidLoadingsError: Loading 'expense_ratio' must be less than 1.0
    """
    if not isinstance(loadings, dict):
        raise InvalidDataTypeError(
            f"loadings must be dict, got {type(loadings).__name__}"
        )
    
    valid_keys = {
        'inflation', 'expense_ratio', 'commission',
        'profit_margin', 'tax_rate', 'profit'
    }
    
    # Keys that must be < 1.0 (ratios)
    ratio_keys = {'expense_ratio', 'commission', 'tax_rate'}
    
    # Keys that should be positive (rates/margins)
    positive_keys = {'inflation', 'profit_margin', 'profit'}
    
    for key, value in loadings.items():
        if key not in valid_keys:
            raise InvalidLoadingsError(
                loading_name=key,
                reason=f"Unknown loading key. Valid keys: {sorted(valid_keys)}"
            )
        
        if not isinstance(value, (int, float)):
            raise InvalidDataTypeError(
                f"Loading '{key}' must be numeric, got {type(value).__name__}"
            )
        
        if value < 0:
            raise InvalidLoadingsError(
                loading_name=key,
                reason=f"cannot be negative, got {value}"
            )
        
        if key in ratio_keys and value >= 1.0:
            raise InvalidLoadingsError(
                loading_name=key,
                reason=f"must be less than 1.0 (100%), got {value}"
            )
        
        if key in positive_keys and value < 0:
            raise InvalidLoadingsError(
                loading_name=key,
                reason=f"Must be non-negative, got {value}"
            )
    
    # Warn if sum of ratios is too high
    ratio_sum = sum(loadings.get(k, 0) for k in ratio_keys)
    if ratio_sum >= 0.9:
        import warnings
        warnings.warn(
            f"Sum of expense_ratio, commission, and tax_rate is {ratio_sum:.1%}, "
            "which leaves very little for claims. This may be intentional but is unusual.",
            UserWarning
        )
    
    return True


def validate_column_exists(
    data: pd.DataFrame,
    column: str,
    data_name: str = "data"
) -> bool:
    """
    Validate that a column exists in a DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to check
    column : str
        Column name
    data_name : str, default='data'
        Name of DataFrame for error messages
    
    Returns
    -------
    bool
        True if column exists
    
    Raises
    ------
    InvalidDataTypeError
        If data is not a DataFrame
    MissingColumnError
        If column not found
    """
    if not isinstance(data, pd.DataFrame):
        raise InvalidDataTypeError(
            f"{data_name} must be DataFrame, got {type(data).__name__}"
        )
    
    if column not in data.columns:
        raise MissingColumnError(
            column=column,
            available_columns=data.columns.tolist()
        )
    
    return True


def validate_positive_values(
    data: pd.DataFrame,
    column: str,
    allow_zero: bool = False
) -> bool:
    """
    Validate that a column contains positive (or non-negative) values.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame
    column : str
        Column name
    allow_zero : bool, default=False
        Whether to allow zero values
    
    Returns
    -------
    bool
        True if valid
    
    Raises
    ------
    MissingColumnError
        If column not found
    InvalidValueError
        If column contains invalid values
    """
    validate_column_exists(data, column)
    
    if allow_zero:
        invalid = data[column] < 0
        condition = "non-negative"
    else:
        invalid = data[column] <= 0
        condition = "positive"
    
    if invalid.any():
        n_invalid = int(invalid.sum())
        min_value = float(data[column].min())
        raise InvalidValueError(
            column=column,
            constraint=f"must contain {condition} values (min={min_value})",
            n_invalid=n_invalid
        )
    
    return True


def validate_no_missing(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    raise_error: bool = True
) -> bool:
    """
    Validate that columns have no missing values.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame
    columns : list of str, optional
        Columns to check (if None, checks all)
    raise_error : bool, default=True
        If True, raises error on missing values. If False, returns False.
    
    Returns
    -------
    bool
        True if no missing values, False if missing and raise_error=False
    
    Raises
    ------
    MissingValueError
        If missing values found and raise_error=True
    """
    if columns is None:
        columns = data.columns.tolist()
    
    missing_info: Dict[str, int] = {}
    for col in columns:
        validate_column_exists(data, col)
        n_missing = int(data[col].isnull().sum())
        if n_missing > 0:
            missing_info[col] = n_missing
    
    if missing_info:
        msg = (
            f"Found {sum(missing_info.values())} missing values across "
            f"{len(missing_info)} columns: {missing_info}"
        )
        if raise_error:
            raise MissingValueError(
                columns=list(missing_info.keys()),
                n_missing=missing_info
            )
        else:
            import warnings
            warnings.warn(msg, UserWarning)
            return False
    
    return True


def validate_numeric_column(
    data: pd.DataFrame,
    column: str
) -> bool:
    """
    Validate that a column is numeric.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame
    column : str
        Column name
    
    Returns
    -------
    bool
        True if numeric
    
    Raises
    ------
    MissingColumnError
        If column not found
    InvalidValueError
        If column is not numeric
    """
    validate_column_exists(data, column)
    
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise InvalidValueError(
            column=column,
            constraint=f"must be numeric, got dtype={data[column].dtype}"
        )
    
    return True


def validate_categorical_column(
    data: pd.DataFrame,
    column: str,
    max_categories: Optional[int] = None
) -> bool:
    """
    Validate that a column is categorical with reasonable cardinality.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame
    column : str
        Column name
    max_categories : int, optional
        Maximum allowed unique values
    
    Returns
    -------
    bool
        True if valid
    
    Raises
    ------
    MissingColumnError
        If column not found
    InvalidValueError
        If too many categories
    """
    validate_column_exists(data, column)
    
    n_unique = data[column].nunique()
    
    if max_categories is not None and n_unique > max_categories:
        raise InvalidValueError(
            column=column,
            constraint=f"Has {n_unique} unique values, exceeds maximum of {max_categories}. "
                      "Consider binning or using a different variable.",
            n_invalid=n_unique - max_categories
        )
    
    # Warn if very high cardinality for categorical
    if n_unique > 50:
        import warnings
        warnings.warn(
            f"Column '{column}' has {n_unique} unique values. "
            "High cardinality may cause fitting issues.",
            UserWarning
        )
    
    return True