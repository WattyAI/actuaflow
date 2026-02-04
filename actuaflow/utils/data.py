"""
Data Loading and Preparation Utilities

Provides functions for loading data from various formats, validation,
and preparing data for modeling workflows.

Author: Michael Watson
License: MPL-2.0
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def load_data(
    file_path: str,
    use_polars: bool = True,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Uses Polars for fast loading of large CSV/Parquet files, then converts to pandas.
    
    Parameters
    ----------
    file_path : str
        Path to data file
    use_polars : bool, default=True
        Use Polars for CSV/Parquet (faster for large files)
    **kwargs
        Additional arguments passed to reader
    
    Returns
    -------
    pd.DataFrame
        Loaded data
    
    Examples
    --------
    >>> data = load_data('policies.csv')
    >>> data = load_data('claims.parquet', use_polars=True)
    >>> data = load_data('losses.xlsx', use_polars=False)
    """
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path_obj.suffix.lower()
    
    if suffix == '.csv' and use_polars:
        df_pl = pl.read_csv(str(file_path_obj), **kwargs)
        return df_pl.to_pandas()
    
    elif suffix == '.csv':
        return pd.read_csv(file_path_obj, **kwargs)
    
    elif suffix == '.parquet' and use_polars:
        df_pl = pl.read_parquet(str(file_path_obj), **kwargs)
        return df_pl.to_pandas()
    
    elif suffix == '.parquet':
        return pd.read_parquet(file_path_obj, **kwargs)
    
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path_obj, **kwargs)
    
    elif suffix == '.feather':
        return pd.read_feather(file_path_obj, **kwargs)
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def validate_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    positive_columns: Optional[List[str]] = None,
    check_missing: bool = True
) -> Dict[str, Any]:
    """
    Validate data quality and structure.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
    required_columns : list of str, optional
        Columns that must be present
    numeric_columns : list of str, optional
        Columns that must be numeric
    positive_columns : list of str, optional
        Columns that must have positive values
    check_missing : bool
        Check for missing values
    
    Returns
    -------
    dict
        Validation results with:
        - is_valid: bool
        - errors: list of error messages
        - warnings: list of warnings
        - summary: data summary
    
    Examples
    --------
    >>> validation = validate_data(
    ...     data,
    ...     required_columns=['policy_id', 'exposure'],
    ...     numeric_columns=['exposure', 'claim_count'],
    ...     positive_columns=['exposure']
    ... )
    >>> if not validation['is_valid']:
    ...     print(validation['errors'])
    """
    errors = []
    warnings = []
    
    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(data.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")
    
    # Check numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(f"Column '{col}' is not numeric")
    
    # Check positive values
    if positive_columns:
        for col in positive_columns:
            if col in data.columns:
                if (data[col] <= 0).any():
                    n_invalid = (data[col] <= 0).sum()
                    warnings.append(
                        f"Column '{col}' has {n_invalid} non-positive values"
                    )
    
    # Check missing values
    if check_missing:
        missing_counts = data.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            warnings.append(f"Missing values found in: {missing_cols.to_dict()}")
    
    # Summary
    summary = {
        'n_rows': len(data),
        'n_columns': len(data.columns),
        'memory_mb': data.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': data.dtypes.value_counts().to_dict()
    }
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'summary': summary
    }


def prepare_frequency_data(
    policy_data: pd.DataFrame,
    claims_data: pd.DataFrame,
    policy_id: str = 'policy_id',
    policy_id_policy: Optional[str] = None,
    policy_id_claims: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare frequency modeling data.
    
    Counts claims per policy and merges onto policy data.
    
    Parameters
    ----------
    policy_data : pd.DataFrame
        Policy-level data
    claims_data : pd.DataFrame
        Claim-level data
    policy_id : str, default='policy_id'
        Policy ID column (if same in both datasets)
    policy_id_policy : str, optional
        Policy ID column name in `policy_data` (overrides `policy_id` if provided)
    policy_id_claims : str, optional
        Policy ID column name in `claims_data` (overrides `policy_id` if provided)
    
    Returns
    -------
    pd.DataFrame
        Frequency modeling dataset
    """
    # Determine actual column names
    pol_id_policy = policy_id_policy if policy_id_policy is not None else policy_id
    pol_id_claims = policy_id_claims if policy_id_claims is not None else policy_id

    # Count claims using claims dataset column name
    claims_per_policy = claims_data.groupby(pol_id_claims).size().reset_index(
        name='claim_count'
    )

    # Rename if column names differ
    if pol_id_claims != pol_id_policy:
        claims_per_policy = claims_per_policy.rename(
            columns={pol_id_claims: pol_id_policy}
        )

    # Merge onto policies
    freq_data = policy_data.merge(
        claims_per_policy,
        on=pol_id_policy,
        how='left'
    )

    # Fill zeros
    freq_data['claim_count'] = freq_data['claim_count'].fillna(0).astype(int)

    return freq_data


def prepare_severity_data(
    claims_data: pd.DataFrame,
    policy_data: pd.DataFrame,
    policy_id: str = 'policy_id',
    filter_zeros: bool = True,
    amount_col: str = 'amount'
) -> pd.DataFrame:
    """
    Prepare severity modeling data.
    
    Merges policy factors onto claims.
    
    Parameters
    ----------
    claims_data : pd.DataFrame
        Claim-level data
    policy_data : pd.DataFrame
        Policy-level data with rating factors
    policy_id : str
        Policy ID column
    filter_zeros : bool
        Remove zero/negative claims
    amount_col : str
        Claim amount column
    
    Returns
    -------
    pd.DataFrame
        Severity modeling dataset
    """
    # Get rating factors
    rating_factors = [
        col for col in policy_data.columns
        if col not in ['claim_count', amount_col] and col != policy_id
    ]
    
    # Merge
    sev_data = claims_data.merge(
        policy_data[[policy_id] + rating_factors],
        on=policy_id,
        how='left'
    )
    
    # Filter
    if filter_zeros and amount_col in sev_data.columns:
        sev_data = sev_data[sev_data[amount_col] > 0]
    
    return sev_data


def split_train_test(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to split
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
    stratify_col : str, optional
        Column for stratified sampling
    
    Returns
    -------
    tuple
        (train_data, test_data)
    """
    from sklearn.model_selection import train_test_split
    
    if stratify_col:
        stratify = data[stratify_col]
    else:
        stratify = None
    
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    return train.reset_index(drop=True), test.reset_index(drop=True)


def calculate_exposure(
    policy_data: pd.DataFrame,
    start_date_col: str = 'policy_start_date',
    end_date_col: str = 'policy_end_date',
    method: str = 'years',
    exposure_col: str = 'exposure'
) -> pd.DataFrame:
    """
    Calculate policy exposure from start and end dates.
    
    Adds an exposure column to the policy data representing the time period
    covered by each policy.
    
    Parameters
    ----------
    policy_data : pd.DataFrame
        Policy data with start and end dates
    start_date_col : str, default='policy_start_date'
        Column name for policy start date
    end_date_col : str, default='policy_end_date'
        Column name for policy end date
    method : str, default='years'
        Method for calculating exposure:
        - 'years': Exposure in years (days / 365.25)
        - 'days': Exposure in days
        - 'months': Exposure in months (days / 30.44)
    exposure_col : str, default='exposure'
        Name for the output exposure column
    
    Returns
    -------
    pd.DataFrame
        Policy data with exposure column added
    
    Raises
    ------
    ValueError
        If start_date_col or end_date_col not found in data
        If method is not 'years', 'days', or 'months'
        If end date is before start date for any policy
    """
    # Validate inputs
    if start_date_col not in policy_data.columns:
        raise ValueError(f"Column '{start_date_col}' not found in policy_data")
    
    if end_date_col not in policy_data.columns:
        raise ValueError(f"Column '{end_date_col}' not found in policy_data")
    
    if method not in ['years', 'days', 'months']:
        raise ValueError(
            f"method must be 'years', 'days', or 'months', got '{method}'"
        )
    
    # Create copy to avoid modifying original
    result = policy_data.copy()
    
    # Convert to datetime
    result[start_date_col] = pd.to_datetime(result[start_date_col])
    result[end_date_col] = pd.to_datetime(result[end_date_col])
    
    # Check for invalid dates
    invalid_dates = result[end_date_col] < result[start_date_col]
    if invalid_dates.any():
        n_invalid = int(invalid_dates.sum())
        raise ValueError(
            f"Found {n_invalid} policies where end_date is before start_date"
        )
    
    # Calculate days difference
    days_diff = (result[end_date_col] - result[start_date_col]).dt.days
    
    # Convert to requested method
    if method == 'years':
        result[exposure_col] = days_diff / 365.25
    elif method == 'months':
        result[exposure_col] = days_diff / 30.44
    else:  # days
        result[exposure_col] = days_diff
    
    logger.info(
        f"Calculated exposure for {len(result)} policies. "
        f"Mean exposure: {result[exposure_col].mean():.2f} {method}, "
        f"Total exposure: {result[exposure_col].sum():.2f} {method}"
    )
    
    return result