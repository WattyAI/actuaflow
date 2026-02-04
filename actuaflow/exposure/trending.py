"""
Trending and Inflation Adjustment Tools

Functions for adjusting historical losses to current cost levels.

Features:
- Historical loss trending to current levels
- Inflation adjustment calculations
- Trend factor computation between dates
- Exposure and premium projection
- Loss development to ultimate
- On-level premium adjustment

Author: Michael Watson
License: MPL-2.0
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_trend_factor(
    historical_value: Union[float, pd.Series, np.ndarray],
    trend_rate: float,
    years: float
) -> Union[float, pd.Series, np.ndarray]:
    """
    Apply trend factor to adjust historical values to current levels.
    
    Trend Formula:
    Current Value = Historical Value × (1 + trend_rate) ^ years
    
    Parameters
    ----------
    historical_value : float or array-like
        Historical loss amounts or rates
    trend_rate : float
        Annual trend rate (e.g., 0.03 for 3%)
    years : float
        Number of years to trend (can be fractional)
    
    Returns
    -------
    current_value : float or array-like
        Trended values at current cost level
    
    Examples
    --------
    >>> # Trend 2020 losses to 2024 with 3% annual trend
    >>> current_losses = apply_trend_factor(100000, 0.03, 4)
    >>> # Result: 100000 × 1.03^4 = 112,551
    """
    trend_factor = (1 + trend_rate) ** years
    return historical_value * trend_factor


def apply_inflation(
    base_amount: Union[float, pd.Series, np.ndarray],
    inflation_rate: float
) -> Union[float, pd.Series, np.ndarray]:
    """
    Apply one-year inflation adjustment.
    
    Inflated Amount = Base Amount × (1 + inflation_rate)
    
    Parameters
    ----------
    base_amount : float or array-like
        Base amount at current price level
    inflation_rate : float
        Expected inflation rate
    
    Returns
    -------
    inflated_amount : float or array-like
        Amount adjusted for inflation
    
    Examples
    --------
    >>> next_year_losses = apply_inflation(100000, 0.025)
    >>> # Result: 102,500
    """
    return base_amount * (1 + inflation_rate)


def compute_trend_factor(
    from_date: Union[str, datetime],
    to_date: Union[str, datetime],
    annual_trend_rate: float
) -> float:
    """
    Compute trend factor between two dates.
    
    Parameters
    ----------
    from_date : str or datetime
        Start date (historical)
    to_date : str or datetime
        End date (current/future)
    annual_trend_rate : float
        Annual trend rate
    
    Returns
    -------
    float
        Compound trend factor
    
    Examples
    --------
    >>> factor = compute_trend_factor('2020-01-01', '2024-06-01', 0.03)
    >>> # Trends from Jan 2020 to Jun 2024 (4.5 years) at 3%
    """
    from_dt = pd.to_datetime(from_date) if isinstance(from_date, str) else from_date
    to_dt = pd.to_datetime(to_date) if isinstance(to_date, str) else to_date
    
    time_diff = to_dt - from_dt
    years = float(time_diff.days) / 365.25
    return float((1 + annual_trend_rate) ** years)


def project_exposures(
    current_exposures: Union[float, pd.Series],
    growth_rate: float,
    years: int = 1
) -> Union[float, pd.Series]:
    """
    Project future exposures based on growth rate.
    
    Future Exposures = Current Exposures × (1 + growth_rate) ^ years
    
    Parameters
    ----------
    current_exposures : float or pd.Series
        Current exposure units
    growth_rate : float
        Annual growth rate
    years : int
        Number of years to project
    
    Returns
    -------
    future_exposures : float or pd.Series
        Projected exposures
    
    Examples
    --------
    >>> future_exp = project_exposures(10000, 0.05, 3)
    >>> # Project 10,000 units forward 3 years at 5% growth
    >>> # Result: 11,576
    """
    return current_exposures * (1 + growth_rate) ** years


def development_to_ultimate(
    reported_losses: Union[float, pd.Series],
    development_factor: float
) -> Union[float, pd.Series]:
    """
    Develop losses to ultimate using loss development factor.
    
    Ultimate Losses = Reported Losses × Development Factor
    
    Parameters
    ----------
    reported_losses : float or array-like
        Losses reported to date
    development_factor : float
        Age-to-ultimate development factor (e.g., 1.15 for 15% development)
    
    Returns
    -------
    ultimate_losses : float or array-like
        Projected ultimate losses
    
    Examples
    --------
    >>> ultimate = development_to_ultimate(100000, 1.15)
    >>> # Result: 115,000
    """
    return reported_losses * development_factor


def onlevel_adjustment(
    historical_premium: Union[float, pd.Series],
    rate_changes: pd.DataFrame
) -> Union[float, pd.Series]:
    """
    Adjust historical premium to current rate level (on-leveling).
    
    Used to adjust earned premium for rate changes that occurred mid-period.
    
    Parameters
    ----------
    historical_premium : float or pd.Series
        Earned premium at historical rates
    rate_changes : pd.DataFrame
        Rate changes with columns:
        - effective_date: Date of rate change
        - rate_change: Rate change factor (e.g., 1.05 for 5% increase)
        - fraction: Fraction of exposure period after change
    
    Returns
    -------
    onlevel_premium : float or pd.Series
        Premium adjusted to current rate level
    
    Examples
    --------
    >>> rate_changes = pd.DataFrame({
    ...     'effective_date': ['2023-07-01'],
    ...     'rate_change': [1.05],
    ...     'fraction': [0.5]  # 6 months of 12-month policy
    ... })
    >>> onlevel = onlevel_adjustment(100000, rate_changes)
    >>> # Premium before change: 100000 × 0.5 = 50000 (no adjustment)
    >>> # Premium after change: 100000 × 0.5 / 1.05 = 47619 (adjust down)
    >>> # On-level: 50000 + 50000 = 100000 (but correctly: 97619)
    """
    onlevel_factor = 1.0
    
    for _, change in rate_changes.iterrows():
        # Compound rate changes
        fraction_before = 1 - change['fraction']
        fraction_after = change['fraction']
        
        # Weight by exposure fraction
        onlevel_factor *= (
            fraction_before +
            fraction_after / change['rate_change']
        )
    
    return historical_premium / onlevel_factor


def parallelogram_method(
    earned_premium_historical: float,
    rate_change_factor: float,
    rate_change_date: datetime,
    period_start: datetime,
    period_end: datetime
) -> float:
    """
    On-level earned premium using parallelogram method.
    
    Adjusts for rate changes that occurred mid-period.
    
    Parameters
    ----------
    earned_premium_historical : float
        Earned premium at historical rates
    rate_change_factor : float
        Rate change factor (e.g., 1.05 for +5%)
    rate_change_date : datetime
        Effective date of rate change
    period_start : datetime
        Start of earned premium period
    period_end : datetime
        End of earned premium period
    
    Returns
    -------
    float
        On-level earned premium
    """
    if rate_change_date <= period_start:
        # Change before period - apply full factor
        return earned_premium_historical / rate_change_factor
    elif rate_change_date >= period_end:
        # Change after period - no adjustment
        return earned_premium_historical
    else:
        # Change during period - weight by time
        total_days = (period_end - period_start).days
        days_before = (rate_change_date - period_start).days
        days_after = (period_end - rate_change_date).days
        
        weight_before = days_before / total_days
        weight_after = days_after / total_days
        
        # Premium on-level factor
        onlevel_factor = weight_before + weight_after / rate_change_factor
        
        return earned_premium_historical / onlevel_factor


def compute_trend_from_history(
    loss_data: pd.DataFrame,
    date_col: str,
    amount_col: str,
    method: str = 'exponential'
) -> float:
    """
    Estimate trend rate from historical loss data.
    
    Parameters
    ----------
    loss_data : pd.DataFrame
        Historical loss data
    date_col : str
        Date column name
    amount_col : str
        Loss amount column name
    method : str
        'exponential' for exponential fit, 'linear' for linear
    
    Returns
    -------
    float
        Estimated annual trend rate
    
    Examples
    --------
    >>> trend_rate = compute_trend_from_history(
    ...     losses,
    ...     date_col='accident_date',
    ...     amount_col='amount'
    ... )
    """
    df = loss_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Aggregate by year
    df['year'] = df[date_col].dt.year
    annual = df.groupby('year')[amount_col].sum().reset_index()
    annual = annual.sort_values('year')
    
    if len(annual) < 2:
        raise ValueError("Need at least 2 years of data to estimate trend")
    
    if method == 'exponential':
        # Fit exponential: y = a * e^(b*t) => log(y) = log(a) + b*t
        annual['log_amount'] = np.log(annual[amount_col] + 1)
        annual['t'] = range(len(annual))
        
        # Linear regression on log scale
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            annual['t'], annual['log_amount']
        )
        
        # Convert slope to annual rate
        annual_rate = np.exp(slope) - 1
        
    else:  # linear
        # Simple linear regression
        annual['t'] = range(len(annual))
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            annual['t'], annual[amount_col]
        )
        
        # Convert to percentage
        mean_amount = annual[amount_col].mean()
        annual_rate = slope / mean_amount if mean_amount > 0 else 0
    
    return float(annual_rate)