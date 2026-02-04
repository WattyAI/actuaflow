"""
Price Elasticity Analysis

Tools for estimating and applying price elasticity of demand.

Features:
- Demand elasticity estimation from historical data
- Elasticity curve computation
- Optimal price determination
- Price sensitivity analysis
- Revenue optimization

Author: Michael Watson
License: MPL-2.0
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)


def estimate_demand_elasticity(
    price_history: pd.DataFrame,
    price_col: str,
    volume_col: str,
    method: str = 'log_log'
) -> Dict[str, Any]:
    """
    Estimate price elasticity of demand from historical data.
    
    Elasticity = % change in quantity / % change in price
    
    Parameters
    ----------
    price_history : pd.DataFrame
        Historical price and volume data
    price_col : str
        Price column name
    volume_col : str
        Volume/quantity column name
    method : str
        'log_log' for constant elasticity or 'linear' for simple regression
    
    Returns
    -------
    dict
        Elasticity estimate with:
        - elasticity: Point estimate
        - std_error: Standard error
        - r_squared: Model fit
        - p_value: Statistical significance
    
    Examples
    --------
    >>> elasticity = estimate_demand_elasticity(
    ...     historical_data,
    ...     price_col='average_premium',
    ...     volume_col='policy_count'
    ... )
    >>> print(f"Elasticity: {elasticity['elasticity']:.3f}")
    """
    df = price_history.copy()
    
    # Remove zeros and negatives
    df = df[(df[price_col] > 0) & (df[volume_col] > 0)]
    
    if len(df) < 3:
        raise ValueError("Need at least 3 data points to estimate elasticity")
    
    if method == 'log_log':
        # Log-log regression: log(Q) = a + elasticity * log(P)
        log_price = np.log(df[price_col])
        log_volume = np.log(df[volume_col])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_price, log_volume
        )
        
        return {
            'elasticity': float(slope),
            'std_error': float(std_err),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'method': 'log_log',
            'interpretation': (
                'Demand is elastic' if slope < -1 else
                'Demand is inelastic' if slope > -1 else
                'Unit elastic'
            )
        }
    
    elif method == 'linear':
        # Simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df[price_col], df[volume_col]
        )
        
        # Convert to elasticity at mean
        mean_price = df[price_col].mean()
        mean_volume = df[volume_col].mean()
        elasticity = slope * mean_price / mean_volume
        
        return {
            'elasticity': float(elasticity),
            'slope': float(slope),
            'std_error': float(std_err),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'method': 'linear'
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_elasticity_curve(
    current_price: float,
    current_volume: float,
    elasticity: float,
    price_range: Tuple[float, float],
    n_points: int = 20
) -> pd.DataFrame:
    """
    Compute demand curve using constant elasticity assumption.
    
    Q = Q0 × (P / P0) ^ elasticity
    
    Parameters
    ----------
    current_price : float
        Current price level
    current_volume : float
        Current volume (policies, units, etc.)
    elasticity : float
        Price elasticity (typically negative)
    price_range : tuple
        (min_price, max_price) for curve
    n_points : int
        Number of points on curve
    
    Returns
    -------
    pd.DataFrame
        Elasticity curve with columns:
        - price: Price levels
        - volume: Expected volume
        - revenue: Price × Volume
        - price_change_pct: % change from current
        - volume_change_pct: % change from current
    
    Examples
    --------
    >>> curve = compute_elasticity_curve(
    ...     current_price=100,
    ...     current_volume=10000,
    ...     elasticity=-1.5,
    ...     price_range=(80, 120)
    ... )
    """
    prices = np.linspace(price_range[0], price_range[1], n_points)
    
    # Compute volumes using elasticity formula
    volumes = current_volume * (prices / current_price) ** elasticity
    
    # Revenue
    revenues = prices * volumes
    
    # Percentage changes
    price_change_pct = (prices - current_price) / current_price * 100
    volume_change_pct = (volumes - current_volume) / current_volume * 100
    
    curve = pd.DataFrame({
        'price': prices,
        'volume': volumes,
        'revenue': revenues,
        'price_change_pct': price_change_pct,
        'volume_change_pct': volume_change_pct
    })
    
    return curve


def optimal_price(
    cost_per_unit: float,
    current_price: float,
    current_volume: float,
    elasticity: float,
    fixed_costs: float = 0
) -> Dict[str, Any]:
    """
    Find profit-maximizing price using elasticity.
    
    Maximizes: Profit = (P - C) × Q - FC
    where: Q = Q0 × (P / P0) ^ elasticity
    
    Parameters
    ----------
    cost_per_unit : float
        Variable cost per unit (pure premium for insurance)
    current_price : float
        Current price
    current_volume : float
        Current volume
    elasticity : float
        Price elasticity
    fixed_costs : float
        Fixed costs
    
    Returns
    -------
    dict
        Optimization results:
        - optimal_price: Profit-maximizing price
        - optimal_volume: Expected volume at optimal price
        - optimal_profit: Expected profit
        - price_change_pct: % change from current
    
    Examples
    --------
    >>> result = optimal_price(
    ...     cost_per_unit=65,
    ...     current_price=100,
    ...     current_volume=10000,
    ...     elasticity=-1.5
    ... )
    """
    def profit_function(price):
        volume = current_volume * (price / current_price) ** elasticity
        profit = (price - cost_per_unit) * volume - fixed_costs
        return -profit  # Negative for minimization
    
    # Optimize
    result = optimize.minimize_scalar(
        profit_function,
        bounds=(cost_per_unit * 1.01, current_price * 2),
        method='bounded'
    )
    
    optimal_p = result.x
    optimal_v = current_volume * (optimal_p / current_price) ** elasticity
    optimal_profit = (optimal_p - cost_per_unit) * optimal_v - fixed_costs
    
    return {
        'optimal_price': float(optimal_p),
        'optimal_volume': float(optimal_v),
        'optimal_profit': float(optimal_profit),
        'price_change_pct': float((optimal_p - current_price) / current_price * 100),
        'volume_change_pct': float((optimal_v - current_volume) / current_volume * 100),
        'current_price': current_price,
        'current_volume': current_volume,
        'current_profit': float((current_price - cost_per_unit) * current_volume - fixed_costs)
    }


def retention_curve(
    price_change_pct: np.ndarray,
    base_retention: float = 0.85,
    sensitivity: float = 0.5
) -> np.ndarray:
    """
    Model customer retention as function of price change.
    
    Uses logistic function to model retention probability.
    
    Parameters
    ----------
    price_change_pct : array-like
        Price change percentages
    base_retention : float
        Retention rate at 0% price change
    sensitivity : float
        Sensitivity to price changes (higher = more sensitive)
    
    Returns
    -------
    np.ndarray
        Retention probabilities
    
    Examples
    --------
    >>> price_changes = np.array([-10, -5, 0, 5, 10, 15])
    >>> retention = retention_curve(price_changes, base_retention=0.85)
    """
    # Logistic function centered at 0
    # Retention decreases with price increases
    retention: np.ndarray = np.asarray(
        base_retention / (1 + sensitivity * np.maximum(price_change_pct, 0) / 100),
        dtype=np.float64
    )
    
    # Increases don't improve retention beyond base
    retention = np.clip(retention, 0, base_retention).astype(np.float64)
    
    return retention


def revenue_optimization(
    portfolio: pd.DataFrame,
    pure_premium_col: str,
    current_premium_col: str,
    elasticity_by_segment: Dict[str, float],
    segment_col: str,
    price_change_range: Tuple[float, float] = (0.9, 1.1),
    target_loss_ratio: float = 0.65
) -> Dict:
    """
    Optimize revenue considering elasticity by segment.
    
    Parameters
    ----------
    portfolio : pd.DataFrame
        Portfolio data
    pure_premium_col : str
        Pure premium (cost) column
    current_premium_col : str
        Current premium column
    elasticity_by_segment : dict
        {segment_value: elasticity}
    segment_col : str
        Segmentation column
    price_change_range : tuple
        Allowed price change range (multipliers)
    target_loss_ratio : float
        Target loss ratio constraint
    
    Returns
    -------
    dict
        Optimization results by segment
    """
    results = {}
    
    for segment, elasticity in elasticity_by_segment.items():
        segment_data = portfolio[portfolio[segment_col] == segment]
        
        if len(segment_data) == 0:
            continue
        
        current_price = segment_data[current_premium_col].mean()
        current_volume = len(segment_data)
        cost_per_unit = segment_data[pure_premium_col].mean()
        
        # Constraint: price must cover cost / target_loss_ratio
        min_price = cost_per_unit / target_loss_ratio
        
        # Find optimal within constraints
        opt = optimal_price(
            cost_per_unit=cost_per_unit,
            current_price=current_price,
            current_volume=current_volume,
            elasticity=elasticity
        )
        
        # Apply price change range constraints
        optimal_price_constrained = np.clip(
            opt['optimal_price'],
            current_price * price_change_range[0],
            current_price * price_change_range[1]
        )
        
        # Ensure minimum price
        optimal_price_constrained = max(optimal_price_constrained, min_price)
        
        # Recompute at constrained price
        optimal_volume = current_volume * (
            optimal_price_constrained / current_price
        ) ** elasticity
        
        results[segment] = {
            'current_price': current_price,
            'optimal_price': optimal_price_constrained,
            'price_change_pct': (
                (optimal_price_constrained - current_price) / current_price * 100
            ),
            'expected_volume': optimal_volume,
            'volume_change_pct': (optimal_volume - current_volume) / current_volume * 100,
            'elasticity': elasticity
        }
    
    return results