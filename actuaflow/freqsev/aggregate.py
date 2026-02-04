"""
Aggregate (Combined) Frequency-Severity Models

Combines frequency and severity models to compute pure premium and loaded premium.

Features:
- Combined frequency-severity models
- Pure premium calculation
- Factor table creation
- Premium loading application
- Elasticity computations

Author: Michael Watson
License: MPL-2.0
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AggregateModel:
    """
    Combined frequency-severity model for pure premium calculation.
    
    Pure Premium = Frequency × Severity
    
    Parameters
    ----------
    frequency_model : FrequencyModel
        Fitted frequency model
    severity_model : SeverityModel
        Fitted severity model
    
    Attributes
    ----------
    base_frequency_ : float
        Base frequency (intercept on response scale)
    base_severity_ : float
        Base severity (intercept on response scale)
    base_pure_premium_ : float
        Base pure premium (frequency × severity)
    
    Examples
    --------
    >>> agg_model = AggregateModel(freq_model, sev_model)
    >>> factor_table = agg_model.create_factor_table()
    >>> premium = agg_model.predict_pure_premium(newdata)
    """
    
    def __init__(self, frequency_model: Any, severity_model: Any) -> None:
        self.frequency_model = frequency_model
        self.severity_model = severity_model
        
        # Extract base rates from intercepts
        freq_intercept = frequency_model.model_.coefficients_.get('Intercept', 0)
        sev_intercept = severity_model.model_.coefficients_.get('Intercept', 0)
        
        # Convert to response scale
        if frequency_model.link == 'log':
            base_freq: float = float(np.exp(freq_intercept))
        else:
            base_freq = float(freq_intercept)
        
        if severity_model.link == 'log':
            base_sev: float = float(np.exp(sev_intercept))
        else:
            base_sev = float(sev_intercept)
        
        self.base_frequency_ = base_freq
        self.base_severity_ = base_sev
        
        self.base_pure_premium_: float = self.base_frequency_ * self.base_severity_
    
    def create_factor_table(self) -> pd.DataFrame:
        """
        Create combined rating factor table.
        
        Multiplies frequency and severity relativities for each factor level.
        
        Returns
        -------
        pd.DataFrame
            Combined factor table with columns:
            - Variable: Factor name
            - Level: Factor level
            - Frequency_Relativity: Frequency relativity
            - Severity_Relativity: Severity relativity
            - Combined_Relativity: Product of freq × sev
            - Pure_Premium: Base premium × combined relativity
        """
        freq_rel = self.frequency_model.get_relativities()
        sev_rel = self.severity_model.get_relativities()
        
        # Extract all variables
        all_vars = set()
        for idx in freq_rel.index:
            if '[' in str(idx):
                var_name = str(idx).split('[')[0]
                all_vars.add(var_name)
        
        for idx in sev_rel.index:
            if '[' in str(idx):
                var_name = str(idx).split('[')[0]
                all_vars.add(var_name)
        
        # Build factor table
        factors = []
        
        for var in sorted(all_vars):
            # Get frequency levels
            freq_levels = {
                idx: val for idx, val in freq_rel['Relativity'].items()
                if str(idx).startswith(f"{var}[")
            }
            
            # Get severity levels
            sev_levels = {
                idx: val for idx, val in sev_rel['Relativity'].items()
                if str(idx).startswith(f"{var}[")
            }
            
            # All unique levels
            all_levels = set(list(freq_levels.keys()) + list(sev_levels.keys()))
            
            for level_key in sorted(all_levels):
                level_name = str(level_key).split('[')[1].rstrip(']')
                
                freq_rel_val = freq_levels.get(level_key, 1.0)
                sev_rel_val = sev_levels.get(level_key, 1.0)
                
                combined_rel = freq_rel_val * sev_rel_val
                pure_premium = self.base_pure_premium_ * combined_rel
                
                factors.append({
                    'Variable': var,
                    'Level': level_name,
                    'Frequency_Relativity': freq_rel_val,
                    'Severity_Relativity': sev_rel_val,
                    'Combined_Relativity': combined_rel,
                    'Pure_Premium': pure_premium
                })
        
        return pd.DataFrame(factors)
    
    def predict_pure_premium(
        self,
        data: pd.DataFrame,
        exposure: Optional[str] = None
    ) -> pd.Series:
        """
        Predict pure premium for new data.
        
        Pure Premium = Predicted Frequency × Predicted Severity
        
        Parameters
        ----------
        data : pd.DataFrame
            New data for prediction
        exposure : str, optional
            Exposure column (if provided, returns total expected loss)
        
        Returns
        -------
        pd.Series
            Pure premium predictions
        """
        freq_pred = self.frequency_model.predict(data)
        sev_pred = self.severity_model.predict(data)
        
        pure_premium = freq_pred * sev_pred
        
        if exposure:
            pure_premium = pure_premium * data[exposure]
        
        return pd.Series(pure_premium, index=data.index)


def combine_models(
    frequency_model: Any,
    severity_model: Any
) -> "AggregateModel":
    """
    Convenience function to combine frequency and severity models.
    
    Parameters
    ----------
    frequency_model : FrequencyModel
        Fitted frequency model
    severity_model : SeverityModel
        Fitted severity model
    
    Returns
    -------
    AggregateModel
        Combined model
    """
    return AggregateModel(frequency_model, severity_model)


def calculate_premium(
    pure_premium: pd.Series,
    loadings: Dict[str, float],
    exposure: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Calculate loaded premium from pure premium with sequential loadings.
    
    Sequential Loading Formula:
    1. Adjust for inflation: PP × (1 + inflation)
    2. Add expenses: / (1 - expense_ratio)
    3. Add commission: / (1 - commission)
    4. Add profit: × (1 + profit_margin)
    5. Add taxes: / (1 - tax_rate)
    
    Parameters
    ----------
    pure_premium : pd.Series
        Pure premium (frequency × severity)
    loadings : dict
        Dictionary of loading factors:
        - inflation: Expected claim inflation rate
        - expense_ratio: Operating expense ratio
        - commission: Agent commission rate
        - profit_margin: Target profit margin
        - tax_rate: Premium tax rate
    exposure : pd.Series, optional
        Exposure for computing per-unit rates
    
    Returns
    -------
    pd.DataFrame
        Premium breakdown with columns:
        - pure_premium: Original pure premium
        - after_inflation: After inflation adjustment
        - after_expenses: After expense loading
        - after_commission: After commission loading
        - after_profit: After profit margin
        - loaded_premium: Final loaded premium
        - premium_per_unit: (if exposure provided)
    """
    inflation = loadings.get('inflation', 0.0)
    expense_ratio = loadings.get('expense_ratio', 0.0)
    commission = loadings.get('commission', 0.0)
    profit_margin = loadings.get('profit_margin', 0.0)
    tax_rate = loadings.get('tax_rate', 0.0)
    
    # Sequential loading
    after_inflation = pure_premium * (1 + inflation)
    after_expenses = after_inflation / max(1 - expense_ratio, 0.01)
    after_commission = after_expenses / max(1 - commission, 0.01)
    after_profit = after_commission * (1 + profit_margin)
    loaded_premium = after_profit / max(1 - tax_rate, 0.01)
    
    result = pd.DataFrame({
        'pure_premium': pure_premium,
        'after_inflation': after_inflation,
        'after_expenses': after_expenses,
        'after_commission': after_commission,
        'after_profit': after_profit,
        'loaded_premium': loaded_premium,
    })
    
    if exposure is not None:
        result['premium_per_unit'] = loaded_premium / exposure.clip(lower=0.01)
    
    return result


def premium_waterfall(
    pure_premium_total: float,
    loadings: Dict[str, float]
) -> pd.DataFrame:
    """
    Create premium loading waterfall for visualization.
    
    Shows step-by-step premium build-up from pure premium to loaded premium.
    
    Parameters
    ----------
    pure_premium_total : float
        Total pure premium
    loadings : dict
        Loading factors
    
    Returns
    -------
    pd.DataFrame
        Waterfall table with columns:
        - Step: Loading step name
        - Amount: Premium at this step
        - Increase: Incremental increase
        - Increase_Pct: Increase as % of pure premium
    """
    inflation = loadings.get('inflation', 0.0)
    expense_ratio = loadings.get('expense_ratio', 0.0)
    commission = loadings.get('commission', 0.0)
    profit_margin = loadings.get('profit_margin', 0.0)
    tax_rate = loadings.get('tax_rate', 0.0)
    
    # Calculate each step
    pure = pure_premium_total
    after_inflation = pure * (1 + inflation)
    after_expenses = after_inflation / max(1 - expense_ratio, 0.01)
    after_commission = after_expenses / max(1 - commission, 0.01)
    after_profit = after_commission * (1 + profit_margin)
    final = after_profit / max(1 - tax_rate, 0.01)
    
    waterfall = pd.DataFrame({
        'Step': [
            'Pure Premium',
            'After Inflation',
            'After Expenses',
            'After Commission',
            'After Profit',
            'Final Premium'
        ],
        'Amount': [
            pure,
            after_inflation,
            after_expenses,
            after_commission,
            after_profit,
            final
        ]
    })
    
    waterfall['Increase'] = waterfall['Amount'].diff().fillna(0)
    waterfall['Increase_Pct'] = (waterfall['Increase'] / pure * 100).round(2)
    
    return waterfall