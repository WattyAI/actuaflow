"""
Severity Modeling Module

High-level workflow for claim severity modeling with best practices.

Features:
- Data preparation (positive claims only, optional large claim capping)
- Model fitting with comprehensive diagnostics
- Severity prediction
- Factor relativity extraction
- Model diagnostics and validation

Author: Michael Watson
License: MPL-2.0
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from actuaflow.glm.diagnostics import compute_diagnostics
from actuaflow.glm.models import SeverityGLM

logger = logging.getLogger(__name__)


class SeverityModel:
    """
    High-level severity modeling workflow.
    
    Automates common tasks:
    - Data preparation (positive claims only, optional large claim capping)
    - Model fitting
    - Diagnostics and validation
    
    Parameters
    ----------
    family : str, default='gamma'
        'gamma', 'inverse_gaussian', or 'lognormal'
    link : str, default='log'
        Link function
    
    Attributes
    ----------
    model_ : SeverityGLM
        Fitted GLM model
    diagnostics_ : dict
        Model diagnostics
    
    Examples
    --------
    >>> sev_model = SeverityModel(family='gamma')
    >>> sev_model.prepare_data(claims, policy_factors, policy_id='policy_id')
    >>> sev_model.fit(formula='amount ~ age_group + injury_type')
    >>> sev_model.summary()
    """
    
    def __init__(self, family: str = 'gamma', link: str = 'log') -> None:
        self.family = family
        self.link = link
        self.model_: Optional[SeverityGLM] = None
        self.data_: Optional[pd.DataFrame] = None
        self.diagnostics_: Optional[Dict[str, Any]] = None
    
    def prepare_data(
        self,
        claims_data: pd.DataFrame,
        policy_data: Optional[pd.DataFrame] = None,
        policy_id: str = 'policy_id',
        amount_col: str = 'amount',
        filter_zeros: bool = True,
        large_claim_threshold: Optional[float] = None,
        large_claim_percentile: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Prepare severity modeling data.
        
        Merges policy factors onto claims and optionally filters/caps claims.
        
        Parameters
        ----------
        claims_data : pd.DataFrame
            Claim-level data
        policy_data : pd.DataFrame, optional
            Policy-level data with rating factors
        policy_id : str
            Column name for policy identifier
        amount_col : str
            Column name for claim amount
        filter_zeros : bool, default=True
            Remove zero/negative claims
        large_claim_threshold : float, optional
            Fixed threshold for capping large claims
        large_claim_percentile : float, optional
            Percentile threshold (e.g., 0.95 for 95th percentile)
        
        Returns
        -------
        pd.DataFrame
            Severity modeling dataset
        """
        self.data_ = claims_data.copy()
        
        # Merge policy factors if provided
        if policy_data is not None:
            # Get rating factors (exclude claim-related columns)
            rating_factors = [
                col for col in policy_data.columns 
                if col not in ['claim_count', amount_col] and col != policy_id
            ]
            
            self.data_ = self.data_.merge(
                policy_data[[policy_id] + rating_factors],
                on=policy_id,
                how='left'
            )
        
        # Filter zeros
        if filter_zeros and amount_col in self.data_.columns:
            n_before = len(self.data_)
            self.data_ = self.data_[self.data_[amount_col] > 0]
            n_removed = n_before - len(self.data_)
            if n_removed > 0:
                print(f"Removed {n_removed} non-positive claims")
        
        # Apply large claim threshold
        if large_claim_percentile is not None:
            threshold = self.data_[amount_col].quantile(large_claim_percentile)
            n_before = len(self.data_)
            self.data_ = self.data_[self.data_[amount_col] <= threshold]
            n_capped = n_before - len(self.data_)
            if n_capped > 0:
                print(f"Capped {n_capped} claims above {threshold:,.2f} "
                      f"({large_claim_percentile*100:.0f}th percentile)")
        
        elif large_claim_threshold is not None:
            n_before = len(self.data_)
            self.data_ = self.data_[self.data_[amount_col] <= large_claim_threshold]
            n_capped = n_before - len(self.data_)
            if n_capped > 0:
                print(f"Capped {n_capped} claims above {large_claim_threshold:,.2f}")
        
        return self.data_
    
    def fit(
        self,
        formula: str,
        data: Optional[pd.DataFrame] = None,
        weights: Optional[str] = None
    ) -> "SeverityModel":
        """
        Fit severity model.
        
        Parameters
        ----------
        formula : str
            Model formula (e.g., 'amount ~ age_group + injury_type')
        data : pd.DataFrame, optional
            Data to fit. If None, uses data from prepare_data()
        weights : str, optional
            Weight column
        
        Returns
        -------
        self
        """
        if data is None and self.data_ is None:
            raise ValueError("No data available. Call prepare_data() first or provide data")
        
        fit_data = data if data is not None else self.data_
        
        self.model_ = SeverityGLM(family=self.family, link=self.link)
        self.model_.fit(fit_data, formula, weights=weights)
        
        # Compute diagnostics
        self.diagnostics_ = compute_diagnostics(self.model_, fit_data)
        
        return self
    
    def predict(self, newdata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate severity predictions.
        
        Parameters
        ----------
        newdata : pd.DataFrame, optional
            New data for prediction
        
        Returns
        -------
        np.ndarray
            Predicted claim severities
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")
        
        return self.model_.predict(newdata)
    
    def summary(self) -> pd.DataFrame:
        """Get model summary table."""
        if self.model_ is None:
            raise ValueError("Model must be fitted first")
        
        return self.model_.summary()
    
    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.
        
        Returns
        -------
        bool
            True if model has been fitted, False otherwise
        """
        return self.model_ is not None
    
    def get_relativities(self) -> pd.DataFrame:
        """
        Extract factor relativities from log-link model.
        
        Returns
        -------
        pd.DataFrame
            Factor relativities (base = 1.0)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")
        
        if self.link != 'log':
            raise ValueError("Relativities only available for log link")
        
        coef_df = self.summary()
        
        # Filter out intercept and extract relativities
        rel_df = coef_df[coef_df.index != 'Intercept'][['Relativity', 'P-value']].copy()
        rel_df['Significant'] = rel_df['P-value'] < 0.05
        
        return rel_df.sort_values('Relativity', ascending=False)
    
    def check_fit(self) -> Dict:
        """
        Comprehensive fit check.
        
        Returns
        -------
        dict
            Fit statistics and warnings
        """
        if self.diagnostics_ is None:
            raise ValueError("Model must be fitted first")
        
        fit_check = {
            'aic': self.diagnostics_['aic'],
            'bic': self.diagnostics_['bic'],
            'dispersion': self.diagnostics_['dispersion'],
            'warnings': [],
        }
        
        # Influential points
        if 'influential_count' in self.diagnostics_:
            if self.diagnostics_['influential_count'] > 0.05 * self.diagnostics_['nobs']:
                fit_check['warnings'].append(
                    f"High number of influential points "
                    f"({self.diagnostics_['influential_count']})"
                )
        
        # Convergence
        if not self.diagnostics_['converged']:
            fit_check['warnings'].append("Model did not converge")
        
        return fit_check
    
    def analyze_residuals(self) -> Dict:
        """
        Detailed residual analysis.
        
        Returns
        -------
        dict
            Residual statistics and plots data
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")
        
        residuals = self.model_.residuals('deviance')
        fitted = self.model_.result_.fitted_values
        
        return {
            'residuals': residuals,
            'fitted': fitted,
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'skewness': float(pd.Series(residuals).skew()),
            'kurtosis': float(pd.Series(residuals).kurtosis()),
            'qq_data': (self.diagnostics_ or {}).get('qq_data', {}),
        }