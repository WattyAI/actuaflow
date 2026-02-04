"""
Frequency Modeling Module

High-level workflow for claim frequency modeling with best practices.

Provides:
- Data preparation and aggregation
- Model fitting with automatic offset handling
- Model diagnostics and interpretation
- Variable selection using stepwise approach
- Relativity extraction for pricing

Author: Michael Watson
License: MPL-2.0
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from actuaflow.glm.diagnostics import check_overdispersion, compute_diagnostics
from actuaflow.glm.models import FrequencyGLM

logger = logging.getLogger(__name__)


class FrequencyModel:
    """
    High-level frequency modeling workflow.
    
    Automates common tasks:
    - Data preparation (claim counts per policy)
    - Model fitting with offset
    - Variable selection
    - Diagnostics and validation
    
    Parameters
    ----------
    family : str, default='poisson'
        'poisson' or 'negative_binomial'
    link : str, default='log'
        Link function
    
    Attributes
    ----------
    model_ : FrequencyGLM
        Fitted GLM model
    diagnostics_ : dict
        Model diagnostics
    
    Examples
    --------
    >>> freq_model = FrequencyModel(family='poisson')
    >>> freq_model.prepare_data(policies, claims, policy_id='policy_id')
    >>> freq_model.fit(formula='claim_count ~ age_group + vehicle_type', 
    ...                offset='exposure')
    >>> freq_model.summary()
    """
    
    def __init__(self, family: str = 'poisson', link: str = 'log') -> None:
        self.family = family
        self.link = link
        self.model_: Optional[FrequencyGLM] = None
        self.data_: Optional[pd.DataFrame] = None
        self.diagnostics_: Optional[Dict[str, Any]] = None
        self.overdispersion_: Optional[Dict[str, Any]] = None
    
    def prepare_data(
        self,
        policy_data: pd.DataFrame,
        claims_data: pd.DataFrame,
        policy_id: str = 'policy_id',
        policy_id_policy: Optional[str] = None,
        policy_id_claims: Optional[str] = None,
        claim_count_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare frequency modeling data.
        
        Merges claim counts onto policy data. Policies with no claims get count=0.
        
        Parameters
        ----------
        policy_data : pd.DataFrame
            Policy-level data with rating factors and exposure
        claims_data : pd.DataFrame
            Claim-level data
        policy_id : str, default='policy_id'
            Column name for policy identifier (if same in both datasets)
        policy_id_policy : str, optional
            Policy ID column name in `policy_data` (overrides `policy_id` if provided)
        policy_id_claims : str, optional
            Policy ID column name in `claims_data` (overrides `policy_id` if provided)
        claim_count_col : str, optional
            If policy data already has claim counts, specify column name
        
        Returns
        -------
        pd.DataFrame
            Frequency modeling dataset (one row per policy with claim counts)
        """
        # Determine actual column names to use
        pol_id_policy = policy_id_policy if policy_id_policy is not None else policy_id
        pol_id_claims = policy_id_claims if policy_id_claims is not None else policy_id

        if claim_count_col and claim_count_col in policy_data.columns:
            # Use existing claim count
            self.data_ = policy_data.copy()
        else:
            # Count claims per policy using claims dataset column name
            claims_per_policy = claims_data.groupby(pol_id_claims).size().reset_index(
                name='claim_count'
            )

            # Rename claims policy_id column to match policy data if different
            if pol_id_claims != pol_id_policy:
                claims_per_policy = claims_per_policy.rename(
                    columns={pol_id_claims: pol_id_policy}
                )

            # Merge onto policies (LEFT join to keep all policies)
            self.data_ = policy_data.merge(
                claims_per_policy,
                on=pol_id_policy,
                how='left'
            )

            # Fill zeros
            self.data_['claim_count'] = self.data_['claim_count'].fillna(0).astype(int)
        
        return self.data_
    
    def fit(
        self,
        formula: str,
        data: Optional[pd.DataFrame] = None,
        offset: Optional[str] = None,
        weights: Optional[str] = None
    ) -> "FrequencyModel":
        """
        Fit frequency model.
        
        Parameters
        ----------
        formula : str
            Model formula (e.g., 'claim_count ~ age_group + vehicle_type')
        data : pd.DataFrame, optional
            Data to fit. If None, uses data from prepare_data()
        offset : str, optional
            Offset column (typically 'exposure')
        weights : str, optional
            Weight column
        
        Returns
        -------
        self
        """
        if data is None and self.data_ is None:
            raise ValueError("No data available. Call prepare_data() first or provide data")
        
        fit_data = data if data is not None else self.data_
        
        self.model_ = FrequencyGLM(family=self.family, link=self.link)
        self.model_.fit(fit_data, formula, offset=offset, weights=weights)
        
        # Compute diagnostics - handle potential attribute errors
        try:
            self.diagnostics_ = compute_diagnostics(self.model_, fit_data)
        except Exception as e:
            # Fallback to basic diagnostics if compute_diagnostics fails
            logger.warning(f"Diagnostic computation failed: {e}")
            self.diagnostics_ = self._basic_diagnostics()
        
        # Check for overdispersion with error handling
        try:
            self.overdispersion_ = check_overdispersion(self.model_)
        except Exception as e:
            # Fallback if overdispersion check fails
            logger.warning(f"Overdispersion check failed: {e}")
            self.overdispersion_ = self._basic_overdispersion()
        
        return self
    
    def predict(self, newdata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate frequency predictions.
        
        Parameters
        ----------
        newdata : pd.DataFrame, optional
            New data for prediction
        
        Returns
        -------
        np.ndarray
            Predicted claim frequencies
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
            Factor relativities (base = 1.0) with p-values
        
        Raises
        ------
        ValueError
            If model not fitted or link is not 'log'
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
    
    def check_fit(self) -> Dict[str, Any]:
        """
        Comprehensive fit check.
        
        Returns
        -------
        dict
            Fit statistics and warnings about model issues
            
        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if self.diagnostics_ is None:
            raise ValueError("Model must be fitted first")
        
        fit_check = {
            'aic': self.diagnostics_['aic'],
            'bic': self.diagnostics_['bic'],
            'dispersion': self.diagnostics_['dispersion'],
            'warnings': [],
        }
        
        # Dispersion warning
        if 'dispersion' in self.diagnostics_ and self.diagnostics_['dispersion'] > 1.5:
            fit_check['warnings'].append(
                f"Overdispersion detected ({self.diagnostics_['dispersion']:.3f}). "
                "Consider Negative Binomial."
            )
        
        # Influential points
        if 'influential_count' in self.diagnostics_:
            if self.diagnostics_['influential_count'] > 0.05 * self.diagnostics_['nobs']:
                fit_check['warnings'].append(
                    f"High number of influential points "
                    f"({self.diagnostics_['influential_count']})"
                )
        
        # Convergence
        if 'converged' in self.diagnostics_ and not self.diagnostics_['converged']:
            fit_check['warnings'].append("Model did not converge")
        
        return fit_check
    
    def _basic_diagnostics(self) -> Dict[str, Any]:
        """Provide basic diagnostics when full diagnostics fail."""
        if self.model_ is None:
            return {}
        
        # Get basic model information
        result = self.model_.result_
        diagnostics = {}
        
        # Try to get basic statistics
        try:
            diagnostics['aic'] = getattr(result, 'aic', np.nan)
            diagnostics['bic'] = getattr(result, 'bic', np.nan)
            diagnostics['nobs'] = getattr(result, 'n_obs', len(self.data_) if self.data_ is not None else 0)
            diagnostics['converged'] = getattr(result, 'converged', True)
            diagnostics['dispersion'] = 1.0  # Default value
        except:
            diagnostics['aic'] = np.nan
            diagnostics['bic'] = np.nan
            diagnostics['nobs'] = len(self.data_) if self.data_ is not None else 0
            diagnostics['converged'] = True
            diagnostics['dispersion'] = 1.0
        
        return diagnostics
    
    def _basic_overdispersion(self) -> Dict[str, Any]:
        """Provide basic overdispersion check when full check fails."""
        return {
            'dispersion': 1.0,
            'p_value': 1.0,
            'significant': False
        }
    
    def variable_selection(
        self,
        candidate_vars: List[str],
        response: str = 'claim_count',
        criterion: str = 'aic',
        direction: str = 'both',
        data: Optional[pd.DataFrame] = None,
        offset: Optional[str] = None
    ) -> Dict:
        """
        Automated variable selection using stepwise approach.
        
        Parameters
        ----------
        candidate_vars : list of str
            Candidate predictor variables
        response : str
            Response variable name
        criterion : str
            'aic' or 'bic'
        direction : str
            'forward', 'backward', or 'both'
        data : pd.DataFrame, optional
            Data to use
        offset : str, optional
            Offset column
        
        Returns
        -------
        dict
            Selected variables and selection history
        """
        from actuaflow.glm.models import BaseGLM
        
        fit_data = data if data is not None else self.data_
        if fit_data is None:
            raise ValueError("No data available")
        
        current_vars: list[str] = []
        remaining_vars = candidate_vars.copy()
        best_criterion = np.inf
        history = []
        
        # Prepare offset
        if offset:
            log_offset = np.log(fit_data[offset].clip(lower=1e-10))
        else:
            log_offset = None
        
        for iteration in range(50):  # Max iterations
            improved = False
            
            # Forward step
            if direction in ['forward', 'both'] and remaining_vars:
                best_add = None
                best_add_criterion = best_criterion
                
                for var in remaining_vars:
                    test_vars: list[str] = current_vars + [var]
                    formula = f"{response} ~ {' + '.join(test_vars)}"
                    
                    try:
                        temp_model = FrequencyGLM(family=self.family, link=self.link)
                        temp_model.fit(fit_data, formula, offset=offset)
                        
                        crit = temp_model.result_.aic if criterion == 'aic' else temp_model.result_.bic
                        
                        if crit < best_add_criterion:
                            best_add_criterion = crit
                            best_add = var
                    except:
                        continue
                
                if best_add and best_add_criterion < best_criterion:
                    current_vars.append(best_add)
                    remaining_vars.remove(best_add)
                    best_criterion = best_add_criterion
                    improved = True
                    history.append({
                        'step': iteration + 1,
                        'action': 'add',
                        'variable': best_add,
                        criterion: best_criterion
                    })
            
            # Backward step
            if direction in ['backward', 'both'] and current_vars:
                best_remove = None
                best_remove_criterion = best_criterion
                
                for var in current_vars:
                    test_vars = [v for v in current_vars if v != var]
                    if test_vars:
                        formula = f"{response} ~ {' + '.join(test_vars)}"
                    else:
                        formula = f"{response} ~ 1"
                    
                    try:
                        temp_model = FrequencyGLM(family=self.family, link=self.link)
                        temp_model.fit(fit_data, formula, offset=offset)
                        
                        crit = temp_model.result_.aic if criterion == 'aic' else temp_model.result_.bic
                        
                        if crit < best_remove_criterion:
                            best_remove_criterion = crit
                            best_remove = var
                    except:
                        continue
                
                if best_remove and best_remove_criterion < best_criterion:
                    current_vars.remove(best_remove)
                    remaining_vars.append(best_remove)
                    best_criterion = best_remove_criterion
                    improved = True
                    history.append({
                        'step': iteration + 1,
                        'action': 'remove',
                        'variable': best_remove,
                        criterion: best_criterion
                    })
            
            if not improved:
                break
        
        final_formula = (
            f"{response} ~ {' + '.join(current_vars)}" if current_vars 
            else f"{response} ~ 1"
        )
        
        return {
            'selected_variables': current_vars,
            'final_formula': final_formula,
            'final_criterion': best_criterion,
            'history': history,
            'iterations': iteration + 1
        }