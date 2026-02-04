"""
Model Diagnostics for GLMs

Comprehensive diagnostic functions for model validation:
- AIC, BIC, deviance
- VIF (Variance Inflation Factor) for multicollinearity
- Lift curves and Gini index for predictive power
- Residual analysis
- Influence diagnostics (leverage, Cook's distance)

Author: Michael Watson
License: MPL-2.0
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def compute_diagnostics(model: Any, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive diagnostics for a fitted GLM model.
    
    Parameters
    ----------
    model : BaseGLM
        Fitted GLM model
    data : pd.DataFrame
        Data used for fitting
    
    Returns
    -------
    dict
        Dictionary of diagnostic measures
    """
    if not model.fitted_:
        raise ValueError("Model must be fitted first")
    
    result = model.result_
    backend_result = model._backend_result  # Get the statsmodels backend result for residuals
    
    # Basic fit statistics
    diag = model.diagnostics()
    
    # VIF
    try:
        vif_dict = compute_vif(result)
        diag['vif'] = vif_dict
    except:
        diag['vif'] = None
    
    # Influence measures
    try:
        influence = result.get_influence()
        diag['leverage'] = influence.hat_matrix_diag
        diag['cooks_distance'] = influence.cooks_distance[0]
        
        # Flag influential points
        leverage_threshold = 2 * result.df_model / result.nobs
        cooks_threshold = 4 / result.nobs
        
        diag['high_leverage_count'] = int((diag['leverage'] > leverage_threshold).sum())
        diag['influential_count'] = int((diag['cooks_distance'] > cooks_threshold).sum())
    except:
        diag['leverage'] = None
        diag['cooks_distance'] = None
    
    # Residual analysis
    diag['residuals'] = {
        'deviance': backend_result.resid_deviance.tolist() if hasattr(backend_result.resid_deviance, 'tolist') else backend_result.resid_deviance,
        'pearson': backend_result.resid_pearson.tolist() if hasattr(backend_result.resid_pearson, 'tolist') else backend_result.resid_pearson,
        'mean': float(backend_result.resid_deviance.mean()),
        'std': float(backend_result.resid_deviance.std()),
        'min': float(backend_result.resid_deviance.min()),
        'max': float(backend_result.resid_deviance.max()),
    }
    
    # QQ plot data
    residuals_std = (backend_result.resid_deviance - backend_result.resid_deviance.mean()) / backend_result.resid_deviance.std()
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_std)))
    sample_quantiles = np.sort(residuals_std)
    
    diag['qq_data'] = {
        'theoretical': theoretical_quantiles,
        'sample': sample_quantiles,
    }
    
    return dict(diag)


def compute_vif(glm_result: Any) -> Dict[str, float]:
    """
    Calculate Variance Inflation Factor for each predictor.
    
    VIF > 10 indicates severe multicollinearity.
    VIF 5-10 indicates moderate multicollinearity.
    
    Parameters
    ----------
    glm_result : statsmodels GLMResults
        Fitted GLM results
    
    Returns
    -------
    dict
        Variable name -> VIF mapping
    """
    # Get design matrix (exclude intercept)
    X = glm_result.model.exog[:, 1:]
    var_names = glm_result.model.exog_names[1:]
    
    if X.shape[1] == 0:
        return {}
    
    vif_dict = {}
    for i, name in enumerate(var_names):
        try:
            vif = variance_inflation_factor(X, i)
            vif_dict[name] = float(vif)
        except:
            vif_dict[name] = np.nan
    
    return vif_dict


def compute_lift_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lift curve for model predictions.
    
    The lift curve shows how well the model separates high-risk from low-risk
    by comparing predicted vs actual rates in deciles.
    
    Parameters
    ----------
    y_true : array-like
        True response values
    y_pred : array-like
        Predicted values
    n_bins : int, default=10
        Number of bins (typically deciles)
    
    Returns
    -------
    bins : np.ndarray
        Bin edges
    actual_rates : np.ndarray
        Actual rates in each bin
    predicted_rates : np.ndarray
        Predicted rates in each bin
    
    Examples
    --------
    >>> bins, actual, predicted = compute_lift_curve(y_true, y_pred, n_bins=10)
    >>> lift = actual / actual.mean()  # Lift relative to average
    """
    # Create bins based on predictions
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Sort by prediction and assign bins
    df = df.sort_values('y_pred', ascending=False).reset_index(drop=True)
    df['bin'] = pd.qcut(df.index, q=n_bins, labels=False, duplicates='drop')
    
    # Compute rates per bin
    grouped = df.groupby('bin').agg({
        'y_true': 'mean',
        'y_pred': 'mean'
    }).reset_index()
    
    bins = grouped['bin'].values
    actual_rates = grouped['y_true'].values
    predicted_rates = grouped['y_pred'].values
    
    return bins, actual_rates, predicted_rates


def compute_gini_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Gini index for model predictions.
    
    The Gini index measures the model's ability to discriminate between
    high and low risk. It ranges from 0 (no discrimination) to 1 (perfect).
    
    Gini = 2 * AUC - 1
    
    Parameters
    ----------
    y_true : array-like
        True binary outcomes (0/1)
    y_pred : array-like
        Predicted probabilities
    
    Returns
    -------
    float
        Gini index (0 to 1)
    
    Examples
    --------
    >>> gini = compute_gini_index(has_claim, predicted_freq)
    >>> print(f"Gini coefficient: {gini:.3f}")
    """
    try:
        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
        return float(gini)
    except:
        return np.nan


def compute_lorenz_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lorenz curve for concentration analysis.
    
    Shows cumulative actual vs cumulative predicted when sorted by prediction.
    
    Parameters
    ----------
    y_true : array-like
        True response values
    y_pred : array-like
        Predicted values
    
    Returns
    -------
    cum_pop : np.ndarray
        Cumulative population proportion
    cum_actual : np.ndarray
        Cumulative actual proportion
    """
    # Sort by prediction
    sorted_idx = np.argsort(y_pred)[::-1]
    y_sorted = y_true[sorted_idx]
    
    # Cumulative sums
    cum_actual = np.cumsum(y_sorted) / np.sum(y_sorted)
    cum_pop = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    
    return cum_pop, cum_actual


def compute_double_lift(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute double lift chart comparing actual vs predicted rates.
    
    Standard actuarial diagnostic showing:
    - Actual frequency/severity by predicted decile
    - Predicted frequency/severity by decile
    - Lift (actual / overall average)
    - Prediction quality
    
    Parameters
    ----------
    y_true : array-like
        True response values
    y_pred : array-like
        Predicted values
    n_bins : int
        Number of bins (deciles)
    
    Returns
    -------
    pd.DataFrame
        Double lift table with columns:
        - bin: Bin number
        - n_obs: Number of observations
        - actual_mean: Actual average in bin
        - predicted_mean: Predicted average in bin
        - actual_lift: Actual / overall average
        - predicted_lift: Predicted / overall average
        - ratio: Actual / Predicted
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Sort by prediction and create bins
    df = df.sort_values('y_pred', ascending=False).reset_index(drop=True)
    df['bin'] = pd.qcut(df.index, q=n_bins, labels=False, duplicates='drop')
    
    # Overall averages
    overall_actual = df['y_true'].mean()
    overall_pred = df['y_pred'].mean()
    
    # Group statistics
    lift_table = df.groupby('bin').agg({
        'y_true': ['count', 'mean'],
        'y_pred': 'mean'
    }).reset_index()
    
    lift_table.columns = ['bin', 'n_obs', 'actual_mean', 'predicted_mean']
    
    # Compute lifts
    lift_table['actual_lift'] = lift_table['actual_mean'] / overall_actual
    lift_table['predicted_lift'] = lift_table['predicted_mean'] / overall_pred
    lift_table['ratio'] = lift_table['actual_mean'] / lift_table['predicted_mean']
    
    return lift_table


def check_overdispersion(model) -> Dict:
    """
    Check for overdispersion in count models.
    
    Overdispersion occurs when variance > mean, violating Poisson assumptions.
    
    Parameters
    ----------
    model : FrequencyGLM
        Fitted frequency model
    
    Returns
    -------
    dict
        Dispersion statistics and interpretation
    """
    if not model.fitted_:
        raise ValueError("Model must be fitted first")
    
    result = model.result_
    backend_result = model._backend_result  # Get the statsmodels backend result
    
    # Pearson chi-square / df
    pearson_chi2 = (backend_result.resid_pearson ** 2).sum()
    df_resid = len(backend_result.resid_pearson) - backend_result.model.exog.shape[1]
    dispersion = pearson_chi2 / df_resid
    
    # Deviance / df
    deviance_dispersion = result.deviance / df_resid
    
    # Interpretation
    if dispersion > 1.5:
        interpretation = "Significant overdispersion detected"
        recommendation = "Consider using Negative Binomial family"
    elif dispersion < 0.7:
        interpretation = "Underdispersion detected (unusual)"
        recommendation = "Check for model overfitting"
    else:
        interpretation = "Dispersion within acceptable range"
        recommendation = "Poisson family appropriate"
    
    return {
        'pearson_dispersion': float(dispersion),
        'deviance_dispersion': float(deviance_dispersion),
        'interpretation': interpretation,
        'recommendation': recommendation,
    }


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_groups: int = 10
) -> Dict:
    """
    Hosmer-Lemeshow goodness-of-fit test for binary outcomes.
    
    Tests whether observed and expected frequencies match across risk groups.
    
    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0/1)
    y_pred : array-like
        Predicted probabilities
    n_groups : int
        Number of risk groups
    
    Returns
    -------
    dict
        Test statistic, p-value, and interpretation
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Create risk groups
    df['group'] = pd.qcut(df['y_pred'], q=n_groups, labels=False, duplicates='drop')
    
    # Observed vs expected
    grouped = df.groupby('group').agg({
        'y_true': ['sum', 'count'],
        'y_pred': 'sum'
    })
    
    observed = grouped[('y_true', 'sum')].values
    expected = grouped[('y_pred', 'sum')].values
    total = grouped[('y_true', 'count')].values
    
    # Chi-square statistic
    chi2_stat: float = float(np.sum((observed - expected) ** 2 / (expected * (1 - expected / total) + 1e-10)))
    
    # P-value (df = n_groups - 2)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=n_groups - 2)
    
    return {
        'chi2_statistic': float(chi2_stat),
        'p_value': float(p_value),
        'df': n_groups - 2,
        'interpretation': 'Good fit' if p_value > 0.05 else 'Poor fit',
    }