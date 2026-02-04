"""
Time-Series Cross-Validation for Actuarial Models

Provides time-series aware cross-validation that respects temporal ordering
of data, which is critical for insurance pricing models where future data
should not leak into training sets.

Author: Michael Watson
License: MPL-2.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """
    Results from cross-validation.
    
    Attributes
    ----------
    fold_scores : List[float]
        Score for each fold
    mean_score : float
        Mean score across folds
    std_score : float
        Standard deviation of scores
    fold_sizes : List[Tuple[int, int]]
        Training and test sizes for each fold
    fold_periods : List[Tuple[str, str]]
        Date ranges for each fold
    metric_name : str
        Name of the scoring metric
    """
    fold_scores: List[float]
    mean_score: float
    std_score: float
    fold_sizes: List[Tuple[int, int]]
    fold_periods: List[Tuple[str, str]]
    metric_name: str


class TimeSeriesSplit:
    """
    Time-series cross-validation with expanding or rolling window.
    
    This splitter ensures that training data always precedes test data
    in time, preventing data leakage. This is essential for insurance
    pricing where models must be validated on future periods.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits/folds
    test_size : int, optional
        Number of observations in each test set
    gap : int, default=0
        Number of observations to skip between train and test
        (useful for accounting for development lag)
    expanding_window : bool, default=True
        If True, training window expands with each fold (default).
        If False, uses rolling window of fixed size.
    min_train_size : int, optional
        Minimum size of training set
    
    Examples
    --------
    >>> from actuaflow.utils.cross_validation import TimeSeriesSplit
    >>> 
    >>> # Expanding window (training set grows)
    >>> cv = TimeSeriesSplit(n_splits=5, test_size=1000, gap=0)
    >>> for train_idx, test_idx in cv.split(data, date_col='accident_date'):
    ...     train_data = data.iloc[train_idx]
    ...     test_data = data.iloc[test_idx]
    ...     # Fit and evaluate model
    
    Notes
    -----
    For insurance data with development lag, use gap parameter to avoid
    data leakage. For example, if claims take 12 months to develop fully,
    set gap to approximately 12 months worth of data.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding_window: bool = True,
        min_train_size: Optional[int] = None
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        if gap < 0:
            raise ValueError("gap must be non-negative")
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
        self.min_train_size = min_train_size
    
    def split(
        self,
        data: pd.DataFrame,
        date_col: Optional[str] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to split (must be sorted by time)
        date_col : str, optional
            Date column for sorting (if None, assumes already sorted)
        
        Yields
        ------
        train_indices : np.ndarray
            Indices for training set
        test_indices : np.ndarray
            Indices for test set
        """
        n_samples = len(data)
        
        # Sort by date if specified
        if date_col:
            if date_col not in data.columns:
                raise ValueError(f"Date column '{date_col}' not found in data")
            data = data.sort_values(date_col).reset_index(drop=True)
        
        # Calculate split sizes
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        if self.min_train_size is None:
            min_train_size = test_size
        else:
            min_train_size = self.min_train_size
        
        # Validate sizes
        total_required = min_train_size + (self.n_splits * test_size) + (self.n_splits * self.gap)
        if total_required > n_samples:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits. "
                f"Required: {total_required}, Available: {n_samples}"
            )
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Calculate split point
            if self.expanding_window:
                # Expanding window: training set grows
                train_end = min_train_size + (i * test_size)
            else:
                # Rolling window: training set has fixed size
                train_end = min_train_size + (i * test_size)
                train_start = train_end - min_train_size
                if train_start < 0:
                    train_start = 0
            
            # Add gap
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            if self.expanding_window:
                train_indices = indices[:train_end]
            else:
                train_indices = indices[train_start:train_end]
            
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


def cross_val_score(
    model: Any,
    data: pd.DataFrame,
    formula: str,
    date_col: str,
    cv: Optional[TimeSeriesSplit] = None,
    scoring: str = 'aic',
    offset: Optional[str] = None,
    weights: Optional[str] = None,
    **fit_params: Any
) -> CVResult:
    """
    Evaluate model using time-series cross-validation.
    
    Parameters
    ----------
    model : BaseGLM or similar
        Model instance to evaluate
    data : pd.DataFrame
        Complete dataset
    formula : str
        Model formula
    date_col : str
        Date column for temporal ordering
    cv : TimeSeriesSplit, optional
        Cross-validation splitter (if None, uses default)
    scoring : str, default='aic'
        Scoring metric: 'aic', 'bic', 'deviance', 'mae', 'rmse'
    offset : str, optional
        Offset column name
    weights : str, optional
        Weights column name
    **fit_params
        Additional parameters passed to model.fit()
    
    Returns
    -------
    CVResult
        Cross-validation results
    
    Examples
    --------
    >>> from actuaflow.glm import FrequencyGLM
    >>> from actuaflow.utils.cross_validation import cross_val_score
    >>> 
    >>> model = FrequencyGLM(family='poisson', link='log')
    >>> results = cross_val_score(
    ...     model=model,
    ...     data=policies,
    ...     formula='claim_count ~ age_group + region',
    ...     date_col='policy_start_date',
    ...     scoring='aic',
    ...     offset='exposure'
    ... )
    >>> print(f"Mean AIC: {results.mean_score:.2f} ± {results.std_score:.2f}")
    """
    if cv is None:
        cv = TimeSeriesSplit(n_splits=5, expanding_window=True)
    elif not isinstance(cv, TimeSeriesSplit):
        logger.warning(
            "Non-timeseries CV provided. Converting to TimeSeriesSplit for temporal integrity."
        )
        cv = TimeSeriesSplit(n_splits=5, expanding_window=True)
    
    # Sort data by date
    data_sorted = data.sort_values(date_col).reset_index(drop=True)
    
    fold_scores = []
    fold_sizes = []
    fold_periods = []
    
    logger.info(f"Starting {cv.n_splits}-fold time-series cross-validation")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(data_sorted, date_col), 1):
        train_data = data_sorted.iloc[train_idx].copy()
        test_data = data_sorted.iloc[test_idx].copy()
        
        # Get date ranges
        train_start = train_data[date_col].min()
        train_end = train_data[date_col].max()
        test_start = test_data[date_col].min()
        test_end = test_data[date_col].max()
        
        logger.info(
            f"Fold {fold_idx}: Train [{train_start} to {train_end}] "
            f"(n={len(train_data)}), Test [{test_start} to {test_end}] "
            f"(n={len(test_data)})"
        )
        
        # Fit model
        try:
            model.fit(
                train_data,
                formula,
                offset=offset,
                weights=weights,
                **fit_params
            )
        except Exception as e:
            logger.warning(f"Fold {fold_idx} fitting failed: {str(e)}")
            continue
        
        # Score model
        score = _compute_score(model, test_data, scoring, offset)
        
        fold_scores.append(score)
        fold_sizes.append((len(train_data), len(test_data)))
        fold_periods.append((
            f"{train_start} to {train_end}",
            f"{test_start} to {test_end}"
        ))
        
        logger.info(f"Fold {fold_idx} {scoring}: {score:.4f}")
    
    if not fold_scores:
        raise ValueError("All folds failed to fit")
    
    result = CVResult(
        fold_scores=fold_scores,
        mean_score=float(np.mean(fold_scores)),
        std_score=float(np.std(fold_scores)),
        fold_sizes=fold_sizes,
        fold_periods=fold_periods,
        metric_name=scoring
    )
    
    logger.info(
        f"Cross-validation complete: {scoring} = "
        f"{result.mean_score:.4f} ± {result.std_score:.4f}"
    )
    
    return result


def _compute_score(
    model: Any,
    test_data: pd.DataFrame,
    scoring: str,
    offset: Optional[str] = None
) -> float:
    """
    Compute score for fitted model on test data.
    
    Parameters
    ----------
    model : fitted model
        Model to score
    test_data : pd.DataFrame
        Test data
    scoring : str
        Scoring metric
    offset : str, optional
        Offset column
    
    Returns
    -------
    float
        Score value
    """
    if scoring in ['aic', 'bic']:
        # Refit on test data to get AIC/BIC
        # (This is technically not correct but commonly done)
        response_var = model._formula.split('~')[0].strip()
        
        test_model = model.__class__(family=model.family, link=model.link)
        test_model.fit(test_data, model._formula, offset=offset)
        
        diag = test_model.diagnostics()
        return float(diag[scoring])
    
    elif scoring == 'deviance':
        # Get predictions
        predictions = model.predict(test_data)
        
        # Extract response
        response_var = model._formula.split('~')[0].strip()
        y_true = test_data[response_var].values
        
        # Compute deviance (depends on family)
        if model.family == 'poisson':
            # Poisson deviance
            mask = y_true > 0
            dev = np.zeros_like(y_true, dtype=float)
            dev[mask] = 2 * (y_true[mask] * np.log(y_true[mask] / predictions[mask]) - 
                            (y_true[mask] - predictions[mask]))
            dev[~mask] = 2 * predictions[~mask]
            return float(np.sum(dev))
        
        else:
            # Generic: use model's deviance
            return float(model.result_.deviance)
    
    elif scoring == 'mae':
        predictions = model.predict(test_data)
        response_var = model._formula.split('~')[0].strip()
        y_true = test_data[response_var].values
        return float(np.mean(np.abs(y_true - predictions)))
    
    elif scoring == 'rmse':
        predictions = model.predict(test_data)
        response_var = model._formula.split('~')[0].strip()
        y_true = test_data[response_var].values
        return float(np.sqrt(np.mean((y_true - predictions) ** 2)))
    
    elif scoring == 'mape':
        predictions = model.predict(test_data)
        response_var = model._formula.split('~')[0].strip()
        y_true = test_data[response_var].values
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100)
    
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")


def temporal_train_test_split(
    data: pd.DataFrame,
    date_col: str,
    test_size: Union[int, float] = 0.2,
    gap: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into temporal train/test sets.
    
    Simple utility for single train/test split respecting temporal ordering.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to split
    date_col : str
        Date column for temporal ordering
    test_size : int or float
        If int: number of observations in test set
        If float: proportion of data in test set
    gap : int, default=0
        Number of observations to skip between train and test
    
    Returns
    -------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    
    Examples
    --------
    >>> train, test = temporal_train_test_split(
    ...     data=policies,
    ...     date_col='policy_start_date',
    ...     test_size=0.2,
    ...     gap=0
    ... )
    """
    if date_col not in data.columns:
        raise ValueError(f"Date column '{date_col}' not found in data")
    
    # Sort by date
    data_sorted = data.sort_values(date_col).reset_index(drop=True)
    n = len(data_sorted)
    
    # Calculate test size
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size as float must be between 0 and 1")
        n_test = int(n * test_size)
    else:
        n_test = test_size
    
    if n_test <= 0 or n_test >= n:
        raise ValueError(f"Invalid test_size: results in {n_test} test observations")
    
    # Calculate split point
    split_point = n - n_test - gap
    
    if split_point <= 0:
        raise ValueError("Not enough data for train/test split with specified gap")
    
    train_data = data_sorted.iloc[:split_point].copy()
    test_data = data_sorted.iloc[split_point + gap:].copy()
    
    logger.info(
        f"Train set: {len(train_data)} obs "
        f"({train_data[date_col].min()} to {train_data[date_col].max()})"
    )
    logger.info(
        f"Test set: {len(test_data)} obs "
        f"({test_data[date_col].min()} to {test_data[date_col].max()})"
    )
    
    return train_data, test_data


def walk_forward_validation(
    model: Any,
    data: pd.DataFrame,
    formula: str,
    date_col: str,
    initial_train_size: int,
    step_size: int = 1,
    offset: Optional[str] = None,
    scoring: str = 'mae',
    **fit_params
) -> pd.DataFrame:
    """
    Walk-forward validation for time-series models.
    
    Trains model on initial period, predicts next period, then retrains
    with expanded data. Useful for evaluating model stability over time.
    
    Parameters
    ----------
    model : BaseGLM or similar
        Model instance
    data : pd.DataFrame
        Complete dataset (must be sorted by time)
    formula : str
        Model formula
    date_col : str
        Date column
    initial_train_size : int
        Size of initial training set
    step_size : int, default=1
        Number of periods to step forward each iteration
    offset : str, optional
        Offset column
    scoring : str, default='mae'
        Scoring metric
    **fit_params
        Additional fit parameters
    
    Returns
    -------
    pd.DataFrame
        Results for each step with columns:
        - step: Step number
        - train_start, train_end: Training period
        - test_start, test_end: Test period
        - train_size, test_size: Data sizes
        - score: Performance score
    
    Examples
    --------
    >>> results = walk_forward_validation(
    ...     model=FrequencyGLM(family='poisson'),
    ...     data=policies,
    ...     formula='claim_count ~ age_group',
    ...     date_col='policy_date',
    ...     initial_train_size=10000,
    ...     step_size=1000
    ... )
    """
    # Sort data
    data_sorted = data.sort_values(date_col).reset_index(drop=True)
    n = len(data_sorted)
    
    results = []
    step = 0
    
    train_end = initial_train_size
    
    while train_end + step_size <= n:
        test_start = train_end
        test_end = train_end + step_size
        
        train_data = data_sorted.iloc[:train_end].copy()
        test_data = data_sorted.iloc[test_start:test_end].copy()
        
        # Fit model
        try:
            model.fit(train_data, formula, offset=offset, **fit_params)
            score = _compute_score(model, test_data, scoring, offset)
        except Exception as e:
            logger.warning(f"Step {step} failed: {str(e)}")
            score = np.nan
        
        results.append({
            'step': step,
            'train_start': train_data[date_col].min(),
            'train_end': train_data[date_col].max(),
            'test_start': test_data[date_col].min(),
            'test_end': test_data[date_col].max(),
            'train_size': len(train_data),
            'test_size': len(test_data),
            'score': score
        })
        
        # Move forward
        train_end = test_end
        step += 1
    
    return pd.DataFrame(results)