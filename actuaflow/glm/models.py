"""
GLM Model Wrappers for Actuarial Pricing - Version 3 (v1.0)

IMPROVEMENTS IN v1.0:
- Complete type hints throughout all functions and methods
- Custom exception hierarchy for better error handling
- Fixed state mutation bugs (proper immutability)
- Comprehensive documentation with examples
- Better separation of concerns
- Thread-safe state management

Author: Michael Watson
License: MPL-2.0
"""

import logging
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod import families
from statsmodels.genmod.families import (
    Gamma,
    Gaussian,
    InverseGaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
)

# Import custom exceptions
try:
    from actuaflow.exceptions import (
        EmptyDataError,
        InvalidDataTypeError,
        InvalidFamilyError,
        InvalidFormulaError,
        InvalidLinkError,
        InvalidPredictionTypeError,
        InvalidValueError,
        MissingColumnError,
        ModelFitError,
        ModelNotFittedError,
        PredictionDataMismatchError,
    )
except ImportError:
    # Fallback to standard exceptions if exceptions.py not available yet
    class ModelNotFittedError(ValueError): pass  # type: ignore
    class ModelFitError(RuntimeError): pass  # type: ignore
    class InvalidDataTypeError(TypeError): pass  # type: ignore
    class MissingColumnError(ValueError): pass  # type: ignore
    class InvalidValueError(ValueError): pass  # type: ignore
    class EmptyDataError(ValueError): pass  # type: ignore
    class InvalidFormulaError(ValueError): pass  # type: ignore
    class InvalidPredictionTypeError(ValueError): pass  # type: ignore
    class InvalidFamilyError(ValueError): pass  # type: ignore
    class InvalidLinkError(ValueError): pass  # type: ignore
    class PredictionDataMismatchError(ValueError): pass  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelResult:
    """
    Immutable standardized model results (backend-agnostic).
    
    Using frozen=True ensures immutability and prevents accidental
    state mutations that could lead to bugs.
    
    Attributes
    ----------
    coefficients : Dict[str, float]
        Model coefficients (parameter estimates)
    std_errors : Dict[str, float]
        Standard errors of coefficients
    p_values : Dict[str, float]
        P-values for hypothesis tests
    confidence_intervals : Dict[str, Tuple[float, float]]
        95% confidence intervals for coefficients
    aic : float
        Akaike Information Criterion (lower is better)
    bic : float
        Bayesian Information Criterion (lower is better)
    deviance : float
        Model deviance (goodness of fit measure)
    null_deviance : float
        Null model deviance
    dispersion : float
        Dispersion parameter (variance / mean)
    predictions : np.ndarray
        Fitted predictions on response scale
    residuals : np.ndarray
        Deviance residuals
    fitted_values : np.ndarray
        Linear predictor fitted values
    n_obs : int
        Number of observations used in fitting
    converged : bool
        Whether optimization converged successfully
    _backend_result : Any
        Internal backend result object (private, not for direct use)
    
    Examples
    --------
    >>> result = ModelResult(
    ...     coefficients={'Intercept': -2.3, 'age_group[25-35]': 0.15},
    ...     std_errors={'Intercept': 0.05, 'age_group[25-35]': 0.03},
    ...     p_values={'Intercept': 0.001, 'age_group[25-35]': 0.01},
    ...     confidence_intervals={
    ...         'Intercept': (-2.4, -2.2),
    ...         'age_group[25-35]': (0.09, 0.21)
    ...     },
    ...     aic=1250.3,
    ...     bic=1275.8,
    ...     deviance=1200.5,
    ...     null_deviance=1500.2,
    ...     dispersion=1.05,
    ...     predictions=np.array([0.1, 0.15, 0.08]),
    ...     residuals=np.array([0.05, -0.02, 0.01]),
    ...     fitted_values=np.array([0.1, 0.15, 0.08]),
    ...     n_obs=1000,
    ...     converged=True
    ... )
    >>> print(f"Model AIC: {result.aic:.2f}")
    Model AIC: 1250.30
    """
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    aic: float
    bic: float
    deviance: float
    null_deviance: float
    dispersion: float
    predictions: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    n_obs: int
    converged: bool
    _backend_result: Any = field(repr=False, default=None, compare=False)


class BaseGLM:
    """
    Base class for Generalized Linear Models with comprehensive validation.
    
    This class provides a unified interface for GLM models with robust error
    handling, validation, and state management. All ActuaFlow GLM models
    inherit from this base class.
    
    Parameters
    ----------
    family : str, default='poisson'
        Distribution family. Valid options:
        - 'poisson': Poisson distribution (count data)
        - 'negative_binomial': Negative Binomial (overdispersed counts)
        - 'gamma': Gamma distribution (positive continuous)
        - 'inverse_gaussian': Inverse Gaussian (positive continuous)
        - 'gaussian': Normal distribution
        - 'lognormal': Log-normal distribution
        - 'tweedie': Tweedie distribution (compound Poisson-Gamma)
    link : str, default='log'
        Link function. Valid options depend on family:
        - 'log': Logarithmic link
        - 'identity': Identity link
        - 'inverse': Inverse link
        - 'inverse_squared': Inverse squared link
    
    Attributes
    ----------
    fitted_ : bool
        Whether the model has been successfully fitted
    family : str
        Distribution family name
    link : str
        Link function name
    
    Raises
    ------
    InvalidFamilyError
        If family is not recognized
    InvalidLinkError
        If link is not valid for the specified family
    
    Examples
    --------
    >>> from actuaflow.glm import BaseGLM
    >>> 
    >>> # Create and fit a basic GLM
    >>> model = BaseGLM(family='poisson', link='log')
    >>> model.fit(data, 'claim_count ~ age_group + region', offset='exposure')
    >>> 
    >>> # Get predictions
    >>> predictions = model.predict(new_data)
    >>> 
    >>> # View summary
    >>> summary = model.summary()
    >>> print(summary)
    
    Notes
    -----
    The base class handles all common GLM operations. Subclasses (FrequencyGLM,
    SeverityGLM, TweedieGLM) provide specialized behavior for specific use cases.
    
    Thread Safety
    -------------
    Model instances are NOT thread-safe during fitting. Create separate instances
    for concurrent fitting operations.
    """
    
    def __init__(self, family: str = 'poisson', link: str = 'log') -> None:
        self.family: str = family.lower()
        self.link: str = link.lower()
        self._reset_state()
        
        # Validate family-link combination
        self._validate_family_link_combination(self.family, self.link)
        
        logger.info(f"Initialized {self.__class__.__name__} with family={family}, link={link}")
    
    def _reset_state(self) -> None:
        """
        Reset all fitted state to None.
        
        Called during initialization and can be called to clear fitted state.
        """
        self._result: Optional[ModelResult] = None
        self._formula: Optional[str] = None
        self._n_obs: Optional[int] = None
        self._backend_result: Any = None
    
    @property
    def fitted_(self) -> bool:
        """
        Check if model has been fitted successfully.
        
        Returns
        -------
        bool
            True if model is fitted and converged, False otherwise
        """
        return self._result is not None and self._result.converged
    
    @property
    def result_(self) -> ModelResult:
        """
        Get model results.
        
        Returns
        -------
        ModelResult
            Fitted model results
        
        Raises
        ------
        ModelNotFittedError
            If model has not been fitted yet
        """
        if self._result is None:
            raise ModelNotFittedError()
        return self._result
    
    @property
    def coefficients_(self) -> Dict[str, float]:
        """
        Get model coefficients.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping parameter names to coefficient values
        
        Raises
        ------
        ModelNotFittedError
            If model has not been fitted yet
        """
        return self.result_.coefficients
    
    @staticmethod
    def _validate_family_link_combination(family: str, link: str) -> None:
        """
        Validate that family and link are compatible.
        
        Parameters
        ----------
        family : str
            Distribution family
        link : str
            Link function
        
        Raises
        ------
        InvalidFamilyError
            If family is not recognized
        InvalidLinkError
            If link is not valid for the family
        """
        valid_combinations: Dict[str, List[str]] = {
            'poisson': ['log', 'identity'],
            'negative_binomial': ['log', 'identity'],
            'gamma': ['log', 'inverse', 'identity'],
            'inverse_gaussian': ['log', 'inverse', 'inverse_squared', 'identity'],
            'gaussian': ['identity', 'log'],
            'lognormal': ['identity'],
            'tweedie': ['log', 'identity'],
        }
        
        if family not in valid_combinations:
            raise InvalidFamilyError(
                f"Invalid family: '{family}'. Valid families: {list(valid_combinations.keys())}"
            )
        
        if link not in valid_combinations[family]:
            raise InvalidLinkError(
                f"Invalid link '{link}' for family '{family}'. "
                f"Valid links: {valid_combinations[family]}"
            )
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data structure and content.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to validate
        
        Raises
        ------
        InvalidDataTypeError
            If data is not a DataFrame
        EmptyDataError
            If data is empty
        InvalidValueError
            If data contains only missing values in some columns
        """
        if not isinstance(data, pd.DataFrame):
            raise InvalidDataTypeError(
                f"data must be pandas DataFrame, got {type(data).__name__}"
            )
        
        if len(data) == 0:
            raise EmptyDataError()
        
        # Check for columns with all missing values
        null_cols = data.columns[data.isnull().all()].tolist()
        if null_cols:
            raise InvalidValueError(
                column=str(null_cols),
                constraint="Column contains only missing values"
            )
    
    def _validate_formula(self, formula: str, data: pd.DataFrame) -> None:
        """
        Validate model formula syntax and variable existence.
        
        Parameters
        ----------
        formula : str
            R-style formula string
        data : pd.DataFrame
            Data containing variables
        
        Raises
        ------
        InvalidFormulaError
            If formula syntax is invalid
        MissingColumnError
            If response variable not found
        InvalidValueError
            If response variable has missing values
        """
        # Check basic syntax
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
        
        # Extract and check response variable
        response = parts[0].strip()
        if not response:
            raise InvalidFormulaError(
                formula=formula,
                reason="Response variable (left side of ~) is empty"
            )
        
        if response not in data.columns:
            raise MissingColumnError(
                column=response,
                available_columns=data.columns.tolist()
            )
        
        # Check for missing values in response
        if data[response].isnull().any():
            n_missing = int(data[response].isnull().sum())
            raise InvalidValueError(
                column=response,
                constraint=f"Response has {n_missing} missing values. Remove or impute them.",
                n_invalid=n_missing
            )
        
        # Check for negative values in response (invalid for count models)
        if (data[response] < 0).any():
            n_negative = int((data[response] < 0).sum())
            raise InvalidValueError(
                column=response,
                constraint=f"Response contains {n_negative} negative values. Response must be non-negative.",
                n_invalid=n_negative
            )
        
        # Extract predictor variables and check for NaN
        predictors_part = parts[1].strip()
        # Simple extraction of variable names (handles basic formulas)
        # Split by common operators but preserve word characters and underscores
        # Find all word-like tokens (handles interactions, polynomials, etc)
        predictor_tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', predictors_part)
        
        for pred in predictor_tokens:
            if pred in data.columns:
                if data[pred].isnull().any():
                    n_missing = int(data[pred].isnull().sum())
                    raise InvalidValueError(
                        column=pred,
                        constraint=f"Predictor has {n_missing} missing values. Remove or impute them.",
                        n_invalid=n_missing
                    )
    
    def _validate_offset(self, offset: Optional[str], data: pd.DataFrame) -> None:
        """
        Validate offset variable.
        
        Parameters
        ----------
        offset : str or None
            Offset column name
        data : pd.DataFrame
            Data containing offset
        
        Raises
        ------
        MissingColumnError
            If offset column not found
        InvalidValueError
            If offset contains non-positive values
        """
        if offset is not None:
            if offset not in data.columns:
                raise ValueError(
                    f"Offset variable '{offset}' not found in data"
                )
            
            if (data[offset] <= 0).any():
                n_invalid = int((data[offset] <= 0).sum())
                raise InvalidValueError(
                    column=offset,
                    constraint="Offset cannot contain non-positive values",
                    n_invalid=n_invalid
                )
    
    def _validate_weights(self, weights: Optional[str], data: pd.DataFrame) -> None:
        """
        Validate weights variable.
        
        Parameters
        ----------
        weights : str or None
            Weights column name
        data : pd.DataFrame
            Data containing weights
        
        Raises
        ------
        MissingColumnError
            If weights column not found
        InvalidValueError
            If weights contain negative or zero values
        """
        if weights is not None:
            if weights not in data.columns:
                raise ValueError(
                    f"Weight variable '{weights}' not found in data"
                )
            
            if (data[weights] < 0).any():
                n_invalid = int((data[weights] < 0).sum())
                raise InvalidValueError(
                    column=weights,
                    constraint="Weights cannot contain negative values",
                    n_invalid=n_invalid
                )
            
            if (data[weights] == 0).any():
                n_zero = int((data[weights] == 0).sum())
                raise InvalidValueError(
                    column=weights,
                    constraint=f"Weights cannot contain zero values ({n_zero} zeros found)",
                    n_invalid=n_zero
                )
    
    def _get_family_and_link(self) -> families.Family:
        """
        Convert family and link strings to statsmodels objects.
        
        Returns
        -------
        families.Family
            Statsmodels family object
        
        Raises
        ------
        InvalidLinkError
            If link function is not recognized
        InvalidFamilyError
            If family is not supported
        """
        link_map: Dict[str, families.links.Link] = {
            'log': families.links.Log(),
            'identity': families.links.Identity(),
            'inverse': families.links.InversePower(),
            'inverse_squared': families.links.Power(power=-2),
        }
        
        link_obj = link_map.get(self.link)
        if link_obj is None:
            raise InvalidLinkError(
                link=self.link,
                family=self.family,
                valid_links=list(link_map.keys())
            )
        
        family_map: Dict[str, families.Family] = {
            'poisson': Poisson(link=link_obj),
            'negative_binomial': NegativeBinomial(link=link_obj),
            'gamma': Gamma(link=link_obj),
            'inverse_gaussian': InverseGaussian(link=link_obj),
            'gaussian': Gaussian(link=link_obj),
            'tweedie': Tweedie(link=link_obj, var_power=1.5),
        }
        
        family_obj = family_map.get(self.family)
        if family_obj is None:
            raise InvalidFamilyError(
                family=self.family,
                valid_families=list(family_map.keys())
            )
        
        return family_obj
    
    def _to_array(self, x: Any) -> np.ndarray:
        """
        Convert various types to numpy array.
        
        Handles pandas Series, numpy arrays, and other array-like objects.
        """
        if hasattr(x, 'values'):  # pandas Series or DataFrame
            result = np.asarray(x.values.copy(), dtype=np.float64)
        elif isinstance(x, np.ndarray):
            result = np.asarray(x.copy(), dtype=np.float64)
        else:
            result = np.asarray(x, dtype=np.float64)
        return result  # type: ignore[no-any-return]
    
    def fit(
        self,
        data: pd.DataFrame,
        formula: str,
        offset: Optional[str] = None,
        weights: Optional[str] = None,
        **kwargs: Any
    ) -> "BaseGLM":
        """
        Fit the GLM model with comprehensive validation.
        
        This method fits the model to the provided data using the specified
        formula. All validation is performed before fitting to provide clear
        error messages.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data containing all variables in formula
        formula : str
            R-style formula (e.g., 'y ~ x1 + x2 + x1:x2')
            Supported operators: +, ``*``, :, I()
        offset : str, optional
            Column name for offset variable. For frequency models, this is
            typically the exposure. The offset is included on the log scale.
        weights : str, optional
            Column name for case weights (frequency weights)
        **kwargs : Any
            Additional arguments passed to statsmodels GLM.fit()
            Common options:
            - maxiter : int, maximum iterations (default 100)
            - method : str, optimization method (default 'IRLS')
            - tol : float, convergence tolerance
        
        Returns
        -------
        self : BaseGLM
            Fitted model instance (for method chaining)
        
        Raises
        ------
        InvalidDataTypeError
            If data is not a DataFrame
        EmptyDataError
            If data is empty
        InvalidFormulaError
            If formula syntax is invalid
        MissingColumnError
            If variables in formula not found in data
        InvalidValueError
            If data contains invalid values (negatives, missing, etc.)
        ModelFitError
            If model fitting fails
        
        Examples
        --------
        >>> model = BaseGLM(family='poisson', link='log')
        >>> 
        >>> # Basic fit
        >>> model.fit(data, 'claim_count ~ age_group + region')
        >>> 
        >>> # With offset (for frequency models)
        >>> model.fit(data, 'claim_count ~ age_group', offset='exposure')
        >>> 
        >>> # With weights
        >>> model.fit(data, 'amount ~ injury_type', weights='case_weights')
        >>> 
        >>> # With custom fit parameters
        >>> model.fit(data, 'y ~ x', maxiter=200, tol=1e-8)
        
        Notes
        -----
        - Data is NOT modified in place. A copy is made internally if needed.
        - For frequency models, always use offset='exposure'
        - Offset is automatically log-transformed
        - Missing values in response will raise an error
        - Model state is reset if fitting fails
        """
        logger.info(f"Fitting {self.__class__.__name__}: {formula}")
        
        # Validate inputs - do not modify original data
        self._validate_data(data)
        self._validate_formula(formula, data)
        self._validate_offset(offset, data)
        self._validate_weights(weights, data)
        
        # Store formula, and offset for later use in predict()
        self._formula = formula
        self._n_obs = len(data)
        self._offset_column = offset  # Store offset column name
        
        try:
            family_obj = self._get_family_and_link()
            
            # Prepare offset (log scale) - work on copy to avoid mutation
            if offset:
                log_offset = np.log(data[offset].clip(lower=1e-10))
                logger.debug(
                    f"Using offset: {offset} "
                    f"(min={data[offset].min():.2e}, max={data[offset].max():.2e})"
                )
                
                if weights:
                    model = smf.glm(
                        formula=formula,
                        data=data,
                        family=family_obj,
                        offset=log_offset,
                        freq_weights=data[weights]
                    )
                else:
                    model = smf.glm(
                        formula=formula,
                        data=data,
                        family=family_obj,
                        offset=log_offset
                    )
            else:
                if weights:
                    model = smf.glm(
                        formula=formula,
                        data=data,
                        family=family_obj,
                        freq_weights=data[weights]
                    )
                else:
                    model = smf.glm(
                        formula=formula,
                        data=data,
                        family=family_obj
                    )
            
            # Fit model
            backend_result = model.fit(**kwargs)
            self._backend_result = backend_result
            
            # Check convergence
            if not backend_result.converged:
                warnings.warn(
                    f"Model did not converge. Results may be unreliable. "
                    "Consider: (1) scaling predictors, (2) checking for perfect "
                    "separation, (3) simplifying model.",
                    UserWarning
                )
            
            # Compute dispersion
            if self.family == 'poisson':
                dispersion = (
                    backend_result.pearson_chi2 / backend_result.df_resid 
                    if backend_result.df_resid > 0 else 1.0
                )
            else:
                dispersion = backend_result.scale
            
            # Create immutable standardized result
            conf_int = backend_result.conf_int()
            
            self._result = ModelResult(
                coefficients=dict(zip(
                    backend_result.params.index,
                    backend_result.params.values
                )),
                std_errors=dict(zip(
                    backend_result.bse.index,
                    backend_result.bse.values
                )),
                p_values=dict(zip(
                    backend_result.pvalues.index,
                    backend_result.pvalues.values
                )),
                confidence_intervals={
                    param: (float(conf_int.loc[param, 0]), float(conf_int.loc[param, 1]))
                    for param in conf_int.index
                },
                aic=float(backend_result.aic),
                bic=float(backend_result.bic),
                deviance=float(backend_result.deviance),
                null_deviance=float(backend_result.null_deviance),
                dispersion=float(dispersion),
                predictions=self._to_array(backend_result.predict()),
                residuals=self._to_array(backend_result.resid_deviance),
                fitted_values=self._to_array(backend_result.fittedvalues),
                n_obs=int(backend_result.nobs),
                converged=bool(backend_result.converged),
                _backend_result=backend_result
            )
            
            logger.info(
                f"Model fitted successfully: AIC={self._result.aic:.2f}, "
                f"dispersion={self._result.dispersion:.3f}, converged={self._result.converged}"
            )
            
            # Warn about overdispersion
            if self.family == 'poisson' and dispersion > 1.5:
                logger.warning(
                    f"Overdispersion detected (dispersion={dispersion:.3f}). "
                    "Consider using Negative Binomial family."
                )
            
            return self
            
        except Exception as e:
            logger.error(f"Model fitting failed: {str(e)}")
            self._reset_state()  # Clear partial state on failure
            raise ModelFitError(
                f"Failed to fit GLM model: {str(e)}"
            ) from e
    
    def predict(
        self,
        newdata: Optional[pd.DataFrame] = None,
        type: str = 'response'
    ) -> np.ndarray:
        """
        Generate predictions from fitted model.
        
        Parameters
        ----------
        newdata : pd.DataFrame, optional
            New data for prediction. If None, returns fitted values from
            training data. Must contain all variables used in the model formula.
            If the model was fitted with an offset, newdata should also have the
            offset column.
        type : str, default='response'
            Type of prediction:
            - 'response': predictions on response scale (default)
            - 'link': predictions on link scale (linear predictor)
        
        Returns
        -------
        predictions : np.ndarray
            Array of predictions
        
        Raises
        ------
        ModelNotFittedError
            If model has not been fitted
        InvalidPredictionTypeError
            If type is not 'response' or 'link'
        
        Examples
        --------
        >>> model = BaseGLM(family='poisson', link='log')
        >>> model.fit(data, 'y ~ x')
        >>> preds = model.predict()
        >>> link_preds = model.predict(type='link')
        """
        if not self.fitted_:
            raise ModelNotFittedError()
        
        # Validate type parameter
        valid_types = ['response', 'link']
        if type not in valid_types:
            raise InvalidPredictionTypeError(
                requested_type=type,
                valid_types=valid_types
            )
        
        result = self._backend_result
        
        # Get response scale predictions
        if newdata is None:
            # For training data, just call predict() without arguments
            # This returns response scale predictions
            response_preds = result.predict()
        else:
            # For new data, pass the DataFrame
            # Default behavior is response scale
            response_preds = result.predict(newdata)
        
        # Convert to link scale if needed using the link function
        if type == 'link':
            # Get the link object to convert response to link scale
            # link() function converts response_scale to link_scale
            # For example, for log link: link(response) = log(response)
            link = result.model.family.link
            predictions = link(response_preds)
        else:
            predictions = response_preds
        
        # Ensure we return a numpy array
        return self._to_array(predictions)
    
    def residuals(self, type: str = 'deviance') -> np.ndarray:
        """
        Get model residuals.
        
        Parameters
        ----------
        type : str, default='deviance'
            Type of residuals: 'deviance', 'pearson', 'response', 'working'
        
        Returns
        -------
        np.ndarray
            Residuals
        
        Raises
        ------
        ModelNotFittedError
            If model not fitted
        InvalidPredictionTypeError
            If type is invalid
        """
        if not self.fitted_:
            raise ModelNotFittedError()
        
        valid_types = ['deviance', 'pearson', 'response', 'working']
        if type not in valid_types:
            raise InvalidPredictionTypeError(
                requested_type=type,
                valid_types=valid_types
            )
        
        backend_result = self._backend_result
        
        # Get residuals from backend result
        if type == 'deviance':
            residuals = backend_result.resid_deviance
        elif type == 'pearson':
            residuals = backend_result.resid_pearson
        elif type == 'response':
            residuals = backend_result.resid_response
        elif type == 'working':
            residuals = backend_result.resid_working
        
        # Convert to numpy array
        return self._to_array(residuals)
    
    def summary(self) -> pd.DataFrame:
        """
        Get model summary as DataFrame.
        
        Returns comprehensive summary table with coefficient estimates,
        standard errors, z-statistics, p-values, confidence intervals,
        and significance stars.
        
        Returns
        -------
        summary_df : pd.DataFrame
            Summary table with columns:
            - Coefficient: Parameter estimate
            - Std Error: Standard error of estimate
            - z-value: z-statistic (Wald test)
            - P-value: Two-tailed p-value
            - CI Lower: Lower bound of 95% CI
            - CI Upper: Upper bound of 95% CI
            - Signif: Significance stars (``***`` p<0.001, ``**`` p<0.01, ``*`` p<0.05)
            - Relativity: exp(Coefficient) for log link models
        
        Raises
        ------
        ModelNotFittedError
            If model has not been fitted
        
        Examples
        --------
        >>> summary = model.summary()
        >>> print(summary)
        >>>                     Coefficient  Std Error  z-value  P-value  ...
        >>> Intercept                -2.303      0.045  -51.178    0.000  ***
        >>> age_group[18-25]          0.405      0.067    6.045    0.000  ***
        >>> age_group[26-35]          0.000      0.000      NaN      NaN
        >>> vehicle_type[suv]         0.182      0.052    3.500    0.000  ***
        >>> 
        >>> # Get only significant predictors
        >>> significant = summary[summary['Signif'] != '']
        """
        if not self.fitted_:
            raise ModelNotFittedError()
        
        result = self.result_
        
        # Build summary DataFrame
        coef_df = pd.DataFrame({
            'Coefficient': list(result.coefficients.values()),
            'Std Error': list(result.std_errors.values()),
            'z-value': [
                result.coefficients[k] / result.std_errors[k] 
                if result.std_errors[k] != 0 else np.nan
                for k in result.coefficients.keys()
            ],
            'P-value': list(result.p_values.values()),
        }, index=list(result.coefficients.keys()))
        
        coef_df['CI Lower'] = [
            result.confidence_intervals[k][0] 
            for k in result.coefficients.keys()
        ]
        coef_df['CI Upper'] = [
            result.confidence_intervals[k][1] 
            for k in result.coefficients.keys()
        ]
        
        # Add significance stars
        coef_df['Signif'] = coef_df['P-value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        )
        
        # Add relativities for log link
        if self.link == 'log':
            coef_df['Relativity'] = np.exp(coef_df['Coefficient'])
        
        return coef_df
    
    def diagnostics(self) -> Dict[str, Any]:
        """
        Compute comprehensive model diagnostics.
        
        Returns dictionary with fit statistics and diagnostic information
        useful for model evaluation and comparison.
        
        Returns
        -------
        diagnostics : dict
            Dictionary containing:
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion
            - deviance: Model deviance
            - null_deviance: Null model deviance
            - pearson_chi2: Pearson chi-squared statistic
            - dispersion: Dispersion parameter
            - df_model: Model degrees of freedom
            - df_resid: Residual degrees of freedom
            - nobs: Number of observations
            - converged: Convergence status
            - formula: Model formula
            - dispersion_warning: Interpretation of dispersion
        
        Raises
        ------
        ModelNotFittedError
            If model has not been fitted
        """
        if not self.fitted_:
            raise ModelNotFittedError()
        
        result = self.result_
        
        diag: Dict[str, Any] = {
            'aic': result.aic,
            'bic': result.bic,
            'deviance': result.deviance,
            'null_deviance': result.null_deviance,
            'pearson_chi2': float(self._backend_result.pearson_chi2),
            'dispersion': result.dispersion,
            'df_model': int(self._backend_result.df_model),
            'df_resid': int(self._backend_result.df_resid),
            'nobs': result.n_obs,
            'converged': result.converged,
            'formula': self._formula,
        }
        
        # Dispersion interpretation
        if self.family == 'poisson':
            if result.dispersion > 1.5:
                diag['dispersion_warning'] = 'Overdispersed - consider Negative Binomial'
            elif result.dispersion < 0.7:
                diag['dispersion_warning'] = 'Underdispersed - possible overfitting'
            else:
                diag['dispersion_warning'] = 'Acceptable'
        
        return diag


class FrequencyGLM(BaseGLM):
    """
    GLM for modeling claim frequency with validation.
    
    Specialized for Poisson and Negative Binomial distributions with offset support.
    
    Parameters
    ----------
    family : str, default='poisson'
        'poisson' or 'negative_binomial'
    link : str, default='log'
        Link function (typically 'log')
    
    Examples
    --------
    >>> freq_model = FrequencyGLM(family='poisson', link='log')
    >>> freq_model.fit(data, 'claim_count ~ age_group + vehicle_type', offset='exposure')
    >>> predictions = freq_model.predict()
    >>> diag = freq_model.diagnostics()
    """
    
    def __init__(self, family: str = 'poisson', link: str = 'log') -> None:
        if family.lower() not in ['poisson', 'negative_binomial']:
            raise InvalidFamilyError(
                f"Frequency models must use 'poisson' or 'negative_binomial', got '{family}'"
            )
        super().__init__(family=family, link=link)


class SeverityGLM(BaseGLM):
    """
    GLM for modeling claim severity with automatic filtering.
    
    Specialized for Gamma, Inverse Gaussian, and Lognormal distributions.
    
    Parameters
    ----------
    family : str, default='gamma'
        'gamma', 'inverse_gaussian', or 'lognormal'
    link : str, default='log'
        Link function (typically 'log')
    
    Examples
    --------
    >>> sev_model = SeverityGLM(family='gamma', link='log')
    >>> # Automatically filters to positive claims
    >>> sev_model.fit(claims_data, 'amount ~ age_group + injury_type')
    >>> predictions = sev_model.predict()
    """
    
    def __init__(self, family: str = 'gamma', link: str = 'log') -> None:
        valid_families = ['gamma', 'inverse_gaussian', 'lognormal']
        if family.lower() not in valid_families:
            raise InvalidFamilyError(
                f"Severity models must use one of {valid_families}, got '{family}'"
            )
        super().__init__(family=family, link=link)
        self._lognormal_transform = False
    
    def fit(
        self,
        data: pd.DataFrame,
        formula: str,
        offset: Optional[str] = None,
        weights: Optional[str] = None,
        response_threshold: Optional[float] = None,
        **kwargs: Any
    ) -> "SeverityGLM":
        """
        Fit severity model with automatic zero filtering.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data
        formula : str
            Model formula
        response_threshold : float, optional
            Maximum response value (for capping large claims)
        **kwargs
            Additional fit parameters
        
        Returns
        -------
        self
        """
        # Extract response variable
        response_var = formula.split('~')[0].strip()
        
        if response_var not in data.columns:
            raise MissingColumnError(
                column=response_var,
                available_columns=data.columns.tolist()
            )
        
        # Filter data - work on copy
        data_filtered = data.copy()
        
        # Remove zeros
        n_zeros = (data_filtered[response_var] <= 0).sum()
        if n_zeros > 0:
            logger.info(f"Removing {n_zeros} non-positive {response_var} observations")
            data_filtered = data_filtered[data_filtered[response_var] > 0]
        
        if len(data_filtered) == 0:
            raise EmptyDataError("No positive observations remaining after filtering zeros")
        
        # Apply threshold
        if response_threshold is not None:
            n_capped = (data_filtered[response_var] > response_threshold).sum()
            if n_capped > 0:
                logger.info(f"Capping {n_capped} observations above {response_threshold}")
                data_filtered = data_filtered[data_filtered[response_var] <= response_threshold]
        
        # Handle lognormal
        if self.family == 'lognormal':
            data_filtered = data_filtered.copy()
            data_filtered[f'_log_{response_var}_'] = np.log(
                data_filtered[response_var].clip(lower=1e-10)
            )
            formula_transformed = formula.replace(response_var, f'_log_{response_var}_')
            original_family = self.family
            self.family = 'gaussian'
            self._lognormal_transform = True
            
            try:
                super().fit(data_filtered, formula_transformed, offset=offset, weights=weights, **kwargs)
                self.family = original_family
                return self
            except Exception:
                self.family = original_family
                raise
        else:
            self._lognormal_transform = False
            super().fit(data_filtered, formula, offset=offset, weights=weights, **kwargs)
            return self
    
    def predict(
        self, 
        newdata: Optional[pd.DataFrame] = None, 
        type: str = 'response'
    ) -> np.ndarray:
        """Predict severity with back-transformation for lognormal."""
        predictions = super().predict(newdata, type)
        
        # Back-transform lognormal
        if self._lognormal_transform and type == 'response':
            result = np.asarray(np.exp(predictions), dtype=np.float64)
        else:
            result = np.asarray(predictions, dtype=np.float64)
        return result  # type: ignore[no-any-return]


class TweedieGLM(BaseGLM):
    """
    Tweedie GLM for modeling pure premium / aggregate loss.
    
    The Tweedie distribution unifies frequency and severity in a single model.
    
    Parameters
    ----------
    var_power : float, default=1.5
        Variance power parameter (p)
        - p = 0: Normal
        - p = 1: Poisson
        - p = 1.5: Compound Poisson-Gamma (typical for pure premium)
        - p = 2: Gamma
        - p = 3: Inverse Gaussian
    link : str, default='log'
        Link function
    
    Examples
    --------
    >>> tweedie = TweedieGLM(var_power=1.5, link='log')
    >>> tweedie.fit(data, 'pure_premium ~ age_group + vehicle_type', offset='exposure')
    >>> predictions = tweedie.predict()
    """
    
    def __init__(self, var_power: float = 1.5, link: str = 'log') -> None:
        if not 0 <= var_power <= 3:
            raise InvalidValueError(
                column='var_power',
                constraint=f"var_power must be between 0 and 3, got {var_power}"
            )
        
        self.var_power = var_power
        super().__init__(family='tweedie', link=link)
    
    def _get_family_and_link(self) -> families.Family:
        """Override to set var_power for Tweedie."""
        link_map: Dict[str, families.links.Link] = {
            'log': families.links.Log(),
            'identity': families.links.Identity(),
        }
        link_obj = link_map.get(self.link, families.links.Log())
        return Tweedie(link=link_obj, var_power=self.var_power)