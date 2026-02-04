"""
ActuaFlow Custom Exception Hierarchy

Provides specific exceptions for different error scenarios to improve
error handling and debugging.

Author: Michael Watson
License: MPL-2.0
"""

from typing import Any, List, Optional


class ActuaFlowError(Exception):
    """
    Base exception for all ActuaFlow errors.
    
    All custom exceptions in ActuaFlow inherit from this base class,
    making it easy to catch all package-specific errors.
    """
    pass


# =============================================================================
# Data Validation Errors
# =============================================================================

class DataValidationError(ActuaFlowError, ValueError):
    """Base class for data validation errors."""
    pass


class MissingColumnError(DataValidationError):
    """
    Raised when a required column is missing from a DataFrame.
    
    Parameters
    ----------
    column : str
        Name of the missing column
    available_columns : list, optional
        List of available columns
    """
    
    def __init__(
        self, 
        column: str, 
        available_columns: Optional[List[str]] = None
    ):
        self.column = column
        self.available_columns = available_columns
        
        msg = f"Column '{column}' not found in data"
        if available_columns:
            msg += f". Available columns: {available_columns}"
        
        super().__init__(msg)


class InvalidDataTypeError(DataValidationError, TypeError):
    """
    Raised when data has an incorrect type.
    
    Can be called with either:
    - A single message string
    - Two type objects (expected, actual)
    """
    
    def __init__(self, message_or_expected=None, actual_type=None):
        # If second parameter provided, it's the two-type calling pattern
        if actual_type is not None:
            self.expected_type = message_or_expected
            self.actual_type = actual_type
            msg = (
                f"Invalid data type: expected {message_or_expected.__name__}, "
                f"got {actual_type.__name__}"
            )
        # If first parameter is a string, use it as message
        elif isinstance(message_or_expected, str):
            msg = message_or_expected
        else:
            msg = str(message_or_expected)
        
        Exception.__init__(self, msg)


class InvalidValueError(DataValidationError):
    """
    Raised when data contains invalid values.
    
    Parameters
    ----------
    column : str
        Column containing invalid values
    constraint : str
        Description of the constraint that was violated
    n_invalid : int, optional
        Number of invalid values
    """
    
    def __init__(
        self, 
        column: str, 
        constraint: str, 
        n_invalid: Optional[int] = None
    ):
        self.column = column
        self.constraint = constraint
        self.n_invalid = n_invalid
        
        msg = f"Column '{column}' violates constraint: {constraint}"
        if n_invalid is not None:
            msg += f" ({n_invalid} invalid values)"
        
        super().__init__(msg)


class MissingValueError(DataValidationError):
    """
    Raised when data contains missing values where they're not allowed.
    
    Parameters
    ----------
    columns : list of str
        Columns containing missing values
    n_missing : dict, optional
        Dictionary mapping column names to count of missing values
    """
    
    def __init__(
        self, 
        columns: List[str], 
        n_missing: Optional[dict] = None
    ):
        self.columns = columns
        self.n_missing = n_missing
        
        msg = f"missing values found in columns: {columns}"
        if n_missing:
            msg += f". Counts: {n_missing}"
        
        super().__init__(msg)


class EmptyDataError(DataValidationError):
    """Raised when attempting to fit a model on empty data."""
    
    def __init__(self, message: str = "Cannot fit model on empty dataset"):
        super().__init__(message)


# =============================================================================
# Model Errors
# =============================================================================

class ModelError(ActuaFlowError, ValueError):
    """Base class for model-related errors."""
    pass


class ModelNotFittedError(ModelError):
    """
    Raised when attempting to use an unfitted model.
    
    This error is raised when methods like predict() or summary() are
    called before the model has been fitted with the fit() method.
    """
    
    def __init__(
        self, 
        message: str = "Model has not been fitted. must be fitted before calling this method"
    ):
        super().__init__(message)


class ModelFitError(ModelError):
    """
    Raised when model fitting fails.
    
    Parameters
    ----------
    reason : str
        Reason for fitting failure
    original_error : Exception, optional
        Original exception that caused the failure
    """
    
    def __init__(
        self, 
        reason: str, 
        original_error: Optional[Exception] = None
    ):
        self.reason = reason
        self.original_error = original_error
        
        msg = f"Model fitting failed: {reason}"
        if original_error:
            msg += f"\nOriginal error: {str(original_error)}"
        
        super().__init__(msg)


class ConvergenceError(ModelError):
    """
    Raised when model fails to converge.
    
    Parameters
    ----------
    n_iterations : int, optional
        Number of iterations attempted
    """
    
    def __init__(
        self, 
        message: str = "Model failed to converge", 
        n_iterations: Optional[int] = None
    ):
        if n_iterations:
            message += f" after {n_iterations} iterations"
        super().__init__(message)


class InvalidModelSpecificationError(ModelError):
    """
    Raised when model specification is invalid.
    
    Parameters
    ----------
    specification : str
        Description of the invalid specification
    """
    
    def __init__(self, specification: str):
        self.specification = specification
        msg = f"Invalid model specification: {specification}"
        super().__init__(msg)


# =============================================================================
# Formula Errors
# =============================================================================

class FormulaError(ActuaFlowError):
    """Base class for formula-related errors."""
    pass


class InvalidFormulaError(FormulaError, ValueError):
    """
    Raised when a model formula is invalid.
    
    Parameters
    ----------
    formula : str
        The invalid formula
    reason : str
        Reason why the formula is invalid
    """
    
    def __init__(self, formula: str, reason: str):
        self.formula = formula
        self.reason = reason
        msg = f"Invalid formula '{formula}': {reason}"
        Exception.__init__(self, msg)


class MissingFormulaVariableError(FormulaError):
    """
    Raised when a variable in the formula is not found in the data.
    
    Parameters
    ----------
    variable : str
        Missing variable name
    formula : str
        The formula containing the missing variable
    """
    
    def __init__(self, variable: str, formula: str):
        self.variable = variable
        self.formula = formula
        msg = f"Variable '{variable}' in formula '{formula}' not found in data"
        super().__init__(msg)


# =============================================================================
# Family/Link Errors
# =============================================================================

class DistributionError(ActuaFlowError):
    """Base class for distribution family errors."""
    pass


class InvalidFamilyError(DistributionError, ValueError):
    """
    Raised when an invalid distribution family is specified.
    
    Parameters
    ----------
    family : str
        The invalid family name
    valid_families : list, optional
        List of valid family names
    """
    
    def __init__(self, family=None, valid_families: Optional[List[str]] = None):
        # Handle both string message and family parameter
        if isinstance(family, str) and not family.startswith("Invalid"):
            self.family = family
            self.valid_families = valid_families
            msg = f"Unknown family: '{family}'. Invalid distribution family"
            if valid_families:
                msg += f". Valid families: {valid_families}"
        else:
            # If called with a message string
            msg = str(family) if family else "Invalid distribution family"
        
        Exception.__init__(self, msg)


class InvalidLinkError(DistributionError, ValueError):
    """
    Raised when an invalid link function is specified.
    
    Parameters
    ----------
    link : str
        The invalid link function
    family : str
        The distribution family
    valid_links : list, optional
        List of valid link functions for this family
    """
    
    def __init__(self, link: Optional[str] = None, family: Optional[str] = None, valid_links: Optional[List[str]] = None):
        # Handle both string message and structured parameters
        if family is not None:
            self.link = link
            self.family = family
            self.valid_links = valid_links
            if not link:
                msg = f"link cannot be empty for family '{family}'"
            else:
                msg = f"Invalid link '{link}' for family '{family}'"
                if valid_links:
                    msg += f". Valid links: {valid_links}"
        else:
            # If called with a message string
            msg = str(link) if link else "Invalid link function"
        
        Exception.__init__(self, msg)


class IncompatibleFamilyLinkError(DistributionError):
    """
    Raised when family and link function are incompatible.
    
    Parameters
    ----------
    family : str
        Distribution family
    link : str
        Link function
    """
    
    def __init__(self, family: str, link: str):
        self.family = family
        self.link = link
        msg = f"Link function '{link}' is incompatible with family '{family}'"
        super().__init__(msg)


# =============================================================================
# Prediction Errors
# =============================================================================

class PredictionError(ActuaFlowError, ValueError):
    """Base class for prediction errors."""
    pass


class InvalidPredictionTypeError(PredictionError):
    """
    Raised when an invalid prediction type is requested.
    
    Parameters
    ----------
    requested_type : str
        The requested prediction type
    valid_types : list
        List of valid prediction types
    """
    
    def __init__(
        self, 
        requested_type: str, 
        valid_types: List[str]
    ):
        self.requested_type = requested_type
        self.valid_types = valid_types
        msg = (
            f"type must be one of {valid_types}, got '{requested_type}'"
        )
        super().__init__(msg)


class PredictionDataMismatchError(PredictionError):
    """
    Raised when prediction data doesn't match training data structure.
    
    Parameters
    ----------
    reason : str
        Description of the mismatch
    """
    
    def __init__(self, reason: str):
        self.reason = reason
        msg = f"Prediction data mismatch: {reason}"
        super().__init__(msg)


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(ActuaFlowError, ValueError):
    """Base class for configuration errors."""
    pass


class InvalidLoadingsError(ConfigurationError):
    """
    Raised when premium loadings are invalid.
    
    Parameters
    ----------
    loading_name : str
        Name of the invalid loading
    reason : str
        Reason why the loading is invalid
    """
    
    def __init__(self, loading_name: str, reason: str):
        self.loading_name = loading_name
        self.reason = reason
        msg = f"Invalid loading '{loading_name}': {reason}"
        super().__init__(msg)


class InvalidParameterError(ConfigurationError):
    """
    Raised when a function parameter has an invalid value.
    
    Parameters
    ----------
    parameter : str
        Parameter name
    value : any
        Invalid value
    reason : str
        Reason why the value is invalid
    """
    
    def __init__(self, parameter: str, value: Any, reason: str):
        self.parameter = parameter
        self.value = value
        self.reason = reason
        msg = f"Invalid parameter '{parameter}' = {value}: {reason}"
        super().__init__(msg)


# =============================================================================
# File I/O Errors
# =============================================================================

class FileIOError(ActuaFlowError):
    """Base class for file I/O errors."""
    pass


class UnsupportedFileFormatError(FileIOError):
    """
    Raised when attempting to load an unsupported file format.
    
    Parameters
    ----------
    file_format : str
        The unsupported file format/extension
    supported_formats : list, optional
        List of supported formats
    """
    
    def __init__(
        self, 
        file_format: str, 
        supported_formats: Optional[List[str]] = None
    ):
        self.file_format = file_format
        self.supported_formats = supported_formats
        
        msg = f"Unsupported file format: '{file_format}'"
        if supported_formats:
            msg += f". Supported formats: {supported_formats}"
        
        super().__init__(msg)


# =============================================================================
# Diagnostic Errors
# =============================================================================

class DiagnosticError(ActuaFlowError):
    """Base class for diagnostic computation errors."""
    pass


class InsufficientDataError(DiagnosticError):
    """
    Raised when there's insufficient data for a diagnostic computation.
    
    Parameters
    ----------
    required : int
        Required number of observations
    actual : int
        Actual number of observations
    """
    
    def __init__(self, required: int, actual: int):
        self.required = required
        self.actual = actual
        msg = (
            f"Insufficient data: requires at least {required} observations, "
            f"but only {actual} available"
        )
        super().__init__(msg)


class OverdispersionError(DiagnosticError):
    """
    Raised to signal significant overdispersion in count models.
    
    Parameters
    ----------
    dispersion : float
        Observed dispersion parameter
    threshold : float
        Threshold for considering dispersion problematic
    """
    
    def __init__(self, dispersion: float, threshold: float = 1.5):
        self.dispersion = dispersion
        self.threshold = threshold
        msg = (
            f"Significant overdispersion detected (dispersion={dispersion:.3f}, "
            f"threshold={threshold}). Consider using Negative Binomial family."
        )
        super().__init__(msg)


# =============================================================================
# Utility Functions
# =============================================================================

def format_validation_errors(errors: List[str]) -> str:
    """
    Format multiple validation errors into a single message.
    
    Parameters
    ----------
    errors : list of str
        List of error messages
    
    Returns
    -------
    str
        Formatted error message
    """
    if not errors:
        return "No errors"
    
    if len(errors) == 1:
        return errors[0]
    
    msg = f"Found {len(errors)} validation errors:\n"
    for i, error in enumerate(errors, 1):
        msg += f"  {i}. {error}\n"
    
    return msg.rstrip()


def raise_if_errors(errors: List[str], error_class: type = DataValidationError) -> None:
    """
    Raise an exception if there are any errors in the list.
    
    Parameters
    ----------
    errors : list of str
        List of error messages
    error_class : Exception class
        Exception class to raise
    
    Raises
    ------
    error_class
        If there are any errors in the list
    """
    if errors:
        raise error_class(format_validation_errors(errors))