"""
Comprehensive exception handling tests for ActuaFlow.

Tests all custom exceptions defined in exceptions.py module.
Ensures proper error hierarchy, messages, and use cases.

Coverage: actuaflow/exceptions.py
"""

import pytest
import pandas as pd
import numpy as np

from actuaflow.exceptions import (
    ActuaFlowError,
    DataValidationError,
    MissingColumnError,
    InvalidDataTypeError,
    InvalidValueError,
    MissingValueError,
    EmptyDataError,
    ModelError,
    ModelNotFittedError,
    ModelFitError,
    ConvergenceError,
    InvalidModelSpecificationError,
    FormulaError,
    InvalidFormulaError,
    MissingFormulaVariableError,
    DistributionError,
    InvalidFamilyError,
    InvalidLinkError,
    IncompatibleFamilyLinkError,
    PredictionError,
    InvalidPredictionTypeError,
    PredictionDataMismatchError,
    ConfigurationError,
    InvalidLoadingsError,
    InvalidParameterError,
    FileIOError,
    UnsupportedFileFormatError,
    DiagnosticError,
    InsufficientDataError,
    OverdispersionError,
    format_validation_errors,
    raise_if_errors,
)


# ============================================================================
# EXCEPTION HIERARCHY TESTS
# ============================================================================

class TestExceptionHierarchy:
    """Test exception class hierarchy and inheritance."""
    
    def test_all_exceptions_inherit_from_actuaflow_error(self):
        """Test that all custom exceptions inherit from ActuaFlowError."""
        exception_classes = [
            DataValidationError, MissingColumnError, InvalidDataTypeError,
            InvalidValueError, MissingValueError, EmptyDataError,
            ModelError, ModelNotFittedError, ModelFitError, ConvergenceError,
            InvalidModelSpecificationError, FormulaError, InvalidFormulaError,
            MissingFormulaVariableError, DistributionError, InvalidFamilyError,
            InvalidLinkError, IncompatibleFamilyLinkError, PredictionError,
            InvalidPredictionTypeError, PredictionDataMismatchError,
            ConfigurationError, InvalidLoadingsError, InvalidParameterError,
            FileIOError, UnsupportedFileFormatError, DiagnosticError,
            InsufficientDataError, OverdispersionError
        ]
        
        for exc_class in exception_classes:
            assert issubclass(exc_class, ActuaFlowError), \
                f"{exc_class.__name__} does not inherit from ActuaFlowError"
    
    def test_actuaflow_error_inherits_from_exception(self):
        """Test that ActuaFlowError inherits from Exception."""
        assert issubclass(ActuaFlowError, Exception)
    
    def test_data_validation_error_hierarchy(self):
        """Test DataValidationError subclass hierarchy."""
        assert issubclass(DataValidationError, ActuaFlowError)
    
    def test_model_error_hierarchy(self):
        """Test ModelError subclass hierarchy."""
        assert issubclass(ModelError, ActuaFlowError)
    
    def test_diagnostic_error_hierarchy(self):
        """Test DiagnosticError subclass hierarchy."""
        assert issubclass(DiagnosticError, ActuaFlowError)


# ============================================================================
# DATA VALIDATION ERROR TESTS
# ============================================================================

class TestDataValidationErrors:
    """Tests for DataValidationError and related exceptions."""
    
    def test_data_validation_error_can_be_raised(self):
        """Test raising DataValidationError."""
        with pytest.raises(DataValidationError):
            raise DataValidationError("Invalid input")
    
    def test_missing_column_error_raised_on_missing_column(self):
        """Test MissingColumnError for missing columns."""
        with pytest.raises(MissingColumnError):
            raise MissingColumnError("age", available_columns=['id', 'name'])
    
    def test_invalid_data_type_error_raised_on_type_mismatch(self):
        """Test InvalidDataTypeError for data type mismatches."""
        with pytest.raises(InvalidDataTypeError):
            raise InvalidDataTypeError("Column 'amount' should be numeric, got object")
    
    def test_invalid_value_error_raised_on_constraint_violation(self):
        """Test InvalidValueError for constraint violations."""
        with pytest.raises(InvalidValueError):
            raise InvalidValueError("amount", "must be positive", n_invalid=5)
    
    def test_missing_value_error_raised_on_nan(self):
        """Test MissingValueError for NaN values."""
        with pytest.raises(MissingValueError):
            raise MissingValueError(['claim_count', 'amount'], n_missing={'claim_count': 3, 'amount': 1})
    
    def test_empty_data_error_raised_on_empty_data(self):
        """Test EmptyDataError for empty datasets."""
        with pytest.raises(EmptyDataError):
            raise EmptyDataError("Cannot fit model on empty dataset")


# ============================================================================
# MODEL ERROR TESTS
# ============================================================================

class TestModelErrors:
    """Tests for model-related exceptions."""
    
    def test_model_error_can_be_raised(self):
        """Test raising ModelError."""
        with pytest.raises(ModelError):
            raise ModelError("Model error occurred")
    
    def test_model_not_fitted_error_raised_before_prediction(self):
        """Test ModelNotFittedError for unfitted models."""
        with pytest.raises(ModelNotFittedError):
            raise ModelNotFittedError("Model must be fitted before calling predict()")
    
    def test_model_fit_error_raised_on_fitting_failure(self):
        """Test ModelFitError for model fitting failures."""
        with pytest.raises(ModelFitError):
            raise ModelFitError("Failed to converge")
    
    def test_convergence_error_raised_on_no_convergence(self):
        """Test ConvergenceError for non-convergence."""
        with pytest.raises(ConvergenceError):
            raise ConvergenceError("Optimization did not converge after 100 iterations")
    
    def test_invalid_model_specification_error_raised_on_invalid_spec(self):
        """Test InvalidModelSpecificationError for invalid specifications."""
        with pytest.raises(InvalidModelSpecificationError):
            raise InvalidModelSpecificationError("Invalid model specification")


# ============================================================================
# FORMULA ERROR TESTS
# ============================================================================

class TestFormulaErrors:
    """Tests for formula-related exceptions."""
    
    def test_formula_error_can_be_raised(self):
        """Test raising FormulaError."""
        with pytest.raises(FormulaError):
            raise FormulaError("Formula error")
    
    def test_invalid_formula_error_raised_on_invalid_formula(self):
        """Test InvalidFormulaError for invalid formulas."""
        with pytest.raises(InvalidFormulaError):
            raise InvalidFormulaError("y ~ x + z", "Formula must contain '~' separator")
    
    def test_missing_formula_variable_error_raised_on_missing_variable(self):
        """Test MissingFormulaVariableError for missing variables."""
        with pytest.raises(MissingFormulaVariableError):
            raise MissingFormulaVariableError("missing_var", "y ~ missing_var + x")


# ============================================================================
# DISTRIBUTION ERROR TESTS
# ============================================================================

class TestDistributionErrors:
    """Tests for distribution family and link errors."""
    
    def test_distribution_error_can_be_raised(self):
        """Test raising DistributionError."""
        with pytest.raises(DistributionError):
            raise DistributionError("Distribution error")
    
    def test_invalid_family_error_raised_on_invalid_family(self):
        """Test InvalidFamilyError for invalid families."""
        with pytest.raises(InvalidFamilyError):
            raise InvalidFamilyError("invalid_family", valid_families=['poisson', 'gaussian'])
    
    def test_invalid_link_error_raised_on_invalid_link(self):
        """Test InvalidLinkError for invalid link functions."""
        with pytest.raises(InvalidLinkError):
            raise InvalidLinkError("invalid_link", "poisson", valid_links=['log', 'identity'])
    
    def test_incompatible_family_link_error_raised_on_incompatibility(self):
        """Test IncompatibleFamilyLinkError for incompatible combinations."""
        with pytest.raises(IncompatibleFamilyLinkError):
            raise IncompatibleFamilyLinkError("poisson", "cloglog")


# ============================================================================
# PREDICTION ERROR TESTS
# ============================================================================

class TestPredictionErrors:
    """Tests for prediction-related exceptions."""
    
    def test_prediction_error_can_be_raised(self):
        """Test raising PredictionError."""
        with pytest.raises(PredictionError):
            raise PredictionError("Prediction computation failed")
    
    def test_invalid_prediction_type_error_raised_on_invalid_type(self):
        """Test InvalidPredictionTypeError for invalid prediction types."""
        with pytest.raises(InvalidPredictionTypeError):
            raise InvalidPredictionTypeError("invalid_type", ['response', 'terms'])
    
    def test_prediction_data_mismatch_error_raised_on_mismatch(self):
        """Test PredictionDataMismatchError for data mismatches."""
        with pytest.raises(PredictionDataMismatchError):
            raise PredictionDataMismatchError("Number of features does not match training data")


# ============================================================================
# CONFIGURATION ERROR TESTS
# ============================================================================

class TestConfigurationErrors:
    """Tests for configuration exceptions."""
    
    def test_configuration_error_can_be_raised(self):
        """Test raising ConfigurationError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Configuration error")
    
    def test_invalid_loadings_error_raised_on_invalid_loading(self):
        """Test InvalidLoadingsError for invalid loadings."""
        with pytest.raises(InvalidLoadingsError):
            raise InvalidLoadingsError("age_loading", "must be positive")
    
    def test_invalid_parameter_error_raised_on_invalid_param(self):
        """Test InvalidParameterError for invalid parameters."""
        with pytest.raises(InvalidParameterError):
            raise InvalidParameterError("n_folds", -1, "must be positive")


# ============================================================================
# FILE I/O ERROR TESTS
# ============================================================================

class TestFileIOErrors:
    """Tests for file I/O exceptions."""
    
    def test_file_io_error_can_be_raised(self):
        """Test raising FileIOError."""
        with pytest.raises(FileIOError):
            raise FileIOError("File I/O error")
    
    def test_unsupported_file_format_error_raised_on_unsupported_format(self):
        """Test UnsupportedFileFormatError for unsupported formats."""
        with pytest.raises(UnsupportedFileFormatError):
            raise UnsupportedFileFormatError(".xyz", supported_formats=['.csv', '.xlsx'])


# ============================================================================
# DIAGNOSTIC ERROR TESTS
# ============================================================================

class TestDiagnosticErrors:
    """Tests for diagnostic computation exceptions."""
    
    def test_diagnostic_error_can_be_raised(self):
        """Test raising DiagnosticError."""
        with pytest.raises(DiagnosticError):
            raise DiagnosticError("Diagnostic error")
    
    def test_insufficient_data_error_raised_on_insufficient_data(self):
        """Test InsufficientDataError for insufficient data."""
        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError(required=100, actual=50)
    
    def test_overdispersion_error_raised_on_overdispersion(self):
        """Test OverdispersionError for overdispersion detection."""
        with pytest.raises(OverdispersionError):
            raise OverdispersionError(dispersion=2.5, threshold=1.5)


# ============================================================================
# ERROR MESSAGE TESTS
# ============================================================================

class TestErrorMessages:
    """Test exception error messages."""
    
    def test_missing_column_error_message(self):
        """Test MissingColumnError message."""
        with pytest.raises(MissingColumnError) as exc_info:
            raise MissingColumnError("age")
        assert "age" in str(exc_info.value)
    
    def test_invalid_formula_error_message(self):
        """Test InvalidFormulaError message."""
        with pytest.raises(InvalidFormulaError) as exc_info:
            raise InvalidFormulaError("invalid", "missing separator")
        assert "invalid" in str(exc_info.value)
    
    def test_convergence_error_message(self):
        """Test ConvergenceError message."""
        msg = "Failed to converge"
        with pytest.raises(ConvergenceError, match=msg):
            raise ConvergenceError(msg)
    
    def test_insufficient_data_error_message(self):
        """Test InsufficientDataError message."""
        with pytest.raises(InsufficientDataError) as exc_info:
            raise InsufficientDataError(100, 50)
        assert "100" in str(exc_info.value)
        assert "50" in str(exc_info.value)


# ============================================================================
# EXCEPTION USE CASES
# ============================================================================

class TestExceptionUseCases:
    """Test realistic exception scenarios."""
    
    def test_catch_all_actuaflow_errors(self):
        """Test catching all ActuaFlow errors with parent class."""
        errors_to_test = [
            DataValidationError("test"),
            ModelError("test"),
            FormulaError("test"),
            DiagnosticError("test"),
        ]
        
        for error in errors_to_test:
            with pytest.raises(ActuaFlowError):
                raise error
    
    def test_specific_error_not_caught_by_other_types(self):
        """Test that specific errors are not caught by unrelated types."""
        with pytest.raises(InvalidFormulaError):
            try:
                raise InvalidFormulaError("y ~ x", "Invalid formula")
            except InvalidLinkError:
                pytest.fail("InvalidFormulaError should not be caught by InvalidLinkError handler")
    
    def test_data_validation_error_with_context(self):
        """Test DataValidationError with detailed context."""
        try:
            data = pd.DataFrame({'a': [1, 2, 3]})
            if 'missing_col' not in data.columns:
                raise MissingColumnError("missing_col", available_columns=list(data.columns))
        except MissingColumnError as e:
            assert "missing_col" in str(e)
    
    def test_model_fit_error_with_operation_context(self):
        """Test ModelFitError with operation context."""
        try:
            raise ModelFitError("Optimization failed", original_error=ValueError("test"))
        except ModelFitError as e:
            assert "Optimization failed" in str(e)


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions in exceptions module."""
    
    def test_format_validation_errors_single_error(self):
        """Test format_validation_errors with single error."""
        result = format_validation_errors(["Error 1"])
        assert "Error 1" in result
    
    def test_format_validation_errors_multiple_errors(self):
        """Test format_validation_errors with multiple errors."""
        errors = ["Error 1", "Error 2", "Error 3"]
        result = format_validation_errors(errors)
        assert "3" in result  # Should mention count
        assert "Error 1" in result
    
    def test_format_validation_errors_empty(self):
        """Test format_validation_errors with empty list."""
        result = format_validation_errors([])
        assert "No errors" in result or result == "No errors"
    
    def test_raise_if_errors_with_errors(self):
        """Test raise_if_errors raises when errors present."""
        with pytest.raises(DataValidationError):
            raise_if_errors(["Error 1", "Error 2"])
    
    def test_raise_if_errors_without_errors(self):
        """Test raise_if_errors doesn't raise when no errors."""
        # Should not raise
        raise_if_errors([])


# ============================================================================
# EXCEPTION STRINGIFICATION TESTS
# ============================================================================

class TestExceptionStringification:
    """Test exception string representations."""
    
    def test_exception_str_representation(self):
        """Test string representation of exceptions."""
        exc = DataValidationError("Test message")
        assert "Test message" in str(exc)
    
    def test_exception_repr_contains_type(self):
        """Test repr contains exception type."""
        exc = InvalidFormulaError("test", "reason")
        exc_repr = repr(exc)
        assert "InvalidFormulaError" in exc_repr or "Formula" in exc_repr
    
    def test_invalid_parameter_error_representation(self):
        """Test InvalidParameterError string representation."""
        exc = InvalidParameterError("param", "value", "reason")
        exc_str = str(exc)
        assert "param" in exc_str
        assert "value" in exc_str


# ============================================================================
# EXCEPTION PROPAGATION TESTS
# ============================================================================

class TestExceptionPropagation:
    """Test exception propagation through code."""
    
    def test_exception_propagates_through_function(self):
        """Test that exceptions propagate correctly through functions."""
        def inner_function():
            raise ModelNotFittedError("Model not fitted")
        
        def outer_function():
            inner_function()
        
        with pytest.raises(ModelNotFittedError):
            outer_function()
    
    def test_exception_can_be_re_raised(self):
        """Test that exceptions can be re-raised."""
        try:
            try:
                raise DataValidationError("Original error")
            except DataValidationError:
                raise ConvergenceError("Re-raised as convergence error")
        except ConvergenceError as e:
            assert "convergence" in str(e).lower()
    
    def test_exception_with_traceback(self):
        """Test that exceptions preserve traceback information."""
        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise ModelFitError(f"Wrapped: {str(e)}") from e
        except ModelFitError as e:
            assert e.__cause__ is not None


# ============================================================================
# EXCEPTION DOCUMENTATION
# ============================================================================

class TestExceptionDocumentation:
    """Test that exceptions have proper documentation."""
    
    def test_data_validation_error_has_docstring(self):
        """Test DataValidationError has documentation."""
        assert DataValidationError.__doc__ is not None
    
    def test_model_fit_error_has_docstring(self):
        """Test ModelFitError has documentation."""
        assert ModelFitError.__doc__ is not None
    
    def test_diagnostic_error_has_docstring(self):
        """Test DiagnosticError has documentation."""
        assert DiagnosticError.__doc__ is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

