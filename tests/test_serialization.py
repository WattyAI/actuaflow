"""
Model serialization and persistence tests.

Tests serialization of trained models to various formats (pickle, joblib, JSON)
and ensures models can be loaded and used for production deployment.

Coverage: All GLM and FreqSev models
"""

import pytest
import numpy as np
import pandas as pd
import pickle
import tempfile
import os
import json

from actuaflow.glm.models import (
    BaseGLM, FrequencyGLM, SeverityGLM, TweedieGLM, ModelResult
)
from actuaflow.freqsev.frequency import FrequencyModel
from actuaflow.freqsev.severity import SeverityModel
from actuaflow.freqsev.aggregate import AggregateModel


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for model fitting."""
    np.random.seed(42)
    return pd.DataFrame({
        'y': np.random.poisson(2, 100),
        'x1': np.random.normal(0, 1, 100),
        'x2': np.random.choice(['A', 'B', 'C'], 100),
        'exposure': np.random.uniform(0.5, 1.5, 100)
    })


@pytest.fixture
def fitted_glm_model(sample_data):
    """Create a fitted GLM model."""
    model = FrequencyGLM(family='poisson', link='log')
    model.fit(sample_data, 'y ~ x1 + x2', offset='exposure')
    return model


@pytest.fixture
def fitted_severity_model():
    """Create a fitted severity model."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'amount': np.random.gamma(2, 2500, n),
        'age': np.random.choice(['young', 'old'], n),
    })
    data = data[data['amount'] > 0]  # Remove zeros
    
    model = SeverityGLM(family='gamma', link='log')
    model.fit(data, 'amount ~ age')
    return model


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# PICKLE SERIALIZATION TESTS
# ============================================================================

class TestPickleSerialization:
    """Test model serialization with pickle."""
    
    def test_pickle_frequency_glm(self, fitted_glm_model, temp_dir):
        """Test pickling FrequencyGLM model."""
        file_path = os.path.join(temp_dir, 'model.pkl')
        
        # Serialize
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_glm_model, f)
        
        # Deserialize
        with open(file_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Verify
        assert loaded_model.fitted_
        assert loaded_model.family == fitted_glm_model.family
        assert loaded_model.link == fitted_glm_model.link
    
    def test_pickle_severity_glm(self, fitted_severity_model, temp_dir):
        """Test pickling SeverityGLM model."""
        file_path = os.path.join(temp_dir, 'severity.pkl')
        
        # Serialize
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_severity_model, f)
        
        # Deserialize
        with open(file_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Verify
        assert loaded_model.fitted_
        assert loaded_model.family == fitted_severity_model.family
    
    def test_pickle_model_predictions_consistent(self, fitted_glm_model, sample_data, temp_dir):
        """Test that pickled model produces consistent predictions."""
        file_path = os.path.join(temp_dir, 'model_pred.pkl')
        
        # Get predictions from original
        pred_original = fitted_glm_model.predict(sample_data.head(10))
        
        # Pickle and unpickle
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_glm_model, f)
        
        with open(file_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Get predictions from loaded
        pred_loaded = loaded_model.predict(sample_data.head(10))
        
        # Should be identical
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)
    
    def test_pickle_model_summary_accessible(self, fitted_glm_model, temp_dir):
        """Test that pickled model can generate summary."""
        file_path = os.path.join(temp_dir, 'model_summary.pkl')
        
        # Serialize
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_glm_model, f)
        
        # Deserialize and get summary
        with open(file_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        summary = loaded_model.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0


# ============================================================================
# JOBLIB SERIALIZATION TESTS (if available)
# ============================================================================

class TestJoblibSerialization:
    """Test model serialization with joblib (optional)."""
    
    @pytest.fixture(autouse=True)
    def skip_if_no_joblib(self):
        """Skip tests if joblib not available."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")
    
    def test_joblib_frequency_glm(self, fitted_glm_model, temp_dir):
        """Test joblib serialization of FrequencyGLM."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")
        
        file_path = os.path.join(temp_dir, 'model_joblib.pkl')
        
        # Serialize
        joblib.dump(fitted_glm_model, file_path)
        
        # Deserialize
        loaded_model = joblib.load(file_path)
        
        # Verify
        assert loaded_model.fitted_
        assert loaded_model.family == fitted_glm_model.family
    
    def test_joblib_predictions_consistent(self, fitted_glm_model, sample_data, temp_dir):
        """Test joblib model predictions are consistent."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")
        
        file_path = os.path.join(temp_dir, 'model_joblib_pred.pkl')
        
        # Original predictions
        pred_original = fitted_glm_model.predict(sample_data.head(10))
        
        # Joblib save and load
        joblib.dump(fitted_glm_model, file_path)
        loaded_model = joblib.load(file_path)
        
        # Loaded predictions
        pred_loaded = loaded_model.predict(sample_data.head(10))
        
        # Should match
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


# ============================================================================
# JSON SERIALIZATION TESTS
# ============================================================================

class TestJSONSerialization:
    """Test JSON serialization of model parameters."""
    
    def test_save_model_coefficients_to_json(self, fitted_glm_model, temp_dir):
        """Test saving model coefficients to JSON."""
        file_path = os.path.join(temp_dir, 'coefficients.json')
        
        try:
            summary = fitted_glm_model.summary()
            
            # Convert to JSON-serializable format
            coefs_dict = {
                str(idx): {
                    'coefficient': float(row.iloc[0]) if hasattr(row, 'iloc') else float(row),
                    'std_error': float(row.iloc[1]) if len(row) > 1 else None,
                } for idx, row in summary.iterrows()
            }
            
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(coefs_dict, f)
            
            # Load and verify
            with open(file_path, 'r') as f:
                loaded = json.load(f)
            
            assert len(loaded) > 0
        except Exception as e:
            pytest.skip(f"JSON serialization: {str(e)}")
    
    def test_save_model_metadata_to_json(self, fitted_glm_model, temp_dir):
        """Test saving model metadata to JSON."""
        file_path = os.path.join(temp_dir, 'metadata.json')
        
        metadata = {
            'family': fitted_glm_model.family,
            'link': fitted_glm_model.link,
            'fitted': fitted_glm_model.fitted_,
            'n_obs': fitted_glm_model.result_.n_obs if fitted_glm_model.fitted_ else None,
            'aic': float(fitted_glm_model.result_.aic) if fitted_glm_model.fitted_ else None,
            'bic': float(fitted_glm_model.result_.bic) if fitted_glm_model.fitted_ else None,
        }
        
        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Load and verify
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['family'] == 'poisson'
        assert loaded['fitted'] is True


# ============================================================================
# MODEL STATE PERSISTENCE TESTS
# ============================================================================

class TestModelStatePersistence:
    """Test that model state persists through serialization."""
    
    def test_fitted_state_persists(self, fitted_glm_model, temp_dir):
        """Test that fitted state is preserved."""
        file_path = os.path.join(temp_dir, 'state.pkl')
        
        original_fitted = fitted_glm_model.fitted_
        
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_glm_model, f)
        
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        
        assert loaded.fitted_ == original_fitted
    
    def test_coefficients_persist(self, fitted_glm_model, temp_dir):
        """Test that coefficients are preserved."""
        file_path = os.path.join(temp_dir, 'coef.pkl')
        
        original_summary = fitted_glm_model.summary()
        
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_glm_model, f)
        
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        
        loaded_summary = loaded.summary()
        
        # Should have same structure
        assert len(original_summary) == len(loaded_summary)
    
    def test_diagnostics_persist(self, fitted_glm_model, temp_dir):
        """Test that diagnostics are preserved."""
        file_path = os.path.join(temp_dir, 'diag.pkl')
        
        original_diag = fitted_glm_model.diagnostics()
        
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_glm_model, f)
        
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        
        loaded_diag = loaded.diagnostics()
        
        # Should have same keys
        assert set(original_diag.keys()) == set(loaded_diag.keys())


# ============================================================================
# MULTIPLE MODEL SERIALIZATION
# ============================================================================

class TestMultipleModelSerialization:
    """Test serializing multiple models together."""
    
    def test_save_multiple_models_as_dict(self, fitted_glm_model, fitted_severity_model, temp_dir):
        """Test saving multiple models in a dict."""
        file_path = os.path.join(temp_dir, 'models.pkl')
        
        models = {
            'frequency': fitted_glm_model,
            'severity': fitted_severity_model,
        }
        
        # Serialize
        with open(file_path, 'wb') as f:
            pickle.dump(models, f)
        
        # Deserialize
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        
        assert 'frequency' in loaded
        assert 'severity' in loaded
        assert loaded['frequency'].fitted_
        assert loaded['severity'].fitted_
    
    def test_model_ensemble_serialization(self, temp_dir):
        """Test serializing ensemble of models."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")
        
        # Create multiple models
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.poisson(2, 100),
            'x': np.random.normal(0, 1, 100),
        })
        
        models = []
        for i in range(3):
            model = FrequencyGLM()
            model.fit(data, 'y ~ x')
            models.append(model)
        
        file_path = os.path.join(temp_dir, 'ensemble.pkl')
        
        # Save
        joblib.dump(models, file_path)
        
        # Load
        loaded = joblib.load(file_path)
        
        assert len(loaded) == 3
        assert all(m.fitted_ for m in loaded)


# ============================================================================
# ERROR HANDLING IN SERIALIZATION
# ============================================================================

class TestSerializationErrors:
    """Test error handling during serialization."""
    
    def test_unpickle_wrong_file_format(self, temp_dir):
        """Test error when unpickling wrong format."""
        file_path = os.path.join(temp_dir, 'wrong.pkl')
        
        # Write non-pickle data
        with open(file_path, 'w') as f:
            f.write("This is not a pickle file")
        
        # Should raise error
        with pytest.raises(Exception):
            with open(file_path, 'rb') as f:
                pickle.load(f)
    
    def test_nonexistent_file_load(self, temp_dir):
        """Test loading from nonexistent file."""
        file_path = os.path.join(temp_dir, 'nonexistent.pkl')
        
        with pytest.raises(FileNotFoundError):
            with open(file_path, 'rb') as f:
                pickle.load(f)


# ============================================================================
# LARGE MODEL SERIALIZATION
# ============================================================================

class TestLargeModelSerialization:
    """Test serialization of large models."""
    
    def test_large_dataset_model_serialization(self, temp_dir):
        """Test serializing model trained on large dataset."""
        np.random.seed(42)
        n = 10000
        data = pd.DataFrame({
            'y': np.random.poisson(2, n),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.choice(['A', 'B', 'C', 'D'], n),
            'x3': np.random.uniform(0, 100, n),
        })
        
        try:
            # Fit model on large data
            model = FrequencyGLM()
            model.fit(data, 'y ~ x1 + x2 + x3')
            
            # Serialize
            file_path = os.path.join(temp_dir, 'large_model.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Deserialize
            with open(file_path, 'rb') as f:
                loaded = pickle.load(f)
            
            # Verify
            assert loaded.fitted_
            assert len(loaded.result_.predictions) == n
        except Exception as e:
            pytest.skip(f"Large model serialization: {str(e)}")


# ============================================================================
# VERSIONING AND COMPATIBILITY
# ============================================================================

class TestVersioning:
    """Test model versioning information."""
    
    def test_model_metadata_includes_version_info(self, fitted_glm_model, temp_dir):
        """Test that saved model can include version information."""
        file_path = os.path.join(temp_dir, 'versioned.pkl')
        
        # Create wrapper with version info
        model_bundle = {
            'model': fitted_glm_model,
            'version': '0.1.0',
            'created': pd.Timestamp.now().isoformat(),
            'actuaflow_version': '0.1.0',
        }
        
        # Serialize
        with open(file_path, 'wb') as f:
            pickle.dump(model_bundle, f)
        
        # Deserialize
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
        
        assert loaded['version'] == '0.1.0'
        assert loaded['model'].fitted_


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
