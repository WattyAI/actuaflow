"""
Integration Tests for ActuaFlow Package

Tests full workflows combining multiple modules:
- Data loading and validation
- Frequency modeling
- Cross-validation with time-series splits
- Portfolio impact analysis
- Complete actuarial workflows

Author: Michael Watson
Email: michael@watsondataandrisksolutions.com
License: MPL-2.0
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from actuaflow.utils.data import split_train_test
from actuaflow.utils.validation import (
    validate_formula, validate_family_link, validate_column_exists,
    validate_positive_values, validate_no_missing
)
from actuaflow.utils.cross_validation import cross_val_score
from actuaflow.freqsev.frequency import FrequencyModel
from actuaflow.freqsev.severity import SeverityModel
from actuaflow.portfolio.impact import (
    compute_premium_impact, factor_sensitivity, mix_shift_analysis,
    segment_impact_analysis, rate_adequacy_analysis
)
from actuaflow.glm.models import BaseGLM, FrequencyGLM, SeverityGLM

logger = logging.getLogger(__name__)


class TestDataPipelineIntegration:
    """Test data loading, validation, and preparation workflows."""
    
    def test_complete_frequency_data_pipeline(self):
        """Test complete workflow: load → validate → prepare data."""
        # Create sample datasets
        policies = pd.DataFrame({
            'policy_id': [f'P{i:04d}' for i in range(50)],
            'exposure': np.random.uniform(0.5, 1.0, 50),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 50),
            'vehicle_type': np.random.choice(['sedan', 'suv', 'sports'], 50),
            'region': np.random.choice(['urban', 'suburban', 'rural'], 50)
        })
        
        claims = pd.DataFrame({
            'claim_id': [f'C{i:05d}' for i in range(30)],
            'policy_id': np.random.choice(policies['policy_id'], 30),
            'amount': np.random.gamma(2, 2500, 30)
        })
        
        # Validate policies exist
        assert len(policies) > 0
        assert 'policy_id' in policies.columns
        assert 'exposure' in policies.columns
        
        # Validate column values
        assert validate_column_exists(policies, 'exposure')
        assert validate_positive_values(policies, 'exposure')
        
        # Validate formula
        assert validate_formula('exposure ~ age_group + vehicle_type')
        
        # Validate family-link combination
        assert validate_family_link('poisson', 'log')
        
        logger.info("✓ Complete frequency data pipeline test passed")
    
    def test_severity_data_validation(self):
        """Test severity data validation."""
        policies = pd.DataFrame({
            'policy_id': [f'P{i:04d}' for i in range(50)],
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 50),
            'vehicle_type': np.random.choice(['sedan', 'suv', 'sports'], 50),
        })
        
        claims = pd.DataFrame({
            'claim_id': [f'C{i:05d}' for i in range(80)],
            'policy_id': np.random.choice(policies['policy_id'], 80),
            'amount': np.concatenate([
                [0] * 10,
                np.random.gamma(2, 2500, 70)
            ])
        })
        
        # Verify data structure
        assert len(claims) > 0
        assert 'amount' in claims.columns
        assert 'policy_id' in claims.columns
        
        # Test no missing values
        assert validate_no_missing(claims, columns=['amount', 'policy_id'], raise_error=False)
        
        logger.info("✓ Severity data validation test passed")
    
    def test_train_test_split_preserves_data_integrity(self):
        """Test that train/test split doesn't corrupt data."""
        data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.normal(0, 1, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        train, test = split_train_test(data, test_size=0.2, random_state=42)
        
        # Verify no overlap
        assert not set(train['id']).intersection(set(test['id']))
        
        # Verify all data accounted for
        assert len(train) + len(test) == len(data)
        
        # Verify dtypes preserved
        assert train.dtypes.equals(data.dtypes)
        assert test.dtypes.equals(data.dtypes)
        
        logger.info("✓ Train/test split integrity test passed")


class TestFrequencyModelingWorkflow:
    """Test complete frequency modeling workflows."""
    
    def test_complete_frequency_modeling_workflow(self):
        """Test frequency model initialization and basic operations."""
        np.random.seed(42)
        
        # Create sample data
        freq_data = pd.DataFrame({
            'claim_count': np.random.poisson(0.2, 100),
            'exposure': np.random.uniform(0.5, 1.0, 100),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 100),
            'vehicle_type': np.random.choice(['sedan', 'suv'], 100),
        })
        
        # Initialize model
        freq_model = FrequencyModel(family='poisson', link='log')
        
        # Fit model
        freq_model.fit(
            formula='claim_count ~ age_group + vehicle_type',
            data=freq_data,
            offset='exposure'
        )
        
        assert freq_model.is_fitted()
        
        # Get predictions
        predictions = freq_model.predict(freq_data)
        assert len(predictions) == len(freq_data)
        assert (predictions >= 0).all()
        
        # Get summary
        summary = freq_model.summary()
        assert summary is not None
        
        logger.info("✓ Complete frequency modeling workflow test passed")


class TestPortfolioImpactAnalysis:
    """Test portfolio impact analysis workflows."""
    
    def test_premium_impact_calculation(self):
        """Test premium impact calculation with factor changes."""
        portfolio = pd.DataFrame({
            'policy_id': [f'P{i:04d}' for i in range(100)],
            'premium': np.random.uniform(500, 1500, 100),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 100),
            'vehicle_type': np.random.choice(['sedan', 'suv'], 100),
            'region': np.random.choice(['urban', 'suburban', 'rural'], 100)
        })
        
        factor_changes = {
            'age_group': {'18-25': 1.10, '26-35': 1.00, '36-45': 0.95, '46+': 0.90},
            'vehicle_type': {'sedan': 1.00, 'suv': 1.15}
        }
        
        impact = compute_premium_impact(
            data=portfolio,
            base_premium_col='premium',
            factor_changes=factor_changes
        )
        
        assert 'premium_current' in impact.columns
        assert 'premium_proposed' in impact.columns
        assert 'premium_change' in impact.columns
        assert 'premium_change_pct' in impact.columns
        assert (impact['premium_proposed'] > 0).all()
        
        logger.info("✓ Premium impact calculation test passed")
    
    def test_factor_sensitivity_analysis(self):
        """Test factor sensitivity analysis."""
        portfolio = pd.DataFrame({
            'policy_id': [f'P{i:04d}' for i in range(100)],
            'premium': np.random.uniform(500, 1500, 100),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 100),
            'vehicle_type': np.random.choice(['sedan', 'suv'], 100),
        })
        
        sensitivity = factor_sensitivity(
            data=portfolio,
            base_premium_col='premium',
            factor='age_group',
            change_range=(0.8, 1.2),
            n_points=5
        )
        
        assert len(sensitivity) == 5
        assert 'factor_multiplier' in sensitivity.columns
        assert 'total_premium' in sensitivity.columns
        
        logger.info("✓ Factor sensitivity analysis test passed")
    
    def test_segment_impact_analysis(self):
        """Test segment-level impact analysis."""
        np.random.seed(42)
        portfolio = pd.DataFrame({
            'policy_id': [f'P{i:04d}' for i in range(200)],
            'premium_current': np.random.uniform(500, 1500, 200),
            'premium_proposed': np.random.uniform(450, 1600, 200),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 200),
            'territory': np.random.choice(['north', 'south', 'east', 'west'], 200)
        })
        
        segment_analysis = segment_impact_analysis(
            data=portfolio,
            premium_col='premium_current',
            segment_cols=['age_group', 'territory'],
            proposed_premium_col='premium_proposed'
        )
        
        assert len(segment_analysis) > 0
        assert 'age_group' in segment_analysis.columns
        assert 'territory' in segment_analysis.columns
        
        logger.info("✓ Segment impact analysis test passed")
    
    def test_rate_adequacy_analysis(self):
        """Test rate adequacy analysis."""
        np.random.seed(42)
        experience = pd.DataFrame({
            'policy_id': [f'P{i:04d}' for i in range(150)],
            'earned_premium': np.random.uniform(500, 1500, 150),
            'incurred_losses': np.random.uniform(200, 900, 150),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 150),
        })
        
        adequacy = rate_adequacy_analysis(
            data=experience,
            actual_losses_col='incurred_losses',
            premium_col='earned_premium',
            segment_cols=['age_group'],
            target_loss_ratio=0.65
        )
        
        assert len(adequacy) > 0
        assert 'actual_loss_ratio' in adequacy.columns
        assert 'target_loss_ratio' in adequacy.columns
        
        logger.info("✓ Rate adequacy analysis test passed")


class TestErrorHandlingAndValidation:
    """Test error handling and validation across modules."""
    
    def test_invalid_formula_handling(self):
        """Test that invalid formulas are caught."""
        with pytest.raises(Exception):
            validate_formula('invalid_formula_no_tilde')
        
        with pytest.raises(Exception):
            validate_formula('')
    
    def test_invalid_family_link_handling(self):
        """Test that invalid family-link combinations are caught."""
        with pytest.raises(Exception):
            validate_family_link('invalid_family', 'log')
        
        with pytest.raises(Exception):
            validate_family_link('poisson', 'invalid_link')
    
    def test_data_validation_catches_issues(self):
        """Test that data validation catches problems."""
        data = pd.DataFrame({'a': [1, 2, 3]})
        
        with pytest.raises(Exception):
            validate_column_exists(data, 'missing_column')
        
        data_with_negatives = pd.DataFrame({'exposure': [0.5, -0.1, 0.3]})
        
        with pytest.raises(Exception):
            validate_positive_values(data_with_negatives, 'exposure', allow_zero=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])