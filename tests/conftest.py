"""
Pytest configuration and shared fixtures for ActuaFlow tests
"""

import pytest
import numpy as np
import pandas as pd
import warnings


@pytest.fixture(scope='session')
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_frequency_data():
    """Generate sample frequency modeling data."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        'policy_id': [f'P{i:04d}' for i in range(n)],
        'claim_count': np.random.poisson(0.1, n),
        'exposure': np.random.uniform(0.5, 1.0, n),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], n),
        'vehicle_type': np.random.choice(['sedan', 'suv', 'sports'], n),
        'region': np.random.choice(['urban', 'suburban', 'rural'], n)
    })


@pytest.fixture
def sample_severity_data():
    """Generate sample severity modeling data."""
    np.random.seed(42)
    n = 100
    
    # Generate claim amounts (some zeros for testing)
    amounts = np.random.gamma(2, 2500, n)
    amounts[0:10] = 0  # Add some zeros
    
    return pd.DataFrame({
        'claim_id': [f'C{i:05d}' for i in range(n)],
        'policy_id': [f'P{i:04d}' for i in range(n)],
        'amount': amounts,
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], n),
        'injury_type': np.random.choice(['minor', 'moderate', 'severe'], n),
        'region': np.random.choice(['urban', 'suburban', 'rural'], n)
    })


@pytest.fixture
def sample_policies_and_claims():
    """Generate linked policy and claims data."""
    np.random.seed(42)
    n_policies = 100
    
    # Policies
    policies = pd.DataFrame({
        'policy_id': [f'P{i:04d}' for i in range(n_policies)],
        'exposure': np.random.uniform(0.5, 1.0, n_policies),
        'age_group': np.random.choice(['18-25', '26-35', '36+'], n_policies),
        'vehicle_type': np.random.choice(['sedan', 'suv'], n_policies)
    })
    
    # Claims (some policies have claims)
    claims_list = []
    for i in range(n_policies):
        # 10% chance of claim
        if np.random.random() < 0.1:
            claims_list.append({
                'claim_id': f'C{len(claims_list):05d}',
                'policy_id': f'P{i:04d}',
                'amount': np.random.gamma(2, 2500)
            })
    
    claims = pd.DataFrame(claims_list)
    
    return policies, claims


@pytest.fixture
def sample_loadings():
    """Sample premium loadings."""
    return {
        'inflation': 0.03,
        'expense_ratio': 0.15,
        'commission': 0.10,
        'profit_margin': 0.05,
        'tax_rate': 0.02
    }


# Configure warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )