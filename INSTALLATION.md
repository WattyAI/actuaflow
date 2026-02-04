# ActuaFlow Installation & Setup Guide

Complete guide to installing and testing the ActuaFlow package.

## Quick Install

```bash
pip install actuaflow
```

## Install from Source

### 1. Clone Repository

```bash
git clone https://github.com/actuaflow/actuaflow.git
cd actuaflow
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n actuaflow python=3.11
conda activate actuaflow
```

### 3. Install Package

```bash
# Development install (editable mode)
pip install -e .

# Or standard install
pip install .

# With development dependencies
pip install -e ".[dev]"

# With documentation dependencies
pip install -e ".[docs]"

# With all extras
pip install -e ".[dev,docs]"
```

## Verify Installation

```python
import actuaflow as af
print(af.__version__)
print(af.__author__)

# Check available modules
print(dir(af))
```

## Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=actuaflow --cov-report=html

# Run specific test file
pytest tests/test_glm.py

# Run with verbose output
pytest -v
```

## Run Example

```bash
# Run motor insurance example
cd examples
python motor_insurance.py

# Or as Jupyter notebook
jupyter notebook motor_insurance.ipynb
```

## Build Documentation

```bash
cd docs
pip install -r requirements.txt
make html

# View documentation
open _build/html/index.html  # On Mac
# Or: xdg-open _build/html/index.html  # On Linux
# Or: start _build\html\index.html  # On Windows
```

## Dependencies

### Core Dependencies (Automatically Installed)

- numpy >= 1.24.0
- pandas >= 2.0.0
- polars >= 0.19.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0
- pyyaml >= 6.0
- openpyxl >= 3.1.0

### Optional Dependencies

For development:
```bash
pip install pytest pytest-cov black flake8 mypy
```

For visualization (recommended for examples):
```bash
pip install matplotlib seaborn plotly
```

For Jupyter notebooks:
```bash
pip install jupyter
```

## Troubleshooting

### Import Error: No module named 'actuaflow'

**Solution**: Ensure the package is installed:
```bash
pip list | grep actuaflow
# If not found, reinstall:
pip install -e .
```

### ModuleNotFoundError: No module named 'polars'

**Solution**: Install polars:
```bash
pip install polars
```

### statsmodels Version Conflict

**Solution**: Upgrade statsmodels:
```bash
pip install --upgrade statsmodels
```

### Test Failures

**Solution**: Ensure all dependencies are up to date:
```bash
pip install --upgrade -r requirements.txt
pytest -v
```

## Platform-Specific Notes

### macOS

No special considerations. Standard installation works.

### Linux

May need to install system dependencies for some packages:
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# Fedora/RHEL
sudo dnf install python3-devel
```

### Windows

- Use Anaconda/Miniconda for easiest setup
- Or ensure Microsoft Visual C++ 14.0 or greater is installed

## Performance Optimization

### Use Polars for Large Datasets

ActuaFlow automatically uses Polars for fast data loading:

```python
# This uses Polars internally (10x faster for large CSVs)
data = af.load_data('large_policies.csv', use_polars=True)
```

### Parallel Processing

For large portfolios, consider processing in chunks:

```python
# Process in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    results = model.predict(chunk)
    # Process results
```

## Development Setup

For contributing to ActuaFlow:

```bash
# Clone and install in development mode
git clone https://github.com/actuaflow/actuaflow.git
cd actuaflow
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting
black actuaflow/
flake8 actuaflow/

# Run type checking
mypy actuaflow/

# Run tests with coverage
pytest --cov=actuaflow --cov-report=html
```

## Docker Setup (Optional)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python"]
```

Build and run:
```bash
docker build -t actuaflow .
docker run -it actuaflow python
```

## Conda Environment (Alternative)

```yaml
# environment.yml
name: actuaflow
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.24
  - pandas>=2.0
  - scipy>=1.10
  - statsmodels>=0.14
  - scikit-learn>=1.3
  - pip
  - pip:
    - polars>=0.19
    - actuaflow
```

Create environment:
```bash
conda env create -f environment.yml
conda activate actuaflow
```

## Upgrading

```bash
# Upgrade from PyPI
pip install --upgrade actuaflow

# Upgrade from source
cd actuaflow
git pull
pip install --upgrade -e .
```

## Uninstalling

```bash
pip uninstall actuaflow
```

## Getting Help

- **Documentation**: https://actuaflow.readthedocs.io
- **Issues**: https://github.com/actuaflow/actuaflow/issues
- **Discussions**: https://github.com/actuaflow/actuaflow/discussions

## Quick Test

After installation, run this quick test:

```python
import actuaflow as af
import pandas as pd
import numpy as np

# Generate test data
np.random.seed(42)
test_data = pd.DataFrame({
    'policy_id': range(100),
    'claim_count': np.random.poisson(0.1, 100),
    'exposure': np.ones(100),
    'age_group': np.random.choice(['A', 'B'], 100)
})

# Fit frequency model
model = af.FrequencyGLM(family='poisson', link='log')
model.fit(test_data, 'claim_count ~ age_group', offset='exposure')

# Get summary
print(model.summary())

print("\n✓ ActuaFlow is working correctly!")
```

If this runs without errors, your installation is successful!