# ActuaFlow

**Modern Actuarial Pricing Library for Non-Life Insurance**

ActuaFlow is a Python library for building GLM-based frequency-severity pricing models with a focus on actuarial best practices.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Author: Michael Watson](https://img.shields.io/badge/Author-Michael%20Watson-blue.svg)](mailto:michael@watsondataandrisksolutions.com)

## Features

### GLM Framework
- **Distribution families**: Poisson, Negative Binomial, Gamma, Inverse Gaussian, Lognormal, Tweedie
- **Link functions**: Log, identity, inverse, inverse squared
- **Proper offset handling** for exposure-based frequency models
- **Comprehensive diagnostics**: AIC, BIC, deviance, dispersion, overdispersion tests
- **Model validation**: VIF (variance inflation factors), correlation analysis
- **Predictive performance**: Lift curves, Gini index, residual plots
- **Formula parsing**: R-style formula notation with interactions and transformations
- **Robust input validation**: NaN detection, response/weight validation, formula parsing

### Frequency-Severity Workflow
- **FrequencyModel**: Poisson/Negative Binomial models for claim counts
- **SeverityModel**: Gamma/Lognormal/Inverse Gaussian for claim amounts
- **AggregateModel**: Combined frequency-severity with pure premium calculation
- **Automatic data preparation**: Policy+claims aggregation, automatic frequency table generation
- **Flexible column naming**: Support for different policy/claim ID column names
- **Premium loading**: Sequential loadings for inflation, expenses, commission, profit, taxes
- **Export capabilities**: Model summaries, factor relativities, rating tables

### Exposure & Rating Tools
- **Exposure calculation**: Days, months, years methods with custom date columns
- **Rate per exposure**: Calculate rate per unit exposure given pure premium
- **Class plan creation**: Rating tables with base rates and factor relativities
- **Factor application**: Apply factor levels from rating tables to policies
- **Trend adjustment**: Historical-to-current trending with configurable rates
- **On-leveling**: Mid-term rate change adjustments
- **Experience modification**: Experience rating factors (EMR/XMod)
- **Credibility weighting**: Complement experience data with prior information

### Portfolio & Impact Analysis
- **Premium impact analysis**: Calculate impact of factor changes on portfolio premium
- **Mix-shift decomposition**: Separate impact of rate changes from volume/mix changes
- **Segment impact**: Winners and losers analysis by segment
- **Rate adequacy**: Loss ratio, earned premium, change metrics
- **Price elasticity**: Demand curve estimation and revenue optimization
- **Elasticity curves**: Revenue impact curves across price ranges
- **Retention modeling**: Retention rate impact on portfolio profitability
- **Scenario analysis**: Compare multiple rating strategies

### Data Utilities
- **Data loading**: CSV/Parquet support with polars fast reader
- **Data validation**: Schema checking, missing value detection, type validation
- **Train-test split**: Stratified splits with random state control
- **Time series cross-validation**: Temporal splits for time-dependent data
- **Cross-validation scoring**: Model performance across folds

### Advanced Features
- **Statsmodels integration**: Leverages industry-standard GLM library
- **Robust error handling**: Custom exceptions with detailed diagnostics
- **Model serialization**: Export/import for production deployment
- **Testing**: 68%+ test coverage with pytest
- **Documentation**: Docstrings, examples, user guide

## Installation

```bash
pip install actuaflow
```

Or install from source:

```bash
git clone https://github.com/actuaflow/actuaflow.git
cd actuaflow
pip install -e .
```

## Quick Start

### Basic Frequency-Severity Model

```python
import actuaflow as af
import pandas as pd

# Load data
policies = af.load_data('policies.csv')
claims = af.load_data('claims.csv')

# 1. Frequency Model
freq_model = af.FrequencyModel(family='poisson', link='log')
freq_model.prepare_data(policies, claims, policy_id='policy_id')
freq_model.fit(
    formula='claim_count ~ age_group + vehicle_type + region',
    offset='exposure'
)

print(freq_model.summary())
print(f"Dispersion: {freq_model.diagnostics_['dispersion']:.3f}")

# 2. Severity Model
sev_model = af.SeverityModel(family='gamma', link='log')
sev_model.prepare_data(claims, policies, policy_id='policy_id')
sev_model.fit(formula='amount ~ age_group + injury_type')

print(sev_model.summary())

# 3. Combined Model
agg_model = af.combine_models(freq_model, sev_model)
print(f"Base Pure Premium: ${agg_model.base_pure_premium_:.2f}")

# Get factor table
factor_table = agg_model.create_factor_table()
print(factor_table)

# 4. Premium Calculation
pure_premium = agg_model.predict_pure_premium(policies, exposure='exposure')

loadings = {
    'inflation': 0.03,
    'expense_ratio': 0.15,
    'commission': 0.10,
    'profit_margin': 0.05,
    'tax_rate': 0.02
}

premium_df = af.calculate_premium(pure_premium, loadings, exposure=policies['exposure'])
print(f"Total Loaded Premium: ${premium_df['loaded_premium'].sum():,.0f}")
```

### Model Diagnostics

### Working with Different Column Names

ActuaFlow functions that prepare frequency data accept flexible column names for policy and claims identifiers. Example:

```python
from actuaflow import FrequencyModel

# Policy table uses 'policy_key', claims table uses 'pol_id'
freq_model = FrequencyModel()
freq_model.prepare_data(
    policy_data=policies,
    claims_data=claims,
    policy_id_policy='policy_key',
    policy_id_claims='pol_id'
)
```

This makes it easy to work with data exports that use different naming conventions.

### Calculating Policy Exposure

Use the `calculate_exposure` helper to compute exposure between a start and end date. It supports `years`, `days`, and `months` methods and accepts custom column names for the start/end dates as well as a custom output column name.

```python
from actuaflow import calculate_exposure

policies_with_exposure = calculate_exposure(
    policy_data=policies,
    start_date_col='policy_start_date',
    end_date_col='policy_end_date',
    method='years',
    exposure_col='exposure'
)

print(policies_with_exposure[['policy_id', 'exposure']].head())
```

The function validates date columns and returns a copy of the dataframe with the exposure column added.

```python
# Comprehensive diagnostics
diag = af.compute_diagnostics(freq_model.model_, policies)

print(f"AIC: {diag['aic']:.2f}")
print(f"Dispersion: {diag['dispersion']:.3f}")

# VIF for multicollinearity
if diag['vif']:
    for var, vif in diag['vif'].items():
        print(f"{var}: VIF = {vif:.2f}")

# Lift curve
y_true = policies['claim_count']
y_pred = freq_model.predict()
bins, actual, predicted = af.compute_lift_curve(y_true, y_pred, n_bins=10)

# Gini index
has_claim = (y_true > 0).astype(int)
gini = af.compute_gini_index(has_claim, y_pred)
print(f"Gini coefficient: {gini:.3f}")
```

### Exposure Rating

```python
# Compute rate per exposure
rate = af.compute_rate_per_exposure(
    pure_premium=policies['pure_premium'],
    exposure=policies['exposure'],
    loadings={'profit': 0.05, 'expenses': 0.15}
)

# Create class plan
relativities = {
    'age_group': {'18-25': 1.5, '26-35': 1.0, '36-45': 0.8, '46+': 0.7},
    'vehicle_type': {'sedan': 1.0, 'suv': 1.2, 'sports': 1.8}
}

rated_policies = af.create_class_plan(
    data=policies,
    rating_factors=['age_group', 'vehicle_type'],
    base_rate=100.0,
    relativities=relativities,
    exposure_col='exposure'
)

# Apply trend factor
trended_losses = af.apply_trend_factor(
    historical_value=100000,
    trend_rate=0.03,
    years=4
)
print(f"Trended to current: ${trended_losses:,.0f}")
```

### Portfolio Impact Analysis

```python
# Premium impact from factor changes
factor_changes = {
    'age_group': {'18-25': 1.10, '26-35': 1.00, '36+': 0.95}
}

impact = af.compute_premium_impact(
    data=policies,
    base_premium_col='premium',
    factor_changes=factor_changes
)

print(f"Total impact: ${impact['premium_change'].sum():,.0f}")
print(f"Average change: {impact['premium_change_pct'].mean():.1f}%")

# Segment analysis
segment_impact = af.segment_impact_analysis(
    data=impact,
    premium_col='premium_current',
    segment_cols=['age_group', 'territory'],
    proposed_premium_col='premium_proposed'
)

# Price elasticity
elasticity = af.estimate_demand_elasticity(
    price_history=historical_data,
    price_col='average_premium',
    volume_col='policy_count'
)
print(f"Elasticity: {elasticity['elasticity']:.3f}")

# Elasticity curve
curve = af.compute_elasticity_curve(
    current_price=100,
    current_volume=10000,
    elasticity=-1.5,
    price_range=(80, 120)
)
```

## Documentation

Full documentation available at: [https://actuaflow.readthedocs.io](https://actuaflow.readthedocs.io)

- [API Reference](https://actuaflow.readthedocs.io/api/)
- [User Guide](https://actuaflow.readthedocs.io/guide/)
- [Examples](https://actuaflow.readthedocs.io/examples/)

## Examples

Jupyter notebooks demonstrating common workflows:

- `examples/motor_insurance.ipynb` - Complete motor insurance pricing
- `examples/property_pricing.ipynb` - Property insurance GLMs
- `examples/glm_basics.ipynb` - GLM fundamentals
- `examples/exposure_rating.ipynb` - Exposure-based rating
- `examples/portfolio_analysis.ipynb` - Premium impact and elasticity

## Package Structure

```
actuaflow/
├── glm/              # GLM models and diagnostics
│   ├── models.py     # FrequencyGLM, SeverityGLM, TweedieGLM
│   └── diagnostics.py # VIF, lift curves, Gini, etc.
├── freqsev/          # Frequency-severity workflow
│   ├── frequency.py  # FrequencyModel
│   ├── severity.py   # SeverityModel
│   └── aggregate.py  # AggregateModel, premium calculation
├── exposure/         # Exposure rating tools
│   ├── rating.py     # Class plans, rate per exposure
│   └── trending.py   # Trend factors, on-leveling
├── portfolio/        # Portfolio analysis
│   ├── impact.py     # Premium impact, mix-shift
│   └── elasticity.py # Price elasticity, optimization
└── utils/            # Utilities
    ├── data.py       # Data loading, preparation
    └── validation.py # Input validation
```

## Key Concepts

### Frequency-Severity Approach

The package follows the standard actuarial practice of modeling claim frequency and severity separately:

- **Frequency Model**: Models claim counts per policy (Poisson or Negative Binomial)
- **Severity Model**: Models claim amounts given a claim occurred (Gamma or Lognormal)
- **Pure Premium**: Frequency × Severity
- **Loaded Premium**: Pure Premium × Loading Factors

### Offset in Frequency Models

Exposure is handled as an **offset** (not a predictor):

```python
# Correct - offset parameter
freq_model.fit(data, formula='claim_count ~ age + vehicle', offset='exposure')

# Incorrect - don't include offset in formula
# freq_model.fit(data, formula='claim_count ~ age + vehicle + offset(log(exposure))')
```

### Factor Relativities

For log-link models, relativities are exp(coefficient):

```python
relativities = freq_model.get_relativities()
# age_group[18-25]: 1.50 means 50% higher frequency than base
# age_group[36-45]: 0.80 means 20% lower frequency than base
```

## Best Practices

1. **Always check dispersion** for Poisson models - use Negative Binomial if overdispersed
2. **Filter zero claims** when fitting severity models
3. **Use proper offsets** for frequency models with varying exposure
4. **Check VIF** for multicollinearity before finalizing model
5. **Validate on holdout data** using lift curves and Gini
6. **Document model changes** using export functions
7. **Monitor actual vs expected** regularly

## Requirements

- Python >= 3.9
- pandas >= 2.0.0
- polars >= 0.19.0 (for fast data loading)
- numpy >= 1.24.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0

## Contributing

ActuaFlow is a **solo-maintained project**. External code contributions are not accepted at this time. However, feedback, bug reports, and feature requests are always welcome!

**To report issues or request features:**
- Email: michael@watsondataandrisksolutions.com
- Subject: `[BUG REPORT]` for bugs or `[FEATURE REQUEST]` for ideas

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

**Mozilla Public License v2.0 (MPL-2.0)**

ActuaFlow is licensed under the Mozilla Public License v2.0, an OSI-approved open-source license that:
- ✅ Allows commercial and research use
- ✅ Permits use in proprietary applications
- ✅ Requires attribution and license inclusion

**Special Licensing Available:** Implementation services, consulting, GUI software, and alternative licensing terms. Contact the author for details.

See [LICENSING.md](LICENSING.md) for comprehensive licensing information.

**Author:** Michael Watson (michael@watsondataandrisksolutions.com)  
**Copyright:** (c) 2026–present Michael Watson

For complete details, see:
- [LICENSE](LICENSE) - Full legal text
- [LICENSING.md](LICENSING.md) - Comprehensive usage guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributor guidelines

**Licensing Options:**
- **Open Source:** MPL-2.0 (free for research and commercial use)
- **Commercial:** Custom licensing available - contact michael@watsondataandrisksolutions.com

## Citation

If you use ActuaFlow in academic research, please cite:

```bibtex
@software{actuaflow2025,
  title = {ActuaFlow: Modern Actuarial Pricing Library},
  author = {Watson, Michael},
  year = {2025},
  url = {https://github.com/actuaflow/actuaflow},
  license = {MPL-2.0}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/actuaflow/actuaflow/issues)
- **Discussions**: [GitHub Discussions](https://github.com/actuaflow/actuaflow/discussions)
- **Documentation**: [ReadTheDocs](https://actuaflow.readthedocs.io)

## Acknowledgments

ActuaFlow is built on:
- [statsmodels](https://www.statsmodels.org/) for GLM implementation
- [polars](https://www.pola.rs/) for fast data processing
- Industry best practices from CAS monographs and academic research

---

**ActuaFlow** - Modern tools for modern actuaries