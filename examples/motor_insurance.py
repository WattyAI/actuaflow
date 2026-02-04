"""
ActuaFlow Example: Motor Insurance Pricing
Complete workflow from data to rated premium

This demonstrates:
1. Data preparation
2. Frequency modeling
3. Severity modeling
4. Combined model and pure premium
5. Premium loading
6. Diagnostics and validation
7. Portfolio impact analysis
8. Price optimization with elasticity
"""

# ============================================================================
# SETUP
# ============================================================================

import actuaflow as af
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print(f"ActuaFlow version: {af.__version__}")


# ============================================================================
# 1. GENERATE DEMO DATA
# ============================================================================

print("\n" + "="*80)
print("1. GENERATING DEMO MOTOR INSURANCE DATA")
print("="*80)

# Generate 10,000 policies
n_policies = 10000

policies = pd.DataFrame({
    'policy_id': [f'POL{i:05d}' for i in range(n_policies)],
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n_policies, 
                                   p=[0.15, 0.25, 0.25, 0.20, 0.15]),
    'vehicle_type': np.random.choice(['Sedan', 'SUV', 'Truck', 'Sports'], n_policies,
                                     p=[0.50, 0.30, 0.15, 0.05]),
    'region': np.random.choice(['Urban', 'Suburban', 'Rural'], n_policies,
                               p=[0.40, 0.40, 0.20]),
    'vehicle_age': np.random.randint(0, 15, n_policies),
    'exposure': np.random.uniform(0.1, 1.0, n_policies)
})

# Generate claims with realistic patterns
freq_multipliers = {
    '18-25': 1.5, '26-35': 0.9, '36-45': 0.8, '46-55': 0.9, '56+': 1.1
}
policies['freq_mult'] = policies['age_group'].map(freq_multipliers)

base_freq = 0.08
policies['expected_freq'] = base_freq * policies['freq_mult'] * policies['exposure']
policies['claim_count'] = np.random.poisson(policies['expected_freq'])

# Generate claim amounts
claims_list = []
for idx, row in policies[policies['claim_count'] > 0].iterrows():
    for _ in range(int(row['claim_count'])):
        # Severity varies by age and vehicle type
        sev_mult = freq_multipliers[row['age_group']]
        base_sev = 5000
        
        if row['vehicle_type'] == 'Sports':
            sev_mult *= 1.5
        elif row['vehicle_type'] == 'SUV':
            sev_mult *= 1.2
        
        severity = np.random.gamma(2, base_sev * sev_mult / 2)
        
        claims_list.append({
            'claim_id': f"CLM{len(claims_list):06d}",
            'policy_id': row['policy_id'],
            'amount': severity,
            'injury_type': np.random.choice(['Minor', 'Moderate', 'Severe'], 
                                           p=[0.7, 0.25, 0.05])
        })

claims = pd.DataFrame(claims_list)

print(f"\nGenerated {len(policies):,} policies")
print(f"Generated {len(claims):,} claims")
print(f"Overall frequency: {len(claims) / policies['exposure'].sum():.4f}")
print(f"Average severity: ${claims['amount'].mean():,.2f}")


# ============================================================================
# 2. FREQUENCY MODELING
# ============================================================================

print("\n" + "="*80)
print("2. FREQUENCY MODELING")
print("="*80)

# Prepare frequency data
freq_model = af.FrequencyModel(family='poisson', link='log')
freq_data = freq_model.prepare_data(policies, claims, policy_id='policy_id')

print(f"\nFrequency data: {len(freq_data):,} policies")
print(f"Policies with claims: {(freq_data['claim_count'] > 0).sum():,}")

# Fit model
freq_model.fit(
    formula='claim_count ~ age_group + vehicle_type + region',
    offset='exposure'
)

print("\nFrequency Model Summary:")
print(freq_model.summary())

# Check fit
fit_check = freq_model.check_fit()
print(f"\nModel Fit:")
print(f"AIC: {fit_check['aic']:.2f}")
print(f"Dispersion: {fit_check['dispersion']:.3f}")

if fit_check['warnings']:
    print("\nWarnings:")
    for warning in fit_check['warnings']:
        print(f"  - {warning}")

# Get relativities
print("\nFrequency Relativities:")
rel_freq = freq_model.get_relativities()
print(rel_freq.head(10))

# Diagnostics
diag_freq = af.compute_diagnostics(freq_model.model_, freq_data)

print(f"\nVIF (Multicollinearity):")
if diag_freq['vif']:
    for var, vif in diag_freq['vif'].items():
        status = "OK" if vif < 5 else "WARNING" if vif < 10 else "SEVERE"
        print(f"  {var}: {vif:.2f} [{status}]")


# ============================================================================
# 3. SEVERITY MODELING
# ============================================================================

print("\n" + "="*80)
print("3. SEVERITY MODELING")
print("="*80)

# Prepare severity data
sev_model = af.SeverityModel(family='gamma', link='log')
sev_data = sev_model.prepare_data(
    claims, 
    policies, 
    policy_id='policy_id',
    filter_zeros=True,
    large_claim_percentile=0.95  # Cap at 95th percentile
)

print(f"\nSeverity data: {len(sev_data):,} claims")

# Fit model
sev_model.fit(formula='amount ~ age_group + vehicle_type + injury_type')

print("\nSeverity Model Summary:")
print(sev_model.summary())

# Get relativities
print("\nSeverity Relativities:")
rel_sev = sev_model.get_relativities()
print(rel_sev.head(10))


# ============================================================================
# 4. COMBINED MODEL & PURE PREMIUM
# ============================================================================

print("\n" + "="*80)
print("4. COMBINED MODEL & PURE PREMIUM")
print("="*80)

# Combine models
agg_model = af.combine_models(freq_model, sev_model)

print(f"\nBase Rates:")
print(f"Base Frequency: {agg_model.base_frequency_:.6f}")
print(f"Base Severity: ${agg_model.base_severity_:,.2f}")
print(f"Base Pure Premium: ${agg_model.base_pure_premium_:,.2f}")

# Factor table
factor_table = agg_model.create_factor_table()
print(f"\nCombined Factor Table ({len(factor_table)} factors):")
print(factor_table.head(15))

# Predict pure premium
pure_premium = agg_model.predict_pure_premium(policies, exposure='exposure')
print(f"\nTotal Pure Premium: ${pure_premium.sum():,.2f}")

# Distribution
print(f"\nPure Premium Distribution:")
print(f"  Mean: ${pure_premium.mean():,.2f}")
print(f"  Median: ${pure_premium.median():,.2f}")
print(f"  Std Dev: ${pure_premium.std():,.2f}")
print(f"  Min: ${pure_premium.min():,.2f}")
print(f"  Max: ${pure_premium.max():,.2f}")


# ============================================================================
# 5. PREMIUM LOADING
# ============================================================================

print("\n" + "="*80)
print("5. PREMIUM LOADING")
print("="*80)

# Define loadings
loadings = {
    'inflation': 0.03,      # 3% claim inflation
    'expense_ratio': 0.15,  # 15% expenses
    'commission': 0.10,     # 10% commission
    'profit_margin': 0.05,  # 5% profit
    'tax_rate': 0.02        # 2% premium tax
}

print("\nLoading Assumptions:")
for key, value in loadings.items():
    print(f"  {key.replace('_', ' ').title()}: {value*100:.1f}%")

# Calculate premium
premium_df = af.calculate_premium(pure_premium, loadings, exposure=policies['exposure'])

print(f"\nPremium Summary:")
print(f"  Pure Premium: ${premium_df['pure_premium'].sum():,.2f}")
print(f"  Loaded Premium: ${premium_df['loaded_premium'].sum():,.2f}")
print(f"  Loading Factor: {premium_df['loaded_premium'].sum() / premium_df['pure_premium'].sum():.3f}")

# Waterfall
waterfall = af.premium_waterfall(premium_df['pure_premium'].sum(), loadings)
print("\nPremium Loading Waterfall:")
print(waterfall)


# ============================================================================
# 6. DIAGNOSTICS & VALIDATION
# ============================================================================

print("\n" + "="*80)
print("6. DIAGNOSTICS & VALIDATION")
print("="*80)

# Lift curve for frequency
y_true = freq_data['claim_count']
y_pred = freq_model.predict()

bins, actual_rates, pred_rates = af.compute_lift_curve(y_true, y_pred, n_bins=10)

print("\nFrequency Lift Curve (Deciles):")
lift_df = pd.DataFrame({
    'Decile': bins + 1,
    'Actual Rate': actual_rates,
    'Predicted Rate': pred_rates,
    'Lift': actual_rates / actual_rates.mean()
})
print(lift_df)

# Gini index
has_claim = (y_true > 0).astype(int)
gini = af.compute_gini_index(has_claim, y_pred)
print(f"\nGini Coefficient: {gini:.3f}")
print(f"AUC: {(gini + 1) / 2:.3f}")

# Double lift
double_lift = af.compute_double_lift(y_true, y_pred, n_bins=10)
print("\nDouble Lift Chart:")
print(double_lift)


# ============================================================================
# 7. PORTFOLIO IMPACT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("7. PORTFOLIO IMPACT ANALYSIS")
print("="*80)

# Add current premium to policies
policies['current_premium'] = premium_df['loaded_premium'].values

# Simulate factor changes (10% increase for young drivers, 5% decrease for older)
factor_changes = {
    'age_group': {
        '18-25': 1.10,
        '26-35': 1.00,
        '36-45': 0.95,
        '46-55': 0.95,
        '56+': 0.98
    }
}

impact = af.compute_premium_impact(
    data=policies,
    base_premium_col='current_premium',
    factor_changes=factor_changes
)

print(f"\nImpact Summary:")
print(f"  Total Current Premium: ${impact['premium_current'].sum():,.2f}")
print(f"  Total Proposed Premium: ${impact['premium_proposed'].sum():,.2f}")
print(f"  Total Change: ${impact['premium_change'].sum():,.2f}")
print(f"  Average Change: {impact['premium_change_pct'].mean():.2f}%")

# By age group
age_impact = impact.groupby('age_group').agg({
    'premium_current': 'sum',
    'premium_proposed': 'sum',
    'premium_change': 'sum',
    'premium_change_pct': 'mean'
}).round(2)

print("\nImpact by Age Group:")
print(age_impact)


# ============================================================================
# 8. PRICE ELASTICITY & OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("8. PRICE ELASTICITY & OPTIMIZATION")
print("="*80)

# Simulate historical price/volume data
np.random.seed(42)
historical = pd.DataFrame({
    'year': range(2019, 2025),
    'average_premium': [950, 980, 1010, 1050, 1100, 1150],
    'policy_count': [12000, 11800, 11500, 11200, 10800, 10500]
})

print("\nHistorical Data:")
print(historical)

# Estimate elasticity
elasticity_result = af.estimate_demand_elasticity(
    historical,
    price_col='average_premium',
    volume_col='policy_count',
    method='log_log'
)

print(f"\nElasticity Analysis:")
print(f"  Elasticity: {elasticity_result['elasticity']:.3f}")
print(f"  R-squared: {elasticity_result['r_squared']:.3f}")
print(f"  P-value: {elasticity_result['p_value']:.4f}")
print(f"  Interpretation: {elasticity_result['interpretation']}")

# Elasticity curve
curve = af.compute_elasticity_curve(
    current_price=1150,
    current_volume=10500,
    elasticity=elasticity_result['elasticity'],
    price_range=(1000, 1300),
    n_points=30
)

print("\nElasticity Curve (sample points):")
print(curve[::5])  # Every 5th point

# Find optimal price
optimal = af.optimal_price(
    cost_per_unit=premium_df['pure_premium'].mean(),  # Average pure premium
    current_price=1150,
    current_volume=10500,
    elasticity=elasticity_result['elasticity']
)

print(f"\nOptimal Pricing:")
print(f"  Current Price: ${optimal['current_price']:.2f}")
print(f"  Optimal Price: ${optimal['optimal_price']:.2f}")
print(f"  Price Change: {optimal['price_change_pct']:+.1f}%")
print(f"  Expected Volume: {optimal['optimal_volume']:,.0f}")
print(f"  Volume Change: {optimal['volume_change_pct']:+.1f}%")
print(f"  Current Profit: ${optimal['current_profit']:,.0f}")
print(f"  Optimal Profit: ${optimal['optimal_profit']:,.0f}")
print(f"  Profit Increase: ${optimal['optimal_profit'] - optimal['current_profit']:,.0f}")


# ============================================================================
# 9. EXPORT & SUMMARY
# ============================================================================

print("\n" + "="*80)
print("9. SUMMARY")
print("="*80)

print("\nModel Performance:")
print(f"  Frequency AIC: {freq_model.diagnostics_['aic']:.2f}")
print(f"  Severity AIC: {sev_model.diagnostics_['aic']:.2f}")
print(f"  Frequency Gini: {gini:.3f}")

print("\nPortfolio Metrics:")
print(f"  Total Policies: {len(policies):,}")
print(f"  Total Claims: {len(claims):,}")
print(f"  Overall Frequency: {len(claims) / policies['exposure'].sum():.4f}")
print(f"  Average Severity: ${claims['amount'].mean():,.2f}")
print(f"  Pure Premium: ${premium_df['pure_premium'].sum():,.2f}")
print(f"  Loaded Premium: ${premium_df['loaded_premium'].sum():,.2f}")
print(f"  Loss Ratio (target): {premium_df['pure_premium'].sum() / premium_df['loaded_premium'].sum():.1%}")

print("\nFactor Ranges:")
print(f"  Frequency: {rel_freq['Relativity'].min():.3f} - {rel_freq['Relativity'].max():.3f}")
print(f"  Severity: {rel_sev['Relativity'].min():.3f} - {rel_sev['Relativity'].max():.3f}")
print(f"  Combined: {factor_table['Combined_Relativity'].min():.3f} - {factor_table['Combined_Relativity'].max():.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Export results
print("\nExporting results...")
factor_table.to_csv('motor_rating_factors.csv', index=False)
premium_df.to_csv('motor_premium_results.csv', index=False)
print("  ✓ Factor table saved: motor_rating_factors.csv")
print("  ✓ Premium results saved: motor_premium_results.csv")

print("\nActuaFlow demonstration complete!")