# models/portfolio_optimizer.py

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("LENDIQ PORTFOLIO OPTIMIZATION")
print("=" * 70)

# Load priced portfolio
portfolio = pd.read_csv('data/priced_portfolio.csv')
print(f"\n✓ Loaded {len(portfolio):,} loans")

def mean_variance_optimization(loans_df, max_weight=0.05, target_return=None):
    """Markowitz mean-variance optimization"""
    n = len(loans_df)
    
    expected_returns = loans_df['expected_return'].values
    risks = loans_df['probability_of_default'].values
    
    # Covariance matrix
    base_volatility = np.diag(risks ** 2)
    correlation = 0.3
    cov_matrix = base_volatility + correlation * np.outer(risks, risks)
    
    # Optimization
    weights = cp.Variable(n)
    portfolio_return = expected_returns @ weights
    portfolio_risk = cp.quad_form(weights, cov_matrix)
    
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight
    ]
    
    if target_return:
        constraints.append(portfolio_return >= target_return)
        objective = cp.Minimize(portfolio_risk)
    else:
        objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status == 'optimal':
        return weights.value
    else:
        return np.ones(n) / n

# Optimize
print("\nRunning mean-variance optimization...")
optimal_weights = mean_variance_optimization(portfolio, max_weight=0.05)
portfolio['optimal_weight'] = optimal_weights

# Calculate metrics
portfolio_return = (portfolio['expected_return'] * portfolio['optimal_weight']).sum()
portfolio_pd = (portfolio['probability_of_default'] * portfolio['optimal_weight']).sum()
portfolio_exposure = (portfolio['loan_amount'] * portfolio['optimal_weight']).sum()
expected_loss = (portfolio['probability_of_default'] * 0.40 * 
                 portfolio['loan_amount'] * portfolio['optimal_weight']).sum()

print(f"\n✓ Optimization complete")
print(f"\n" + "=" * 70)
print("OPTIMIZED PORTFOLIO METRICS")
print("=" * 70)
print(f"Expected Return: {portfolio_return:.2%}")
print(f"Portfolio PD: {portfolio_pd:.2%}")
print(f"Total Exposure: ${portfolio_exposure:,.2f}")
print(f"Expected Loss: ${expected_loss:,.2f}")

# Top holdings
print(f"\nTop 10 Holdings:")
top_holdings = portfolio.nlargest(10, 'optimal_weight')[
    ['application_id', 'credit_score', 'loan_amount', 'expected_return', 'optimal_weight']
]
print(top_holdings.to_string(index=False))

# Visualizations
os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Portfolio Optimization Results', fontsize=16, fontweight='bold')

# 1. Allocation by risk segment
segment_allocation = portfolio.groupby('risk_segment')['optimal_weight'].sum()
axes[0, 0].pie(segment_allocation.values, labels=segment_allocation.index, 
              autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Allocation by Risk Segment')

# 2. Risk-Return scatter
sample = portfolio.sample(min(200, len(portfolio)))
scatter = axes[0, 1].scatter(
    sample['probability_of_default'] * 100,
    sample['expected_return'] * 100,
    s=sample['optimal_weight'] * 10000,
    c=sample['credit_score'],
    cmap='RdYlGn',
    alpha=0.6,
    edgecolors='black'
)
axes[0, 1].set_xlabel('Probability of Default (%)')
axes[0, 1].set_ylabel('Expected Return (%)')
axes[0, 1].set_title('Risk-Return Profile')
axes[0, 1].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 1], label='Credit Score')

# 3. Weight distribution
axes[1, 0].hist(portfolio['optimal_weight'] * 100, bins=30, 
               edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].set_xlabel('Weight (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Portfolio Weights')
axes[1, 0].axvline(5, color='red', linestyle='--', label='Max Weight (5%)')
axes[1, 0].legend()

# 4. Concentration curve
sorted_weights = portfolio.sort_values('optimal_weight', ascending=False)['optimal_weight']
cumulative = np.cumsum(sorted_weights) * 100
axes[1, 1].plot(range(len(cumulative)), cumulative, linewidth=2)
axes[1, 1].axhline(y=50, color='r', linestyle='--', label='50% of portfolio')
axes[1, 1].axhline(y=80, color='orange', linestyle='--', label='80% of portfolio')
axes[1, 1].set_xlabel('Number of Loans')
axes[1, 1].set_ylabel('Cumulative Allocation (%)')
axes[1, 1].set_title('Concentration Curve')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/portfolio_optimization.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Visualization saved: figures/portfolio_optimization.png")

# Stress testing
print("\n" + "=" * 70)
print("STRESS TESTING")
print("=" * 70)

scenarios = {
    'Base Case': {'pd_mult': 1.0, 'lgd_add': 0.0},
    'Mild Recession': {'pd_mult': 1.5, 'lgd_add': 0.05},
    'Severe Recession': {'pd_mult': 2.0, 'lgd_add': 0.10},
    'Credit Crisis': {'pd_mult': 2.5, 'lgd_add': 0.15}
}

stress_results = []

for scenario_name, params in scenarios.items():
    stressed_pd = portfolio['probability_of_default'] * params['pd_mult']
    stressed_lgd = 0.40 + params['lgd_add']
    
    expected_loss = (stressed_pd * stressed_lgd * 
                     portfolio['loan_amount'] * portfolio['optimal_weight']).sum()
    loss_rate = expected_loss / portfolio_exposure
    
    stressed_return = portfolio['interest_rate'] - (stressed_pd * stressed_lgd) - 0.02
    portfolio_return_stressed = (stressed_return * portfolio['optimal_weight']).sum()
    
    stress_results.append({
        'Scenario': scenario_name,
        'Expected Loss ($M)': expected_loss / 1e6,
        'Loss Rate (%)': loss_rate * 100,
        'Portfolio Return (%)': portfolio_return_stressed * 100
    })

stress_df = pd.DataFrame(stress_results)
print(stress_df.to_string(index=False))

# Save optimized portfolio
portfolio.to_csv('data/optimized_portfolio.csv', index=False)
print(f"\n✓ Saved optimized portfolio to: data/optimized_portfolio.csv")

print("\n✅ Portfolio optimization complete!")