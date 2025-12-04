# models/price_portfolio.py

import pandas as pd
import numpy as np
from pricing_engine import DynamicPricingEngine
import os

print("=" * 70)
print("LENDIQ PORTFOLIO PRICING")
print("=" * 70)

# Load data
data = pd.read_csv('data/loan_applications.csv')
approved_data = data[data['default'] == 0].copy()

# Sample for speed (use first 1000 approved loans)
approved_data = approved_data.head(1000)
print(f"\n✓ Loaded {len(approved_data):,} approved loans")

# Initialize pricing engine
engine = DynamicPricingEngine()

# Price each loan
print("\nPricing loans...")
prices = []

for idx, row in approved_data.iterrows():
    pricing = engine.price_loan(
        credit_score=row['credit_score'],
        loan_amount=row['loan_amount'],
        loan_term_years=row['loan_term_months'] / 12,
        dti=(row['total_debt'] + row['loan_amount']) / row['income'],
        delinquencies=row['delinquencies_2y'],
        employment_months=row['employment_length_months']
    )
    
    prices.append({
        'application_id': row['application_id'],
        'credit_score': row['credit_score'],
        'loan_amount': row['loan_amount'],
        'interest_rate': pricing['interest_rate'],
        'monthly_payment': pricing['monthly_payment'],
        'probability_of_default': pricing['probability_of_default'],
        'expected_loss': pricing['expected_loss_dollars'],
        'expected_return': pricing['interest_rate'] - (pricing['probability_of_default'] * 0.40) - 0.02
    })
    
    if (idx + 1) % 100 == 0:
        print(f"  Priced {idx + 1:,} loans...")

prices_df = pd.DataFrame(prices)

# Merge with original data
approved_data = approved_data.merge(prices_df[['application_id', 'interest_rate', 
                                               'monthly_payment', 'probability_of_default',
                                               'expected_loss', 'expected_return']], 
                                    on='application_id')

# Create risk segments
approved_data['risk_segment'] = pd.cut(
    approved_data['credit_score'],
    bins=[0, 620, 680, 740, 850],
    labels=['High Risk', 'Medium-High Risk', 'Medium-Low Risk', 'Low Risk']
)

# Save
os.makedirs('data', exist_ok=True)
approved_data.to_csv('data/priced_portfolio.csv', index=False)
print(f"\n✓ Saved priced portfolio to: data/priced_portfolio.csv")

# Summary statistics
print("\n" + "=" * 70)
print("PRICING SUMMARY")
print("=" * 70)
print(f"\nPortfolio Size: {len(approved_data):,} loans")
print(f"Total Loan Amount: ${approved_data['loan_amount'].sum():,.2f}")
print(f"\nAverage Interest Rate: {approved_data['interest_rate'].mean():.2%}")
print(f"Rate Range: {approved_data['interest_rate'].min():.2%} - {approved_data['interest_rate'].max():.2%}")
print(f"\nAverage Expected Return: {approved_data['expected_return'].mean():.2%}")
print(f"Average Default Probability: {approved_data['probability_of_default'].mean():.2%}")

print("\n✅ Portfolio pricing complete!")