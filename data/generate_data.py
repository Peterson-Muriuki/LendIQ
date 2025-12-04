# data/generate_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import os

# Initialize
np.random.seed(42)
fake = Faker()
Faker.seed(42)

print("=" * 70)
print("LENDIQ DATA GENERATOR")
print("=" * 70)

# Configuration
n_samples = 10000
print(f"\nGenerating {n_samples:,} synthetic loan applications...")

# Generate applicant data
data = pd.DataFrame({
    # Personal Info
    'application_id': range(1, n_samples + 1),
    'applicant_name': [fake.name() for _ in range(n_samples)],
    'email': [fake.email() for _ in range(n_samples)],
    'phone': [fake.phone_number() for _ in range(n_samples)],
    'state': [fake.state_abbr() for _ in range(n_samples)],
    'zip_code': [fake.zipcode() for _ in range(n_samples)],
    
    # Credit Bureau Data
    'credit_score': np.random.normal(700, 60, n_samples).clip(300, 850).astype(int),
    'num_credit_accounts': np.random.poisson(5, n_samples),
    'credit_history_months': np.random.exponential(60, n_samples).clip(0, 360).astype(int),
    'delinquencies_2y': np.random.poisson(0.3, n_samples),
    'public_records': np.random.poisson(0.1, n_samples),
    'inquiries_6m': np.random.poisson(1, n_samples),
    
    # Financial Data
    'income': np.random.lognormal(11, 0.5, n_samples).clip(20000, 300000).round(2),
    'total_debt': np.random.lognormal(10, 1, n_samples).clip(0, 200000).round(2),
    'monthly_debt_payment': np.random.uniform(200, 3000, n_samples).round(2),
    
    # Credit Utilization
    'total_credit_limit': np.random.uniform(5000, 100000, n_samples).round(2),
    'total_credit_balance': np.random.uniform(0, 50000, n_samples).round(2),
    
    # Employment
    'employment_status': np.random.choice(
        ['Employed', 'Self-Employed', 'Unemployed'], 
        n_samples, 
        p=[0.8, 0.15, 0.05]
    ),
    'employment_length_months': np.random.exponential(36, n_samples).clip(0, 300).astype(int),
    'occupation': [fake.job() for _ in range(n_samples)],
    
    # Loan Request
    'loan_amount': np.random.uniform(5000, 50000, n_samples).round(2),
    'loan_purpose': np.random.choice(
        ['debt_consolidation', 'home_improvement', 'business', 'auto', 'other'],
        n_samples,
        p=[0.4, 0.2, 0.15, 0.15, 0.1]
    ),
    'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
    
    # Banking Data (Alternative Data)
    'avg_monthly_balance': np.random.uniform(500, 20000, n_samples).round(2),
    'num_bank_accounts': np.random.poisson(2, n_samples),
    'has_savings_account': np.random.choice([True, False], n_samples, p=[0.7, 0.3]),
    'monthly_income_deposits': np.random.uniform(2000, 15000, n_samples).round(2),
    'num_nsf_fees': np.random.poisson(0.2, n_samples),
    
    # Application Metadata
    'application_date': [
        datetime.now() - timedelta(days=int(x)) 
        for x in np.random.uniform(0, 180, n_samples)
    ],
    'referral_source': np.random.choice(
        ['organic', 'paid_search', 'affiliate', 'direct'],
        n_samples,
        p=[0.3, 0.3, 0.2, 0.2]
    )
})

print(f"✓ Generated {len(data):,} applications with {len(data.columns)} features")

# Calculate default probability
def calculate_default_probability(row):
    """Calculate PD based on risk factors"""
    base_pd = 0.10
    
    # Credit score impact
    if row['credit_score'] >= 800:
        credit_factor = 0.3
    elif row['credit_score'] >= 740:
        credit_factor = 0.5
    elif row['credit_score'] >= 670:
        credit_factor = 1.0
    elif row['credit_score'] >= 620:
        credit_factor = 2.0
    else:
        credit_factor = 3.0
    
    # DTI impact
    dti = (row['total_debt'] + row['loan_amount']) / row['income']
    dti_factor = 1 + dti
    
    # Delinquency impact
    delinq_factor = 1 + (row['delinquencies_2y'] * 0.5)
    
    # Employment impact
    if row['employment_length_months'] < 6:
        employment_factor = 1.5
    elif row['employment_length_months'] < 12:
        employment_factor = 1.2
    else:
        employment_factor = 1.0
    
    pd = base_pd * credit_factor * dti_factor * delinq_factor * employment_factor
    pd = pd * np.random.uniform(0.8, 1.2)  # Add randomness
    
    return min(pd, 0.8)

print("\nCalculating probability of default for each application...")
data['probability_of_default'] = data.apply(calculate_default_probability, axis=1)

# Create binary default indicator
data['default'] = (np.random.random(len(data)) < data['probability_of_default']).astype(int)

# Calculate derived features
data['debt_to_income'] = (data['total_debt'] + data['loan_amount']) / data['income']
data['credit_utilization'] = data['total_credit_balance'] / data['total_credit_limit'].replace(0, 1)
data['payment_to_income'] = (data['monthly_debt_payment'] * 12) / data['income']

print(f"✓ Default rate: {data['default'].mean():.2%}")
print(f"✓ Average PD: {data['probability_of_default'].mean():.2%}")

# Save to CSV
output_file = 'data/loan_applications.csv'
data.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")
print(f"✓ File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

# Display summary statistics
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"\nTotal Applications: {len(data):,}")
print(f"Features: {len(data.columns)}")
print(f"Date Range: {data['application_date'].min().date()} to {data['application_date'].max().date()}")
print(f"Default Rate: {data['default'].mean():.2%}")

print("\n" + "=" * 70)
print("CREDIT SCORE DISTRIBUTION")
print("=" * 70)
print(data['credit_score'].describe())

print("\n" + "=" * 70)
print("LOAN AMOUNT DISTRIBUTION")
print("=" * 70)
print(data['loan_amount'].describe())

print("\n✅ Data generation complete!")