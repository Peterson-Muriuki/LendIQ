# notebooks/01_data_exploration.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 70)
print("LENDIQ DATA EXPLORATION")
print("=" * 70)

# Load data
data = pd.read_csv('data/loan_applications.csv')
print(f"\n✓ Loaded {len(data):,} records")

# Basic info
print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(data.info())

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('LendIQ: Data Exploration', fontsize=16, fontweight='bold')

# 1. Credit Score Distribution
axes[0, 0].hist(data['credit_score'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Credit Score Distribution')
axes[0, 0].set_xlabel('Credit Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(data['credit_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {data["credit_score"].mean():.0f}')
axes[0, 0].legend()

# 2. Loan Amount Distribution
axes[0, 1].hist(data['loan_amount'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('Loan Amount Distribution')
axes[0, 1].set_xlabel('Loan Amount ($)')
axes[0, 1].set_ylabel('Frequency')

# 3. Default Rate by Credit Score
credit_bins = pd.cut(data['credit_score'], bins=10)
default_by_credit = data.groupby(credit_bins)['default'].mean()
axes[0, 2].plot(range(len(default_by_credit)), default_by_credit.values, 
               marker='o', linewidth=2, markersize=8, color='red')
axes[0, 2].set_title('Default Rate by Credit Score Bin')
axes[0, 2].set_xlabel('Credit Score Bin')
axes[0, 2].set_ylabel('Default Rate')
axes[0, 2].grid(alpha=0.3)

# 4. DTI Distribution
axes[1, 0].hist(data['debt_to_income'].clip(0, 1), bins=30, 
               edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_title('Debt-to-Income Ratio Distribution')
axes[1, 0].set_xlabel('DTI Ratio')
axes[1, 0].set_ylabel('Frequency')

# 5. Default vs Non-Default
default_counts = data['default'].value_counts()
axes[1, 1].bar(['No Default', 'Default'], default_counts.values, 
              color=['green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Default Distribution')
axes[1, 1].set_ylabel('Count')
for i, v in enumerate(default_counts.values):
    axes[1, 1].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

# 6. Loan Purpose Distribution
purpose_counts = data['loan_purpose'].value_counts()
axes[1, 2].barh(purpose_counts.index, purpose_counts.values, 
               color='purple', alpha=0.7, edgecolor='black')
axes[1, 2].set_title('Loan Purpose Distribution')
axes[1, 2].set_xlabel('Count')

plt.tight_layout()
plt.savefig('figures/data_exploration.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: figures/data_exploration.png")
plt.close()

# Correlation heatmap
numeric_cols = data.select_dtypes(include=[np.number]).columns[:15]  # First 15 numeric columns
correlation_matrix = data[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, square=True, linewidths=1)
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: figures/correlation_matrix.png")
plt.close()

print("\n✅ Data exploration complete!")
print(f"\nCheck the 'figures' folder for visualizations")