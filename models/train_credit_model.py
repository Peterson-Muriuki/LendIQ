# models/train_credit_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
import os

print("=" * 70)
print("LENDIQ CREDIT SCORING MODEL TRAINING")
print("=" * 70)

# Load data
print("\nLoading data...")
data = pd.read_csv('data/loan_applications.csv')
print(f"✓ Loaded {len(data):,} records")

# Feature Engineering
def engineer_features(df):
    """Create advanced features"""
    df = df.copy()
    print("\nEngineering features...")
    
    # Basic ratios
    df['dti_ratio'] = (df['total_debt'] + df['loan_amount']) / df['income']
    df['payment_to_income'] = (df['monthly_debt_payment'] * 12) / df['income']
    df['credit_utilization'] = df['total_credit_balance'] / df['total_credit_limit'].replace(0, 1)
    df['loan_to_income'] = df['loan_amount'] / df['income']
    
    # Credit metrics
    df['credit_score_to_income'] = df['credit_score'] / (df['income'] / 1000)
    df['available_credit'] = df['total_credit_limit'] - df['total_credit_balance']
    df['available_credit_ratio'] = df['available_credit'] / df['total_credit_limit'].replace(0, 1)
    
    # Account management
    df['avg_credit_per_account'] = df['total_credit_limit'] / df['num_credit_accounts'].replace(0, 1)
    df['credit_age_years'] = df['credit_history_months'] / 12
    df['is_new_credit'] = (df['credit_history_months'] < 24).astype(int)
    
    # Delinquency features
    df['has_delinquency'] = (df['delinquencies_2y'] > 0).astype(int)
    df['has_public_record'] = (df['public_records'] > 0).astype(int)
    df['delinquency_severity'] = np.log1p(df['delinquencies_2y'])
    
    # Employment
    df['employment_years'] = df['employment_length_months'] / 12
    df['is_new_employee'] = (df['employment_length_months'] < 12).astype(int)
    df['employment_stability_score'] = np.minimum(df['employment_length_months'] / 60, 1)
    
    # Banking behavior
    df['savings_rate'] = df['avg_monthly_balance'] / (df['income'] / 12)
    df['income_volatility'] = np.abs(df['monthly_income_deposits'] - df['income'] / 12) / (df['income'] / 12)
    df['has_banking_issues'] = (df['num_nsf_fees'] > 0).astype(int)
    
    # Loan characteristics
    df['loan_term_years'] = df['loan_term_months'] / 12
    df['estimated_monthly_payment'] = df['loan_amount'] / df['loan_term_months']
    df['payment_burden'] = df['estimated_monthly_payment'] / (df['income'] / 12)
    
    # Employment status
    df['is_employed'] = (df['employment_status'] == 'Employed').astype(int)
    df['is_self_employed'] = (df['employment_status'] == 'Self-Employed').astype(int)
    
    # Loan purpose dummies
    loan_purpose_dummies = pd.get_dummies(df['loan_purpose'], prefix='purpose')
    df = pd.concat([df, loan_purpose_dummies], axis=1)
    
    print(f"✓ Created {len(df.columns)} total features")
    return df

# Engineer features
data_engineered = engineer_features(data)

# Select features for modeling
feature_columns = [
    # Credit bureau
    'credit_score', 'num_credit_accounts', 'credit_history_months',
    'delinquencies_2y', 'public_records', 'inquiries_6m',
    
    # Financial
    'income', 'total_debt', 'monthly_debt_payment',
    
    # Ratios
    'dti_ratio', 'payment_to_income', 'credit_utilization',
    'loan_to_income', 'credit_score_to_income',
    
    # Engineered
    'available_credit_ratio', 'credit_age_years', 'is_new_credit',
    'has_delinquency', 'has_public_record', 'delinquency_severity',
    'employment_years', 'is_new_employee', 'employment_stability_score',
    'savings_rate', 'income_volatility', 'has_banking_issues',
    'loan_term_years', 'payment_burden',
    'is_employed', 'is_self_employed',
    
    # Loan purpose
    'purpose_debt_consolidation', 'purpose_home_improvement',
    'purpose_business', 'purpose_auto', 'purpose_other'
]

X = data_engineered[feature_columns].fillna(0)
y = data_engineered['default']

print(f"\n✓ Features: {X.shape[1]}")
print(f"✓ Samples: {X.shape[0]:,}")
print(f"✓ Default rate: {y.mean():.2%}")

# Train-test split
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training:   {len(X_train):,} samples ({y_train.mean():.2%} default)")
print(f"Validation: {len(X_val):,} samples ({y_val.mean():.2%} default)")
print(f"Test:       {len(X_test):,} samples ({y_test.mean():.2%} default)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
print("\nApplying SMOTE for class balance...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {len(y_train_balanced):,} samples ({y_train_balanced.mean():.2%} default)")

# Train models
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

models = {}
results = {}

# 1. Logistic Regression
print("\n1. Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_balanced, y_train_balanced)
y_pred_lr = lr.predict_proba(X_val_scaled)[:, 1]
auc_lr = roc_auc_score(y_val, y_pred_lr)
print(f"   Validation AUC: {auc_lr:.4f}")
models['Logistic Regression'] = lr
results['Logistic Regression'] = auc_lr

# 2. Random Forest
print("\n2. Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=50,
    min_samples_leaf=20, random_state=42, n_jobs=-1
)
rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf.predict_proba(X_val_scaled)[:, 1]
auc_rf = roc_auc_score(y_val, y_pred_rf)
print(f"   Validation AUC: {auc_rf:.4f}")
models['Random Forest'] = rf
results['Random Forest'] = auc_rf

# 3. XGBoost
print("\n3. XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3,
    random_state=42, eval_metric='auc'
)
xgb_model.fit(X_train_balanced, y_train_balanced, verbose=False)
y_pred_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
auc_xgb = roc_auc_score(y_val, y_pred_xgb)
print(f"   Validation AUC: {auc_xgb:.4f}")
models['XGBoost'] = xgb_model
results['XGBoost'] = auc_xgb

# 4. LightGBM
print("\n4. LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', random_state=42, verbose=-1
)
lgb_model.fit(X_train_balanced, y_train_balanced)
y_pred_lgb = lgb_model.predict_proba(X_val_scaled)[:, 1]
auc_lgb = roc_auc_score(y_val, y_pred_lgb)
print(f"   Validation AUC: {auc_lgb:.4f}")
models['LightGBM'] = lgb_model
results['LightGBM'] = auc_lgb

# Model comparison
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
for model_name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:20s}: {auc:.4f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")

# Evaluate on test set
print("\n" + "=" * 70)
print(f"TEST SET EVALUATION - {best_model_name}")
print("=" * 70)

y_pred_test = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred_test_binary = (y_pred_test >= 0.5).astype(int)
test_auc = roc_auc_score(y_test, y_pred_test)

print(f"\nAUC-ROC: {test_auc:.4f}")

cm = confusion_matrix(y_test, y_pred_test_binary)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"\nFalse Positive Rate: {fp / (fp + tn):.2%}")
print(f"False Negative Rate: {fn / (fn + tp):.2%}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'{best_model_name} (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ ROC curve saved")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance plot saved")

# Save model
os.makedirs('models', exist_ok=True)
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'model_name': best_model_name,
    'test_auc': test_auc,
    'feature_importance': feature_importance if hasattr(best_model, 'feature_importances_') else None
}

joblib.dump(model_artifacts, 'models/credit_scoring_model.pkl')
print("\n✓ Model saved to: models/credit_scoring_model.pkl")

print("\n✅ Model training complete!")
print(f"\n📊 Final Results:")
print(f"   Model: {best_model_name}")
print(f"   Test AUC: {test_auc:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")