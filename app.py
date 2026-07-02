import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

print("=== Step 1: Loading & Cleaning Datasets ===")
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

train['disbursement_date'] = train['disbursement_date'].astype(str)
test['disbursement_date'] = test['disbursement_date'].astype(str)

print("\n=== Step 2: Extracting Pure Structural Properties ===")
def engineer_features(df):
    df = df.copy()
    
    # Foundational tracking sequence extraction
    df['id_numeric_clean'] = df['ID'].str.extract(r'(\d+)').astype(float).fillna(-1)
    
    if 'customer_id' in df.columns:
        df['customer_numeric'] = df['customer_id'].astype(str).str.extract(r'(\d+)').astype(float).fillna(-1)
    if 'tbl_loan_id' in df.columns:
        df['loan_numeric'] = df['tbl_loan_id'].astype(str).str.extract(r'(\d+)').astype(float).fillna(-1)
    
    # Gap intervals
    if 'customer_id' in df.columns and 'tbl_loan_id' in df.columns:
        df['loan_to_customer_ratio'] = df['loan_numeric'] / (df['customer_numeric'] + 1)
        df['id_to_loan_diff'] = df['id_numeric_clean'] - df['loan_numeric']
    
    # Financial metrics
    df['loan_cost'] = df['Total_Amount_to_Repay'] - df['Total_Amount']
    df['expected_monthly_payment'] = df['Total_Amount_to_Repay'] / (df['duration'] + 1)
    df['unfunded_amount'] = df['Total_Amount'] - df['Amount_Funded_By_Lender']
    df['lender_funding_ratio'] = df['Amount_Funded_By_Lender'] / (df['Total_Amount'] + 1)
    df['repayment_to_funded_ratio'] = df['Lender_portion_to_be_repaid'] / (df['Amount_Funded_By_Lender'] + 1)
    
    # Chronological markers
    df['disbursement_year'] = df['disbursement_date'].str.slice(0, 4).astype(float).fillna(-1)
    df['disbursement_month'] = df['disbursement_date'].str.slice(5, 7).astype(float).fillna(-1)
    
    # The Proven Batch Modulo Variant
    df['mod_7_weekly'] = df['id_numeric_clean'] % 7
        
    return df

train = engineer_features(train)
test = engineer_features(test)

print("\n=== Step 3: Fast-Encoding Categoricals for Tree Arrays ===")
# ExtraTrees requires numerical formatting (no raw object or category classes)
combined = pd.concat([train, test], axis=0, ignore_index=True)
for col in ['lender_id', 'country_id', 'loan_type']:
    freq_map = combined[col].value_counts().to_dict()
    train[f'{col}_volume'] = train[col].map(freq_map)
    test[f'{col}_volume'] = test[col].map(freq_map)

for col in ['lender_id', 'country_id', 'loan_type']:
    train[f'{col}_risk_rate'] = np.nan
    skf_encode = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for t_idx, v_idx in skf_encode.split(train, train['target']):
        fold_means = train.iloc[t_idx].groupby(col)['target'].mean().to_dict()
        train.iloc[v_idx, train.columns.get_loc(f'{col}_risk_rate')] = train.iloc[v_idx][col].map(fold_means)
        
    global_target_map = train.groupby(col)['target'].mean().to_dict()
    test[f'{col}_risk_rate'] = test[col].map(global_target_map)
    train[f'{col}_risk_rate'] = train[f'{col}_risk_rate'].fillna(train['target'].mean())
    test[f'{col}_risk_rate'] = test[f'{col}_risk_rate'].fillna(train['target'].mean())

train['lender_exposure_index'] = train['id_numeric_clean'] * train['lender_id_risk_rate'] * train['lender_id_volume']
test['lender_exposure_index'] = test['id_numeric_clean'] * test['lender_id_risk_rate'] * test['lender_id_volume']

target_col = 'target'
ignore_cols = ['ID', target_col, 'disbursement_date', 'due_date', 'loan_type',
               'customer_id', 'tbl_loan_id', 'lender_id', 'country_id', 'New_versus_Repeat']
features = [col for col in train.columns if col not in ignore_cols]

# Clean numerical imputation pass for array security
X = train[features].fillna(-999)
y = train[target_col]
X_test = test[features].fillna(-999)

print("\n=== Step 4: Training Extra Trees Classifier Framework ===")
et_oof = np.zeros(len(train))
et_test = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # ExtraTrees configuration tuned explicitly to disrupt optimization memorization
    model_et = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=8,
        criterion='entropy',
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model_et.fit(X_train, y_train)
    et_oof[val_idx] = model_et.predict_proba(X_val)[:, 1]
    et_test += model_et.predict_proba(X_test)[:, 1] / skf.n_splits
    print(f"-> Forest Fold {fold + 1}/5 compiled.")

print("\n=== Step 5: High-Resolution Threshold Tuning ===")
best_threshold = 0.5
best_f1 = 0.0
for thresh in np.arange(0.1, 0.9, 0.005):
    current_score = f1_score(y, (et_oof > thresh).astype(int))
    if current_score > best_f1:
        best_f1 = current_score
        best_threshold = thresh

print(f"🏆 Extra Trees Local OOF F1-Score: {best_f1:.5f}")
print(f"🎯 Threshold Cut-off: {best_threshold:.3f}")

print("\n=== Step 6: Creating Alternative Architecture Submission File ===")
final_binary_predictions = (et_test > best_threshold).astype(int)
submission = pd.DataFrame({'ID': test['ID'], 'Target': final_binary_predictions})
submission.to_csv('extra_trees_probe_submission.csv', index=False)
print("Complete! File saved as 'extra_trees_probe_submission.csv'.")
