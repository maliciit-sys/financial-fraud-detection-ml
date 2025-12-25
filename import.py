import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('/media/maliciit/Data/Ali/MS Data Scince/Semester 1/4. Machine Learning/AI_ML Project/financial_fraud_detection_dataset.csv')  # Replace with your file path

print(f"Initial shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].sum() / len(df) * 100:.2f}%")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ===== 1. HANDLE MISSING VALUES =====
df['fraud_type'].fillna('Unknown', inplace=True)
df['time_since_last_transaction'].fillna(df['time_since_last_transaction'].median(), inplace=True)

# ===== 2. DROP UNNECESSARY COLUMNS =====
df.drop(['transaction_id', 'ip_address', 'device_hash', 'timestamp'], axis=1, inplace=True)

# ===== 3. ENCODE CATEGORICAL FEATURES =====
# In your preprocessing code, change this line:
categorical_cols = ['transaction_type', 'merchant_category', 'location',
                    'device_used', 'payment_channel', 'fraud_type',
                    'sender_account', 'receiver_account']  # Added these 2

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ===== 4. SEPARATE FEATURES AND TARGET =====
X = df.drop(['is_fraud'], axis=1)
y = df['is_fraud']

# ===== 5. TRAIN/VAL/TEST SPLIT =====
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train fraud rate: {y_train.sum() / len(y_train) * 100:.2f}%")

# ===== 6. NORMALIZE NUMERIC FEATURES =====
numeric_cols = ['amount', 'time_since_last_transaction', 'spending_deviation_score',
                'velocity_score', 'geo_anomaly_score']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ===== 7. SAVE PREPROCESSED DATA =====
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Save scaler and encoders
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("\n✓ Preprocessing complete! Files saved.")
print(f"Feature count: {X_train.shape[1]}")
