# Dataset Reference Guide

## Dataset Naming Convention

| Short Name | Full Name | Type |
|------------|-----------|------|
| **PCA Primary** | Credit Card Fraud (Kaggle ULB) | PCA-transformed |
| **PCA Secondary** | Bank Account Fraud (NeurIPS 2022) | Normalized/Encoded |
| **Non-PCA Primary** | Sparkov Credit Card Transactions | Raw/Interpretable |
| **Non-PCA Secondary** | PaySim Mobile Money | Raw/Interpretable |

---

## Non-PCA Primary: Sparkov Credit Card Transactions

- **Source:** https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Size:** 1,852,394 transactions
- **Fraud Rate:** 0.52%
- **Period:** Jan 2019 - Dec 2020
- **Customers:** 1,000
- **Merchants:** 800

### Features (23 columns - All Interpretable)

| Feature | Type | Description |
|---------|------|-------------|
| `trans_date_trans_time` | datetime | Transaction timestamp |
| `cc_num` | int | Credit card number |
| `merchant` | string | Merchant name |
| `category` | string | Transaction category |
| `amt` | float | Transaction amount |
| `first` | string | Customer first name |
| `last` | string | Customer last name |
| `gender` | string | M/F |
| `street` | string | Street address |
| `city` | string | City |
| `state` | string | State |
| `zip` | int | ZIP code |
| `lat` | float | Customer latitude |
| `long` | float | Customer longitude |
| `city_pop` | int | City population |
| `job` | string | Customer occupation |
| `dob` | date | Date of birth |
| `trans_num` | string | Transaction ID |
| `unix_time` | int | Unix timestamp |
| `merch_lat` | float | Merchant latitude |
| `merch_long` | float | Merchant longitude |
| `is_fraud` | int | Target (0/1) |

---

## Non-PCA Secondary: PaySim Mobile Money

- **Source:** https://www.kaggle.com/datasets/ealaxi/paysim1
- **Size:** 6,362,620 transactions
- **Fraud Rate:** 0.13%
- **Simulation:** 1 month of mobile money transactions

### Features (11 columns - All Interpretable)

| Feature | Type | Description |
|---------|------|-------------|
| `step` | int | Time step (1 hour = 1 step) |
| `type` | string | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | float | Transaction amount |
| `nameOrig` | string | Origin account ID |
| `oldbalanceOrg` | float | Balance before (origin) |
| `newbalanceOrig` | float | Balance after (origin) |
| `nameDest` | string | Destination account ID |
| `oldbalanceDest` | float | Balance before (destination) |
| `newbalanceDest` | float | Balance after (destination) |
| `isFraud` | int | Target (0/1) |
| `isFlaggedFraud` | int | System flag for large transfers |

---

## PCA Primary: Credit Card Fraud (Kaggle ULB)

- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** 284,807 transactions
- **Fraud Rate:** 0.17%
- **Features:** 30 (V1-V28 PCA + Time + Amount)
- **Limitation:** Features are anonymized/PCA-transformed

---

## PCA Secondary: Bank Account Fraud (NeurIPS 2022)

- **Source:** https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
- **Size:** 1,000,000 records
- **Fraud Rate:** ~1.1%
- **Features:** 30+ (pre-encoded/normalized)
- **Limitation:** Features are pre-processed

---

## Why Non-PCA Datasets?

| Aspect | PCA Datasets | Non-PCA Datasets |
|--------|--------------|------------------|
| Interpretability | ❌ Low (V1, V2...) | ✅ High (merchant, amount, location) |
| Feature Engineering | ❌ Limited | ✅ Full flexibility |
| Research Value | ⚠️ Limited insights | ✅ Explainable results |
| Real-world Application | ⚠️ Abstract | ✅ Practical |
