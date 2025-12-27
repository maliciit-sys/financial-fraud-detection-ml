# Root Cause Analysis: Model Collapse Issue

## Problem Statement
The initial fraud detection model collapsed to constant predictions (~0.4095 probability for all samples), resulting in zero fraud detection capability.

## Issues Identified

### 1. High-Cardinality Label Encoding (Critical)
- **Problem**: `sender_account` and `receiver_account` had thousands of unique values
- **Impact**: Label encoding created false ordinal relationships
- **Fix**: Dropped ID columns, used frequency encoding instead

### 2. Data Leakage - Feature Engineering
- **Problem**: Computed `sender_fraud_rate` and `receiver_fraud_rate` on entire dataset before train/test split
- **Impact**: Test set information leaked into training features
- **Fix**: Split data FIRST, then compute aggregations only on training data

### 3. Data Leakage - fraud_type Column
- **Problem**: `fraud_type` had perfect correlation (1.0) with target `is_fraud`
- **Impact**: Model learned to use this single feature, achieving fake 100% accuracy
- **Fix**: Removed `fraud_type` from features

### 4. Focal Loss Misconfiguration
- **Problem**: `alpha=0.25` down-weighted the minority class (fraud)
- **Impact**: Model biased toward predicting non-fraud
- **Fix**: Changed to `alpha=0.5` for balanced weighting

### 5. Model Initialization
- **Problem**: Output bias initialized to 0
- **Impact**: Model started predicting ~50% probability
- **Fix**: Initialize output bias to `log(fraud_rate / (1 - fraud_rate))`

## Final Results After Fixes

| Metric | Value |
|--------|-------|
| Best F1 | 0.0830 |
| Recall | ~83% |
| Precision | ~4.3% |
| Training Epochs | 13 (early stopping) |

## Lessons Learned

1. Always check feature correlations with target before training
2. Split data before any feature engineering
3. Perfect metrics (100%) always indicate data leakage
4. Fraud detection with weak features is genuinely difficult
