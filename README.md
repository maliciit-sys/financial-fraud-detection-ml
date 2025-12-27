# Financial Fraud Detection using Deep Learning

## Project Overview
Neural network-based fraud detection system trained on 5M+ financial transactions.

## Key Results
| Metric | Value |
|--------|-------|
| F1 Score | 0.083 |
| Recall | 83% |
| Precision | 4.3% |
| ROC-AUC | ~0.75 |

## Project Journey

### Initial Model Issues
- Model collapsed to constant predictions (0.4095)
- Fake 100% accuracy due to data leakage

### Root Causes Identified
1. `fraud_type` column perfectly correlated with target
2. Feature engineering computed on full dataset (data leakage)
3. High-cardinality label encoding for account IDs
4. Misconfigured Focal Loss (alpha=0.25)

### Fixes Applied
- Removed leaky `fraud_type` column
- Split data BEFORE feature engineering
- Replaced label encoding with frequency encoding
- Corrected Focal Loss alpha to 0.5
- Added proper output bias initialization

## Repository Structure
```
├── Fraud_Detection_Fixed.ipynb  # Complete working notebook
├── docs/
│   └── ROOT_CAUSE_ANALYSIS.md   # Detailed debugging journey
├── models/
│   ├── best_model_fixed.pth     # Trained model weights
│   ├── scaler_fixed.pkl         # Feature scaler
│   └── label_encoders_fixed.pkl # Category encoders
└── outputs/
    ├── eda_overview_fixed.png
    ├── training_curves_fixed.png
    └── evaluation_results_fixed.png
```

## Technologies
- Python 3.10
- PyTorch
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn

## Lessons Learned
See [Root Cause Analysis](docs/ROOT_CAUSE_ANALYSIS.md) for detailed debugging documentation.
