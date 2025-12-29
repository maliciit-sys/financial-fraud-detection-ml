# Financial Fraud Detection using Deep Learning

## Project Overview

Neural network and XGBoost-based fraud detection system trained on multiple datasets to demonstrate model performance across different data types.

## Dataset Summary

| Dataset | Type | Size | Fraud Rate | Features |
|---------|------|------|------------|----------|
| **Non-PCA Primary** (Sparkov) | Interpretable | 1.85M | 0.52% | 23 raw features |
| **Non-PCA Secondary** (PaySim) | Interpretable | 6.3M | 0.13% | 11 raw features |
| PCA Primary (Credit Card) | Anonymized | 284K | 0.17% | 30 PCA features |
| PCA Secondary (Bank Account) | Encoded | 1M | 1.1% | 30+ encoded features |

## Repository Structure
```
scripts/
├── data/
│   ├── non_pca_primary/      # Sparkov dataset
│   ├── non_pca_secondary/    # PaySim dataset
│   ├── pca_primary/          # Credit Card Fraud (archived)
│   ├── pca_secondary/        # Bank Account Fraud (archived)
│   └── archive/              # Original failed dataset
│
├── notebooks/
│   ├── 01_non_pca_primary_eda.ipynb
│   ├── 02_non_pca_primary_nn.ipynb
│   ├── 03_non_pca_primary_xgboost.ipynb
│   ├── 04_non_pca_secondary_eda.ipynb
│   ├── 05_non_pca_secondary_nn.ipynb
│   ├── 06_non_pca_secondary_xgboost.ipynb
│   └── 07_non_pca_comparison.ipynb
│
├── models/
│   ├── non_pca_primary/      # Sparkov models
│   ├── non_pca_secondary/    # PaySim models
│   └── pca_*/                # Archived PCA models
│
├── outputs/
│   ├── non_pca_primary/      # EDA, training curves, results
│   ├── non_pca_secondary/
│   ├── non_pca_comparison/
│   └── pca_*/                # Archived PCA outputs
│
├── src/                      # Reusable Python modules
├── docs/                     # Documentation
└── archive/                  # Old work from original dataset
```

## Key Results

### Non-PCA Datasets (Current Focus)
*Coming soon...*

### PCA Datasets (Archived)

| Dataset | Model | ROC-AUC | F1 | Precision | Recall |
|---------|-------|---------|-----|-----------|--------|
| PCA Primary | Neural Network | 0.9356 | 0.8054 | 80.0% | 81.1% |
| PCA Primary | XGBoost | 0.9712 | 0.8633 | 92.3% | 81.1% |
| PCA Secondary | Neural Network | 0.8776 | 0.1712 | 10.2% | 52.4% |
| PCA Secondary | XGBoost | 0.8835 | 0.2282 | 23.0% | 22.7% |

## Lessons Learned

See [docs/ROOT_CAUSE_ANALYSIS.md](docs/ROOT_CAUSE_ANALYSIS.md) for detailed analysis of initial model failures and debugging journey.

See [docs/DATASET_NOTES.md](docs/DATASET_NOTES.md) for complete dataset documentation.

## Technologies

- Python 3.10+
- PyTorch
- XGBoost
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn

## Author

Muhammad Ali Tahir  
MS Data Science Program  
Superior University, Lahore
