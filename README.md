# Financial Fraud Detection - Multi-Dataset Comparison

## Project Overview
Comprehensive fraud detection system comparing Neural Networks vs XGBoost across two datasets.

## 🏆 Results Summary

| Dataset | Model | ROC-AUC | F1 Score | Precision | Recall |
|---------|-------|---------|----------|-----------|--------|
| Primary (Credit Card) | Neural Network | 0.9356 | 0.8054 | 80.0% | 81.1% |
| **Primary (Credit Card)** | **XGBoost** ✓ | **0.9712** | **0.8633** | **92.3%** | 81.1% |
| Secondary (Bank Account) | Neural Network | 0.8776 | 0.1712 | 10.2% | 52.4% |
| **Secondary (Bank Account)** | **XGBoost** ✓ | **0.8835** | **0.2282** | **23.0%** | 22.7% |

## Key Findings
1. **XGBoost outperforms Neural Networks** on both datasets
2. **XGBoost is 60-76x faster** in training time
3. **Primary dataset** has stronger fraud signals (higher performance)
4. **Secondary dataset** is more challenging (weaker feature correlations)
5. **Threshold optimization** is crucial for precision-recall balance

## Datasets

### Primary: Credit Card Fraud (Kaggle)
- 284,807 transactions
- 0.17% fraud rate
- 30 PCA-transformed features
- Source: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Secondary: Bank Account Fraud (NeurIPS 2022)
- 1,000,000 transactions
- 1.10% fraud rate
- 31 interpretable features
- Source: [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

## Project Structure
```
├── notebooks/
│   ├── 01_primary_eda.ipynb
│   ├── 02_primary_neural_network.ipynb
│   ├── 03_primary_xgboost.ipynb
│   ├── 04_secondary_eda.ipynb
│   ├── 05_secondary_neural_network.ipynb
│   ├── 06_secondary_xgboost.ipynb
│   └── 07_comparison.ipynb
├── models/
│   ├── primary/
│   │   ├── nn_best_model.pth
│   │   └── xgboost_model.json
│   └── secondary/
│       ├── nn_best_model.pth
│       └── xgboost_model.json
├── outputs/
│   ├── primary/
│   ├── secondary/
│   └── comparison/
├── archive/              # Previous dataset work
└── docs/
```

## Technologies
- Python 3.10
- PyTorch (Neural Networks)
- XGBoost (Gradient Boosting)
- CUDA (GPU Acceleration)
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn

## Lessons Learned
1. Data quality > Model complexity
2. Always check for data leakage
3. Feature correlations indicate predictability
4. XGBoost excels on tabular data
5. Threshold tuning is essential for imbalanced datasets

## Previous Work
See `archive/` folder for initial work on a synthetic fraud dataset that had data leakage issues. Documentation of debugging process in `docs/ROOT_CAUSE_ANALYSIS.md`.
