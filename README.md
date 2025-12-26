# Financial Fraud Detection using Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Project Overview

AI-powered fraud detection system using deep neural networks for fraud detection in financial transactions.

### Key Statistics
- **Dataset Size:** 5,000,000 transactions
- **Fraud Rate:** 3.59% (179,553 cases)
- **Class Imbalance:** 26.8:1 ratio
- **Model Parameters:** 12,161
- **Training Time:** ~7 minutes (CPU)

## 🎯 Project Objectives

- Detect fraudulent transactions with high recall
- Handle severe class imbalance through weighted loss
- Real-time prediction capability (<50ms)
- Minimize financial losses through early detection

## 🏗️ Architecture
```
Input (13 features)
    ↓
Dense(128) → ReLU → Dropout(0.3)
    ↓
Dense(64) → ReLU → Dropout(0.3)
    ↓
Dense(32) → ReLU → Dropout(0.2)
    ↓
Output(1) → Sigmoid
```

## 📁 Repository Structure
```
.
├── data/                   # Data files (not tracked)
├── models/                 # Saved model checkpoints
│   ├── best_model.pth
│   └── final_model.pth
├── outputs/                # Visualizations and results
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_curves.png
├── docs/                   # Documentation
│   ├── baseline_report.md
│   ├── optimization_notes.md
│   └── fraud_detection_report.html
├── scripts/                # Python scripts
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

### Training
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py
```

## 📈 Results

### Baseline Model
- **Accuracy:** 96.4% (misleading - predicts all non-fraud)
- **Fraud Recall:** 0%
- **ROC-AUC:** 0.50

### Optimized Model (In Progress)
- Implementation of weighted loss function
- Target: 70-80% fraud recall
- Expected ROC-AUC: 0.75-0.85

## 🛠️ Technology Stack

- **Framework:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Metrics:** Scikit-learn
- **Version Control:** Git

## 📊 Key Features

- Multi-layer perceptron architecture
- Weighted loss for class imbalance
- Early stopping mechanism
- Real-time prediction capability
- Comprehensive evaluation metrics

## 🌍 SDG Alignment

This project contributes to:
- **SDG 8:** Decent Work and Economic Growth
- **SDG 9:** Industry, Innovation, and Infrastructure
- **SDG 16:** Peace, Justice, and Strong Institutions
- **SDG 17:** Partnerships for the Goals

## 📝 Project Timeline

- **Day 1-2:** Data preprocessing & EDA
- **Day 3-4:** Model development
- **Day 5-7:** Evaluation & optimization
- **Day 8-9:** Documentation
- **Day 10:** Final review

## 🤝 Contributing

Contributions welcome! Please read contributing guidelines before submitting PRs.

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Machine Learning Engineering Student

## 📧 Contact

For questions or collaboration: [Your Email]

---

**Status:** Active Development | **Last Updated:** December 2024
