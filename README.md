# 🛡️ FraudShield: Credit Card Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9969-brightgreen.svg)

**A Deep Learning Solution for Real-Time Fraud Detection with Interpretable Features**

[Features](#-features) • [Results](#-results) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

FraudShield is an advanced credit card fraud detection system that leverages deep learning techniques to identify fraudulent transactions with exceptional accuracy. Unlike traditional approaches that rely on anonymized features (V1, V2...V28), FraudShield operates on **interpretable transaction features**, enabling both high-accuracy detection and explainable results.

### 🎯 Key Results

#### Non-PCA Datasets (Interpretable Features) ✅ Current Focus

| Dataset | Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---------|-------|---------|--------|-----|-----------|--------|
| **Sparkov Primary** | **Neural Network** | **0.9969** | **0.8712** | **0.8196** | **87.76%** | **76.88%** |

**Additional Metrics @ Optimal Threshold (0.90):**
- False Positive Rate: **0.04%**
- True Positives: 1,649 / 2,145 frauds detected
- Estimated Net Benefit: **~$611,462** per 500K transactions

#### PCA Datasets (Archived)

| Dataset | Model | ROC-AUC | F1 | Precision | Recall |
|---------|-------|---------|-----|-----------|--------|
| PCA Primary | Neural Network | 0.9356 | 0.8054 | 80.0% | 81.1% |
| PCA Primary | XGBoost | 0.9712 | 0.8633 | 92.3% | 81.1% |
| PCA Secondary | Neural Network | 0.8776 | 0.1712 | 10.2% | 52.4% |
| PCA Secondary | XGBoost | 0.8835 | 0.2282 | 23.0% | 22.7% |

> **Note:** The Non-PCA approach with interpretable features significantly outperforms PCA-based methods while providing explainable results suitable for regulatory compliance.

---

## ✨ Features

### 🧠 Neural Network Architecture
- **Architecture**: 22 → 256 → 128 → 64 → 1 fully-connected network
- **Regularization**: BatchNorm + Dropout (30%) between layers
- **Loss Function**: Focal Loss (α=0.75, γ=2.0) for class imbalance
- **Optimizer**: Adam with learning rate scheduling
- **Parameters**: 48,001 trainable parameters

### 📊 Interpretable Feature Engineering
- **Temporal Features**: Hour, day of week, month extraction
- **Demographic Features**: Age calculation from DOB
- **Categorical Encoding**: One-hot (14 merchant categories) + Target encoding (51 states)
- **Numerical Scaling**: StandardScaler normalization

### 🖥️ FraudShield Dashboard
- **Single Transaction Analysis**: Real-time fraud probability prediction
- **Batch Processing**: CSV upload for bulk transaction analysis
- **Interactive Visualizations**: ROC curves, confusion matrices, probability distributions
- **Business Impact Calculator**: Estimated financial savings
- **Threshold Optimization**: Adjustable decision boundaries

---

## 📈 Results

### Model Performance

<table>
<tr>
<td width="50%">

#### Classification Metrics @ Threshold 0.90
| Metric | Value |
|--------|-------|
| True Positives | 1,649 |
| True Negatives | 553,344 |
| False Positives | 230 |
| False Negatives | 496 |
| Precision | 87.76% |
| Recall | 76.88% |

</td>
<td width="50%">

#### Threshold Analysis
| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.30 | 4.13% | 99.11% | 0.079 |
| 0.50 | 8.51% | 98.46% | 0.157 |
| 0.70 | 24.11% | 94.73% | 0.384 |
| **0.90** | **87.76%** | **76.88%** | **0.820** |
| 0.95 | 96.96% | 62.47% | 0.760 |

</td>
</tr>
</table>

### Key Insights from Exploratory Data Analysis

| Pattern | Finding | Business Implication |
|---------|---------|---------------------|
| **Transaction Amount** | Fraud avg: $531 vs Legitimate avg: $68 (8x higher) | Flag high-value transactions for review |
| **Temporal** | Late-night (10PM-12AM) fraud rate: 2.84% vs 0.58% avg | Enhanced monitoring during off-hours |
| **Category** | Online shopping: 1.76% fraud rate (3x average) | Additional verification for CNP transactions |
| **Demographics** | Age groups 18-25 and 65+ show elevated risk | Risk-based authentication triggers |

---

## 📁 Project Structure

```
FraudShield/
├── 📂 data/
│   ├── raw/                    # Original Sparkov dataset
│   └── processed/              # Preprocessed features
├── 📂 models/
│   └── non_pca_primary/        # Trained model artifacts
│       ├── nn_model.pth        # PyTorch model weights
│       ├── scaler.pkl          # StandardScaler
│       ├── encoders.pkl        # Label encoders
│       ├── model_config.json   # Architecture config
│       └── training_history.pkl # Training metrics
├── 📂 notebooks/
│   └── non_pca/
│       ├── 01_non_pca_primary_eda.ipynb    # Exploratory analysis
│       └── 02_non_pca_primary_nn.ipynb     # Model training
├── 📂 frontend/
│   ├── app.py                  # FraudShield Streamlit dashboard
│   ├── requirements.txt        # Dashboard dependencies
│   └── README.md               # Dashboard documentation
├── 📂 docs/
│   ├── 01_Project_Proposal.docx
│   ├── 02_Project_Documentation.docx
│   ├── 03_Project_Presentation.pptx
│   ├── 04_Literature_Review.docx
│   └── Sparkov_Fraud_Detection_Research_Final.docx
├── 📂 outputs/
│   └── figures/                # Generated visualizations
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 LICENSE
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/maliciit-sys/financial-fraud-detection-ml
cd FraudShield

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
streamlit>=1.28.0
```

---

## 💻 Usage

### Running the Dashboard

```bash
cd frontend
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Training the Model

```bash
# Run the training notebook
jupyter notebook notebooks/non_pca/02_non_pca_primary_nn.ipynb
```

### Making Predictions

```python
import torch
import pandas as pd
import pickle

# Load model and preprocessors
model = torch.load('models/non_pca_primary/nn_model.pth')
scaler = pickle.load(open('models/non_pca_primary/scaler.pkl', 'rb'))
encoders = pickle.load(open('models/non_pca_primary/encoders.pkl', 'rb'))

# Preprocess and predict
# ... (see notebooks for full example)
```

---

## 📊 Dataset

This project uses the **Sparkov Credit Card Transaction Dataset**:

| Attribute | Value |
|-----------|-------|
| Total Transactions | 1,852,394 |
| Training Set | 1,296,675 (0.58% fraud) |
| Test Set | 555,719 (0.39% fraud) |
| Features | 23 interpretable attributes |
| Time Period | Jan 2019 - Jun 2020 |
| Imbalance Ratio | 1:172 |

### Features Used

| Category | Features |
|----------|----------|
| **Transaction** | amt (amount), category (14 types) |
| **Temporal** | hour, day_of_week, month |
| **Customer** | age, gender, city_pop, state |
| **Location** | lat, long, merch_lat, merch_long |

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Project Proposal](docs/01_Project_Proposal.docx) | Initial project plan and objectives |
| [Project Documentation](docs/02_Project_Documentation.docx) | Technical specifications and timeline |
| [Project Presentation](docs/03_Project_Presentation.pptx) | 15-slide presentation with results |
| [Literature Review](docs/04_Literature_Review.docx) | Review of 14 recent research papers |
| [Research Report](docs/Sparkov_Fraud_Detection_Research_Final.docx) | Comprehensive 13-page research document |

---

## 🔬 Methodology

### Data Pipeline

```
Raw Data → Temporal Extraction → Demographic Derivation → Categorical Encoding → Numeric Scaling → 22 Features
```

### Training Strategy

1. **Class Imbalance Handling**
   - Weighted Random Sampling (50/50 fraud/legitimate per batch)
   - Focal Loss (α=0.75, γ=2.0)

2. **Regularization**
   - Dropout (30%)
   - Batch Normalization
   - Weight Decay (1e-5)
   - Early Stopping (patience=10)

3. **Optimization**
   - Adam Optimizer (lr=0.001)
   - Learning Rate Scheduling (ReduceLROnPlateau)
   - Batch Size: 1024

---

## 🎯 SDG Alignment

This project contributes to the following UN Sustainable Development Goals:

| SDG | Contribution |
|-----|--------------|
| **SDG 8**: Decent Work & Economic Growth | Protects financial institutions and supports secure digital commerce |
| **SDG 9**: Industry, Innovation & Infrastructure | Applies cutting-edge deep learning for resilient financial infrastructure |
| **SDG 16**: Peace, Justice & Strong Institutions | Combats financial crime and strengthens institutional integrity |

---

## 👨‍💻 Author

**Muhammad Ali Tahir**  
MS Data Science Program  
Superior University, Lahore, Pakistan

**Supervisor**: Mr. Talha Nadeem

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Sparkov Data Generation tool for the synthetic dataset
- Superior University for academic support
- PyTorch and Streamlit communities for excellent documentation

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

</div>
