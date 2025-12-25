import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load original data (before preprocessing)
df = pd.read_csv('/media/maliciit/Data/Ali/MS Data Scince/Semester 1/4. Machine Learning/AI_ML Project/financial_fraud_detection_dataset.csv')

# ===== 1. CLASS DISTRIBUTION =====
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Pie chart
fraud_counts = df['is_fraud'].value_counts()
axes[0, 0].pie(fraud_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')

# Bar chart
axes[0, 1].bar(['Non-Fraud', 'Fraud'], fraud_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0, 1].set_ylabel('Count', fontsize=12)
axes[0, 1].set_title('Fraud vs Non-Fraud Transactions', fontsize=14, fontweight='bold')
axes[0, 1].ticklabel_format(style='plain', axis='y')

# ===== 2. AMOUNT DISTRIBUTION =====
fraud_amounts = df[df['is_fraud']==1]['amount']
normal_amounts = df[df['is_fraud']==0]['amount']

axes[1, 0].hist([normal_amounts, fraud_amounts], bins=50, label=['Non-Fraud', 'Fraud'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7)
axes[1, 0].set_xlabel('Transaction Amount', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, df['amount'].quantile(0.95))  # Remove outliers for clarity

# ===== 3. FRAUD BY TRANSACTION TYPE =====
fraud_by_type = df.groupby('transaction_type')['is_fraud'].agg(['sum', 'count'])
fraud_by_type['rate'] = (fraud_by_type['sum'] / fraud_by_type['count'] * 100)
fraud_by_type = fraud_by_type.sort_values('rate', ascending=False)

axes[1, 1].barh(fraud_by_type.index, fraud_by_type['rate'], color='#e74c3c')
axes[1, 1].set_xlabel('Fraud Rate (%)', fontsize=12)
axes[1, 1].set_title('Fraud Rate by Transaction Type', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 4. RISK SCORES COMPARISON =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scores = ['spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
titles = ['Spending Deviation', 'Velocity Score', 'Geo Anomaly']

for idx, (score, title) in enumerate(zip(scores, titles)):
    axes[idx].boxplot([df[df['is_fraud']==0][score].dropna(),
                       df[df['is_fraud']==1][score].dropna()],
                      labels=['Non-Fraud', 'Fraud'],
                      patch_artist=True,
                      boxprops=dict(facecolor='#3498db'))
    axes[idx].set_ylabel('Score', fontsize=12)
    axes[idx].set_title(title, fontsize=14, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_risk_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 5. CORRELATION HEATMAP =====
# Prepare numeric data only
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 6. SUMMARY STATISTICS =====
print("\n" + "="*60)
print("DATASET SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal Transactions: {len(df):,}")
print(f"Fraudulent: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"Non-Fraudulent: {(df['is_fraud']==0).sum():,} ({(1-df['is_fraud'].mean())*100:.2f}%)")
print(f"\nAmount Statistics:")
print(f"  Mean: ${df['amount'].mean():,.2f}")
print(f"  Median: ${df['amount'].median():,.2f}")
print(f"  Fraud Mean: ${df[df['is_fraud']==1]['amount'].mean():,.2f}")
print(f"  Non-Fraud Mean: ${df[df['is_fraud']==0]['amount'].mean():,.2f}")
print("\n" + "="*60)
