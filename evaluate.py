import torch
import pandas as pd
import numpy as np
from model import FraudDataset, FraudDetectionNet
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ===== LOAD TEST DATA =====
print("Loading test data...")
X_test = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/X_test.csv')
y_test = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/y_test.csv')['is_fraud']
y_test = pd.DataFrame({'is_fraud': y_test})

test_dataset = FraudDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# ===== LOAD MODEL =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FraudDetectionNet(input_size=X_test.shape[1]).to(device)
model.load_state_dict(torch.load('/home/maliciit/ml-projects/python-projects/scripts/final_model.pth', map_location=device))
model.eval()

# ===== GET PREDICTIONS =====
print("Generating predictions...")
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(y_batch.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds = (all_probs > 0.5).astype(int)

# ===== METRICS =====
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"\nTest samples: {len(all_labels):,}")
print(f"Actual frauds: {all_labels.sum():,}")
print(f"Predicted frauds: {all_preds.sum():,}")

print("\n" + classification_report(all_labels, all_preds, target_names=['Non-Fraud', 'Fraud']))

try:
    auc_score = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC Score: {auc_score:.4f}")
except:
    print("ROC-AUC: Could not calculate")

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('/home/maliciit/ml-projects/python-projects/scripts/confusion_matrix.png', dpi=300)
print("\n✓ Confusion matrix saved")

# ===== ROC CURVE =====
try:
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_score:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/maliciit/ml-projects/python-projects/scripts/roc_curve.png', dpi=300)
    print("✓ ROC curve saved")
except:
    pass

print("\nDay 5-7 Evaluation Complete!")
