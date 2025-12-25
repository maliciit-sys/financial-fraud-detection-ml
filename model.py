import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# ===== DATASET CLASS (FINAL FIX) =====
class FraudDataset(Dataset):
    def __init__(self, X, y):
        X_numeric = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.X = torch.FloatTensor(X_numeric.values.astype(np.float32))

        # Handle y properly - force to 1D
        if isinstance(y, pd.DataFrame):
            y_values = y.values.flatten()
        else:
            y_values = y.values
        self.y = torch.FloatTensor(y_values.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===== MODEL ARCHITECTURE =====
class FraudDetectionNet(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.relu(self.layer2(x))
        x = self.dropout2(x)

        x = self.relu(self.layer3(x))
        x = self.dropout3(x)

        x = self.sigmoid(self.output(x))
        return x

# ===== TRAINING FUNCTION =====

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ===== VALIDATION FUNCTION =====
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / len(loader), all_preds, all_labels

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/X_train.csv')
    y_train = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/y_train.csv')
    X_val = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/X_val.csv')
    y_val = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/y_val.csv')

    # Create datasets
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FraudDetectionNet(input_size=13).to(device)

    # Loss and optimizer
    pos_weight = torch.tensor([26.8]).to(device)
    criterion = nn.BCELoss()  # Will handle weights manually in training
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print("\nModel ready for training!")
