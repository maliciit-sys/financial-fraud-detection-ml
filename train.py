import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model import FraudDataset, FraudDetectionNet, train_epoch, validate
import matplotlib.pyplot as plt
import time

# ===== CONFIGURATION =====
EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 1024

# ===== LOAD DATA =====
print("Loading data...")
X_train = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/X_train.csv')
y_train = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/y_train.csv')
X_val = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/X_val.csv')
y_val = pd.read_csv('/home/maliciit/ml-projects/python-projects/scripts/y_val.csv')

train_dataset = FraudDataset(X_train, y_train)
val_dataset = FraudDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ===== INITIALIZE MODEL =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FraudDetectionNet(input_size=X_train.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nDevice: {device}")
print(f"Training samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}\n")

# ===== TRAINING LOOP =====
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

    # Validate
    val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)

    # Store losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Print progress
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

total_time = time.time() - start_time
print(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")
print(f"Best validation loss: {best_val_loss:.4f}")

# ===== PLOT TRAINING CURVES =====
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/maliciit/ml-projects/python-projects/scripts/training_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Training curves saved: training_curves.png")

# ===== SAVE FINAL MODEL =====
torch.save(model.state_dict(), '/home/maliciit/ml-projects/python-projects/scripts/final_model.pth')
print("✓ Final model saved: final_model.pth")
print("\nDay 3-4 Complete!")
