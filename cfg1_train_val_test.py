#%%
# AI-assisted implementation
# Reviewed and validated by Mincong Fu

import os
import random
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from utils.dataset import CNNOCSimDataset
from utils.model import CNNOC

# ==========================================
# 1. Utility functions
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. Main training logic
# ==========================================
seed = 42
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# paths
data_dir = "/home/fumincong/sim_epg/epg_simu_1"
log_dir = "./logs/cfg1"
saved_model_dir = "./saved_models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(saved_model_dir, exist_ok=True)
best_model_path = os.path.join(saved_model_dir, "cfg1_best.pth")

# hyperparameters
batch_size = 32
num_workers = 10
n_epochs = 30
lr = 3e-4
weight_decay = 1e-4

writer = SummaryWriter(log_dir=log_dir)


# ===== 3. Dataset loading and 8:1:1 split =====
new_dataset = CNNOCSimDataset(
    base_dir=data_dir,
    preload=False
)

# Extract labels for stratified splitting
labels = np.array([
    int(os.path.basename(fp).split("_")[0]) - 1
    for fp in new_dataset.file_list
])
indices = np.arange(len(new_dataset))

# --- Step 1: Split into 80% training set and 20% temporary set ---
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=seed,
    stratify=labels
)

# --- Step 2: Split the remaining 20% into validation (10%) and test (10%) ---
temp_labels = labels[temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5, # Half of the 20%
    random_state=seed,
    stratify=temp_labels
)

train_dataset = Subset(new_dataset, train_idx.tolist())
val_dataset = Subset(new_dataset, val_idx.tolist())
test_dataset = Subset(new_dataset, test_idx.tolist())

print(f"Total samples: {len(new_dataset)}")
print(f"Train samples: {len(train_dataset)} (80%)")
print(f"Val samples:   {len(val_dataset)} (10%)")
print(f"Test samples:  {len(test_dataset)} (10%)")

# Build DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=max(2, num_workers // 2), pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=max(2, num_workers // 2), pin_memory=True)

# ===== 4. Model initialization =====
model = CNNOC().to(device)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=weight_decay
)
ce_loss = nn.CrossEntropyLoss()


best_loss = float('inf')
best_epoch = -1

#%%
# ===== 5. Training loop =====
print("===== Pretrain mixnum_head =====")
for epoch in range(n_epochs):
    model.train()
    train_loss_sum, train_total = 0.0, 0
    train_preds, train_gts = [], []

    for X, mix_num in train_loader:
        X, mix_num = X.to(device, non_blocking=True), mix_num.to(device, non_blocking=True)

        optimizer.zero_grad()
        mix_num_out = model(X)
        loss = ce_loss(mix_num_out, mix_num)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        train_loss_sum += loss.item() * bs
        train_total += bs
        train_preds.extend(mix_num_out.argmax(dim=1).detach().cpu().numpy())
        train_gts.extend(mix_num.detach().cpu().numpy())

    train_loss = train_loss_sum / max(1, train_total)
    train_acc = balanced_accuracy_score(train_gts, train_preds)

    print(f"Epoch {epoch+1:03d}/{n_epochs} | Train Loss: {train_loss:.4f} | Train Bal Acc: {train_acc:.4f}")
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)

    # ===== Validation phase =====
    model.eval()
    val_loss_sum, val_total = 0.0, 0
    val_preds, val_gts = [], []

    with torch.no_grad():
        for X, mix_num in val_loader:
            X, mix_num = X.to(device, non_blocking=True), mix_num.to(device, non_blocking=True)
            mix_num_out = model(X)
            loss = ce_loss(mix_num_out, mix_num)

            bs = X.size(0)
            val_loss_sum += loss.item() * bs
            val_total += bs
            val_preds.extend(mix_num_out.argmax(dim=1).cpu().numpy())
            val_gts.extend(mix_num.cpu().numpy())

    val_loss = val_loss_sum / max(1, val_total)
    val_acc = balanced_accuracy_score(val_gts, val_preds)

    print(f"            | Val   Loss: {val_loss:.4f} | Val   Bal Acc: {val_acc:.4f}")
    writer.add_scalar("Loss/Val", val_loss, epoch)
    writer.add_scalar("Accuracy/Val", val_acc, epoch)

    # Save the best model based on validation set
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> Saved best model at epoch {best_epoch}")

writer.close()
#%%
# ===== 6. Final evaluation on the test set =====
print("\n===== Load Best Model and Evaluate on TEST SET =====")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_preds, all_gts = [], []
with torch.no_grad():
    for X, mix_num in test_loader: # Note: using test_loader here
        X, mix_num = X.to(device, non_blocking=True), mix_num.to(device, non_blocking=True)
        mix_num_out = model(X)
        preds = mix_num_out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_gts.extend(mix_num.cpu().numpy())

final_bal_acc = balanced_accuracy_score(all_gts, all_preds)
print(f"Best Epoch (based on Val Loss): {best_epoch}")
print(f"Test Set Balanced Accuracy: {final_bal_acc:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(all_gts, all_preds, target_names=["1p", "2p", "3p", "4p", "5p"], digits=4))

print("Confusion Matrix (Test Set):")
print(confusion_matrix(all_gts, all_preds))
