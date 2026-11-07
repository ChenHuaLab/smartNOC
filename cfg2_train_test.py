#%%
# AI-assisted implementation
# Reviewed and validated by Mincong Fu

import os
import torch
import random

import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report


from utils.dataset import CNNOCRealDataset
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
data_dir = "/home/fumincong/provedit/data/cnnoc_pth/GF"
log_dir = "./logs/cfg2"
csv_file = "./GF_cfg2.csv"
crop_csv = "./GF_crop_info.csv"
saved_model_dir = "./saved_models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(saved_model_dir, exist_ok=True)
best_model_path = os.path.join(saved_model_dir, "cfg2_best.pth")

# hyperparameters
batch_size = 32
n_epochs = 100
learning_rate = 3e-4
weight_decay = 1e-4

writer = SummaryWriter(log_dir=log_dir)

# ===== 3. Dataset loading =====
train_dataset = CNNOCRealDataset(csv_file, data_dir, split="train", max_len=5000, preload=True, crop_csv=crop_csv)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

test_dataset = CNNOCRealDataset(csv_file, data_dir, split="test", max_len=5000, preload=True, crop_csv=crop_csv)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ===== 4. Model initialization =====
model = CNNOC().to(device)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    weight_decay=weight_decay
)
ce_loss = nn.CrossEntropyLoss()

best_loss = float("inf")
best_epoch = -1

#%%
# ===== 5. Training loop =====
for epoch in range(n_epochs):
    model.train()
    train_loss_sum, train_total = 0.0, 0
    train_preds, train_gts = [], []

    for X, _, mix_num in train_loader:
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

    # Save the best model based on training set
    if train_loss < best_loss:
        best_loss = train_loss
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
    for X, _, mix_num in test_loader: # Note: using test_loader here
        X, mix_num = X.to(device, non_blocking=True), mix_num.to(device, non_blocking=True)
        mix_num_out = model(X)
        preds = mix_num_out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_gts.extend(mix_num.cpu().numpy())

final_bal_acc = balanced_accuracy_score(all_gts, all_preds)
print(f"Best Epoch (based on Train Loss): {best_epoch}")
print(f"Test Set Balanced Accuracy: {final_bal_acc:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(all_gts, all_preds, target_names=["1p", "2p", "3p", "4p", "5p"], digits=4))

print("Confusion Matrix (Test Set):")
print(confusion_matrix(all_gts, all_preds))
