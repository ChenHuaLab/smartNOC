#%%
# AI-assisted implementation
# Reviewed and validated by Mincong Fu

import random
import torch
import os

import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

from utils.dataset import CNNOCRealDataset, BalancedSampler
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
log_dir = "./logs/cfg3"
csv_file = "./GF_cfg3.csv"
crop_csv = "./GF_crop_info.csv"
saved_model_dir = "./saved_models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(saved_model_dir, exist_ok=True)
pretrain_model_path = "./saved_models/cfg1_best.pth"
best_model_path = os.path.join(saved_model_dir, "cfg3_best.pth")

# hyperparameters
batch_size = 32
n_epochs = 100
lr_train = 5e-5
wd_train = 1e-3

writer = SummaryWriter(log_dir=log_dir)

# ===== 3. Dataset loading =====
train_dataset = CNNOCRealDataset(csv_file, data_dir, split="train", max_len=5000, preload=True)
train_labels = [
    int(train_dataset.labels[i]["mix_num"])
    for i in range(len(train_dataset))
]
# sampler = BalancedSampler(train_labels, num_per_class=200)
sampler = BalancedSampler(train_labels, num_per_class=450)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,   # use sampler
    shuffle=False,     # must disable shuffle when using a sampler
    num_workers=10,
    pin_memory=True
)

val_dataset = CNNOCRealDataset(csv_file, data_dir, split="val", max_len=5000, preload=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

test_dataset = CNNOCRealDataset(csv_file, data_dir, split="test", max_len=5000, preload=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ===== 4. Model initialization =====
model = CNNOC().to(device)
model.load_state_dict(torch.load(pretrain_model_path))

for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr_train,
    weight_decay=wd_train
)

ce_loss = nn.CrossEntropyLoss()

best_loss = float("inf")
best_epoch = -1
best_acc = 0

#%%
# ===== 5. Training loop =====
for epoch in range(n_epochs):
    model.train()

    all_mix_preds = []
    all_mix_labels = []

    total_loss = 0
    correct = 0
    total_samples = 0

    for X, _, mix_num in train_loader:
        X, mix_num = X.to(device), mix_num.to(device)
        optimizer.zero_grad()

        mix_num_out = model(X)
        loss_mixnum = ce_loss(mix_num_out, mix_num)
        loss = loss_mixnum

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

        # collect predictions and labels for computing balanced accuracy
        all_mix_preds.extend(mix_num_out.argmax(1).cpu().numpy())
        all_mix_labels.extend(mix_num.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc_balanced = balanced_accuracy_score(all_mix_labels, all_mix_preds)

    print(f"Train Epoch {epoch + 1} - Loss: {avg_loss:.4f} - NoC Acc: {acc_balanced:.4f}")

    writer.add_scalar('Loss/Train', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train', acc_balanced, epoch)

    # ===== validation =====
    model.eval()
    val_loss_sum, val_total = 0.0, 0
    val_preds, val_gts = [], []

    with torch.no_grad():
        for X, _, mix_num in val_loader:
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
    for X, _, mix_num in test_loader: # Note: using test_loader here
        X, mix_num = X.to(device, non_blocking=True), mix_num.to(device, non_blocking=True)
        mix_num_out = model(X)
        preds = mix_num_out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_gts.extend(mix_num.cpu().numpy())

final_bal_acc = balanced_accuracy_score(all_gts, all_preds)
print(f"Best Epoch (based on Val Accuracy): {best_epoch}")
print(f"Test Set Balanced Accuracy: {final_bal_acc:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(all_gts, all_preds, target_names=["1p", "2p", "3p", "4p", "5p"], digits=4))

print("Confusion Matrix (Test Set):")
print(confusion_matrix(all_gts, all_preds))
