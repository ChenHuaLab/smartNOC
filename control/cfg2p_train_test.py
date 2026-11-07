#%%
# AI-assisted implementation
# Reviewed and validated by Mincong Fu

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
    accuracy_score
)

from utils.dataset import CNNOCAlleleDataset
from utils.model import CNNOC


# ============================================================
# 1. Random Seed
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Configuration
# ============================================================
set_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

# Data paths
data_dir = "./data"

train_x = os.path.join(data_dir, "X_train.npy")
train_y = os.path.join(data_dir, "y_train.npy")
test_x = os.path.join(data_dir, "X_test.npy")
test_y = os.path.join(data_dir, "y_test.npy")

# Logging and model checkpoint paths
log_dir = "./logs/cfg2p"
saved_model_dir = "./saved_models"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(saved_model_dir, exist_ok=True)

best_model_path = os.path.join(
    saved_model_dir, "cfg2p_best.pth"
)

batch_size = 32
n_epochs = 100
learning_rate = 3e-4
weight_decay = 1e-4
num_workers = 4

num_classes = 5
class_names = ["1p", "2p", "3p", "4p", "5p"]


# ============================================================
# 3. Dataset and DataLoader
# ============================================================
train_dataset = CNNOCAlleleDataset(train_x, train_y)
test_dataset = CNNOCAlleleDataset(test_x, test_y)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

print("\nTrain samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

# Check the actual input data
X0, y0 = train_dataset[0]
print("Single input shape:", X0.shape)  # (1, 24, 80)
print("Single label:", y0.item())


# ============================================================
# 4. Model Initialization
# ============================================================
model = CNNOC(num_classes=num_classes).to(device)

# Check the output dimensions of the forward pass before training
with torch.no_grad():
    dummy_x = torch.zeros(2, *X0.shape, device=device)
    dummy_out = model(dummy_x)

assert dummy_out.shape == (2, num_classes), (
    f"Invalid model output shape: {dummy_out.shape}"
)

print("Model output shape:", dummy_out.shape)

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir=log_dir)


# ============================================================
# 5. Training
# ============================================================
best_train_loss = float("inf")
best_epoch = -1

for epoch in range(n_epochs):
    model.train()

    loss_sum = 0.0
    total = 0
    all_preds = []
    all_labels = []

    for X, y in train_loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        bs = X.size(0)
        loss_sum += loss.item() * bs
        total += bs

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    train_loss = loss_sum / max(total, 1)
    train_acc = accuracy_score(all_labels, all_preds)
    train_bal_acc = balanced_accuracy_score(
        all_labels, all_preds
    )

    print(
        f"Epoch {epoch + 1:03d}/{n_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train Bal Acc: {train_bal_acc:.4f}"
    )

    writer.add_scalar("Loss/Train", train_loss, epoch + 1)
    writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
    writer.add_scalar(
        "BalancedAccuracy/Train", train_bal_acc, epoch + 1
    )

    # Continue selecting the best model based on training loss
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": best_train_loss,
                "num_classes": num_classes,
            },
            best_model_path
        )

        print(f"  --> Saved best model: {best_model_path}")


writer.close()


# ============================================================
# 6. Test Set Evaluation
# ============================================================
print("\n===== Evaluate on TEST SET =====")

checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds = []
all_gts = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device, non_blocking=True)

        logits = model(X)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_gts.extend(y.numpy())

all_labels = list(range(num_classes))

test_acc = accuracy_score(all_gts, all_preds)
test_bal_acc = balanced_accuracy_score(all_gts, all_preds)

print("Best epoch:", checkpoint["epoch"])
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Balanced Accuracy: {test_bal_acc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        all_gts,
        all_preds,
        labels=all_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        all_gts,
        all_preds,
        labels=all_labels
    )
)

print("\nTraining completed.")
print("Best model:", best_model_path)