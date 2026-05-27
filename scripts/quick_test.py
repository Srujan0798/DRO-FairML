#!/usr/bin/env python3
"""Quick test of FairnessTargetedPGD."""

import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/srujansai/Desktop/DRO-FairML')

from src.data.datasets import get_dataset
from src.corruption.adversarial import FairnessTargetedPGD
from src.models.classifier import MLPClassifier


def compute_dp(y_pred, a):
    mask0 = (a == 0)
    mask1 = (a == 1)
    if np.sum(mask0) == 0 or np.sum(mask1) == 0:
        return 0.0
    return abs(np.mean(y_pred[mask0]) - np.mean(y_pred[mask1]))


def quick_train(X, y, a, epochs=20):
    device = 'cpu'
    model = MLPClassifier(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


print("Loading Adult dataset...", flush=True)
X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = get_dataset('adult')
print(f"Train: {len(y_train)}, Test: {len(y_test)}", flush=True)

print("Training clean model...", flush=True)
model_clean = quick_train(X_train, y_train, a_train)

device = 'cpu'
X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
with torch.no_grad():
    y_pred_clean = torch.sigmoid(model_clean(X_test_t)).numpy()

dp_clean = compute_dp(y_pred_clean, a_test)
acc_clean = np.mean((y_pred_clean >= 0.5) == y_test)
print(f"Clean: DP={dp_clean:.4f}, Acc={acc_clean:.4f}", flush=True)

print("Applying FairnessTargetedPGD...", flush=True)
ftpgd = FairnessTargetedPGD(alpha=0.2, target_metric='dp', coordinated=True, random_state=42)
X_ft, y_ft, a_ft, mask = ftpgd.corrupt(X_train.copy(), y_train.copy(), a_train.copy())
print(f"Corrupted {np.sum(mask)} samples", flush=True)

print("Training on corrupted data...", flush=True)
model_corrupt = quick_train(X_ft, y_ft, a_ft)

with torch.no_grad():
    y_pred_corrupt = torch.sigmoid(model_corrupt(X_test_t)).numpy()

dp_corrupt = compute_dp(y_pred_corrupt, a_test)
acc_corrupt = np.mean((y_pred_corrupt >= 0.5) == y_test)
print(f"Corrupt: DP={dp_corrupt:.4f}, Acc={acc_corrupt:.4f}", flush=True)

print(f"\nDP Increase: {dp_corrupt - dp_clean:+.4f}", flush=True)
print("\nDONE", flush=True)