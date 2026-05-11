"""Quick sanity test before full experiments."""
import sys
import time
import torch
import numpy as np
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation

device = 'cpu'
print("Loading Adult dataset...")
X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
    get_dataset('adult', random_state=42)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

# Subsample for quick test
n_sub = 2000
idx = np.random.RandomState(42).choice(len(X_train), n_sub, replace=False)
X_train_s = X_train[idx]
y_train_s = y_train[idx]
a_train_s = a_train[idx]

idx_v = np.random.RandomState(42).choice(len(X_val), n_sub//4, replace=False)
X_val_s = X_val[idx_v]
y_val_s = y_val[idx_v]
a_val_s = a_val[idx_v]

print("Applying adversarial corruption (alpha=0.2)...")
corruptor = AdversarialCorruptor(alpha=0.2, coordinated=True, random_state=42)
X_train_c, y_train_c, a_train_c, _ = corruptor.corrupt(X_train_s, y_train_s, a_train_s)
X_val_c, y_val_c, a_val_c, _ = corruptor.corrupt(X_val_s, y_val_s, a_val_s)

print("Training Naive-FAIR (3 epochs)...")
model_naive = MLPClassifier(X_train_s.shape[1], hidden_dims=[32, 16], dropout=0.1)
trainer_naive = NaiveFairTrainer(model_naive, device=device, epochs=3, tau=100.0, k=5)
t0 = time.time()
trainer_naive.fit(X_train_c, y_train_c, a_train_c, X_val_c, y_val_c, a_val_c, verbose=True)
print(f"Naive-FAIR took {time.time()-t0:.1f}s")
preds_naive = trainer_naive.predict(X_test)
acc_naive = compute_accuracy(y_test, preds_naive)
dp_naive = compute_dp_violation(preds_naive, a_test)
if_naive = compute_if_violation(X_test, preds_naive, a_test, k=5)
print(f"Naive-FAIR -> Acc: {acc_naive:.4f}, DP: {dp_naive:.4f}, IF: {if_naive:.4f}")

print("Training DRO-FAIR (3 epochs)...")
model_dro = MLPClassifier(X_train_s.shape[1], hidden_dims=[32, 16], dropout=0.1)
trainer_dro = DroFairTrainer(model_dro, alpha=0.2, device=device, epochs=3, tau=100.0, k=5, K_inner=3)
t0 = time.time()
trainer_dro.fit(X_train_c, y_train_c, a_train_c, X_val_c, y_val_c, a_val_c, verbose=True)
print(f"DRO-FAIR took {time.time()-t0:.1f}s")
preds_dro = trainer_dro.predict(X_test)
acc_dro = compute_accuracy(y_test, preds_dro)
dp_dro = compute_dp_violation(preds_dro, a_test)
if_dro = compute_if_violation(X_test, preds_dro, a_test, k=5)
print(f"DRO-FAIR -> Acc: {acc_dro:.4f}, DP: {dp_dro:.4f}, IF: {if_dro:.4f}")

print("\nSanity test completed!")
