"""Quick validation: does fixing radii make DRO beat Naive?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.corruption.adversarial import FairnessTargetedPGD
from src.evaluation.metrics import compute_metrics_torch

def train_naive(X_tr, y_tr, a_tr, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPClassifier(X_tr.shape[1], hidden_dims=[128, 64], dropout=0.1)
    trainer = NaiveFairTrainer(model, device='cpu', lr_theta=1e-3, lambda_max=1.0,
                               epochs=60, weight_decay=1e-4, tau=100.0)
    trainer.fit(X_tr, y_tr, a_tr, verbose=False)
    return trainer.model

def train_dro(X_tr, y_tr, a_tr, alpha, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPClassifier(X_tr.shape[1], hidden_dims=[128, 64], dropout=0.1)
    trainer = DroFairTrainer(model, alpha=alpha, device='cpu', lr_theta=1e-3,
                             lr_lambda=5e-3, lr_p=5e-3, lambda_max=1.5, tau=100.0,
                             K_inner=5, epochs=60, weight_decay=1e-4)
    trainer.fit(X_tr, y_tr, a_tr, verbose=False)
    return trainer.model

def eval_dp(model, X_te, y_te, a_te):
    m = compute_metrics_torch(model, X_te, y_te, a_te, device='cpu', temperature=100.0)
    return m['accuracy'], m['dp_violation']

print("=" * 60)
print("QUICK RADII FIX VALIDATION")
print("=" * 60)

seed = 0
alpha = 0.2
X_train, y_train, a_train, _, _, _, X_test, y_test, a_test, _ = get_dataset('adult', random_state=seed)

# Attack
atk = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=20,
                           epsilon=0.3, pgd_step_size=0.02,
                           coordinated=False, random_state=seed)
X_a, y_a, a_a, _ = atk.corrupt(X_train, y_train, a_train)

print(f"\nAdult α={alpha} seed={seed} attack=dp coordinated=False")
print(f"Corrupted train DP: {abs(np.mean(y_a[a_a==0]) - np.mean(y_a[a_a==1])):.4f}")

# Naive
model_n = train_naive(X_a, y_a, a_a, seed)
acc_n, dp_n = eval_dp(model_n, X_test, y_test, a_test)
print(f"\nNaive:  acc={acc_n:.3f}  dp={dp_n:.4f}")

# DRO (with fixed radii)
model_d = train_dro(X_a, y_a, a_a, alpha, seed)
acc_d, dp_d = eval_dp(model_d, X_test, y_test, a_test)
print(f"DRO:    acc={acc_d:.3f}  dp={dp_d:.4f}")

print(f"\nDRO < Naive? {'YES ✅' if dp_d < dp_n else 'NO ❌'}")
print(f"Improvement: {dp_n - dp_d:+.4f}")
