#!/usr/bin/env python3
"""
Comprehensive test of FairnessTargetedPGD on all tabular datasets.

Tests gradient-based fairness attack vs random/heuristic baselines.
"""

import numpy as np
import torch
import sys
import time

sys.path.insert(0, '/Users/srujansai/Desktop/DRO-FairML')

from src.data.datasets import get_dataset
from src.corruption.adversarial import AdversarialCorruptor, FairnessTargetedPGD, RandomCorruptor
from src.models.classifier import MLPClassifier


def compute_dp(y_pred, a):
    mask0 = (a == 0)
    mask1 = (a == 1)
    if np.sum(mask0) == 0 or np.sum(mask1) == 0:
        return 0.0
    return abs(np.mean(y_pred[mask0]) - np.mean(y_pred[mask1]))


def train_model(X, y, epochs=20, lr=0.01):
    device = 'cpu'
    model = MLPClassifier(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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


def evaluate(model, X_test, y_test, a_test):
    device = 'cpu'
    X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_t)).numpy()
    dp = compute_dp(y_pred, a_test)
    acc = np.mean((y_pred >= 0.5) == y_test)
    return dp, acc, y_pred


def test_dataset(dataset_name, alpha=0.2):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}, alpha={alpha}")
    print(f"{'='*60}")

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = get_dataset(dataset_name)

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Protected: a=0: {np.sum(a_train==0)}, a=1: {np.sum(a_train==1)}")
    print(f"Label: y=0: {np.sum(y_train==0)}, y=1: {np.sum(y_train==1)}")

    results = {}

    # Clean baseline
    print("\n[1/5] Clean baseline...", end=' ', flush=True)
    t0 = time.time()
    model_clean = train_model(X_train, y_train)
    dp_clean, acc_clean, _ = evaluate(model_clean, X_test, y_test, a_test)
    print(f"DP={dp_clean:.4f}, Acc={acc_clean:.4f} ({time.time()-t0:.1f}s)")
    results['clean'] = {'dp': dp_clean, 'acc': acc_clean}

    # Random corruption
    print("[2/5] Random corruption...", end=' ', flush=True)
    t0 = time.time()
    rc = RandomCorruptor(alpha=alpha, random_state=42)
    X_rc, y_rc, a_rc, mask_rc = rc.corrupt(X_train.copy(), y_train.copy(), a_train.copy())
    model_rc = train_model(X_rc, y_rc)
    dp_rc, acc_rc, _ = evaluate(model_rc, X_test, y_test, a_test)
    print(f"DP={dp_rc:.4f}, Acc={acc_rc:.4f} ({time.time()-t0:.1f}s)")
    results['random'] = {'dp': dp_rc, 'acc': acc_rc, 'n_corrupt': int(np.sum(mask_rc))}

    # Heuristic adversarial
    print("[3/5] Heuristic adversarial...", end=' ', flush=True)
    t0 = time.time()
    ac = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=42)
    X_ac, y_ac, a_ac, mask_ac = ac.corrupt(X_train.copy(), y_train.copy(), a_train.copy())
    model_ac = train_model(X_ac, y_ac)
    dp_ac, acc_ac, _ = evaluate(model_ac, X_test, y_test, a_test)
    print(f"DP={dp_ac:.4f}, Acc={acc_ac:.4f} ({time.time()-t0:.1f}s)")
    results['heuristic'] = {'dp': dp_ac, 'acc': acc_ac, 'n_corrupt': int(np.sum(mask_ac))}

    # Fairness-Targeted PGD (gradient-based) - NEW
    print("[4/5] Fairness-Targeted PGD (gradient)...", end=' ', flush=True)
    t0 = time.time()
    ftpgd = FairnessTargetedPGD(alpha=alpha, target_metric='dp', coordinated=True, random_state=42)
    X_ft, y_ft, a_ft, mask_ft = ftpgd.corrupt(X_train.copy(), y_train.copy(), a_train.copy())
    model_ft = train_model(X_ft, y_ft)
    dp_ft, acc_ft, _ = evaluate(model_ft, X_test, y_test, a_test)
    print(f"DP={dp_ft:.4f}, Acc={acc_ft:.4f} ({time.time()-t0:.1f}s)")
    results['grad_pgd'] = {'dp': dp_ft, 'acc': acc_ft, 'n_corrupt': int(np.sum(mask_ft))}

    # Summary
    print(f"\n--- Summary for {dataset_name} ---")
    print(f"Clean:        DP={dp_clean:.4f}, Acc={acc_clean:.4f}")
    print(f"Random:       DP={dp_rc:.4f}, Acc={acc_rc:.4f} (n={results['random']['n_corrupt']})")
    print(f"Heuristic:   DP={dp_ac:.4f}, Acc={acc_ac:.4f} (n={results['heuristic']['n_corrupt']})")
    print(f"Grad-PGD:     DP={dp_ft:.4f}, Acc={acc_ft:.4f} (n={results['grad_pgd']['n_corrupt']})")
    print(f"DP Increase:  Clean→Grad: {dp_ft - dp_clean:+.4f}")

    return results


if __name__ == '__main__':
    all_results = {}
    for dataset in ['adult', 'credit', 'lsac']:
        try:
            all_results[dataset] = test_dataset(dataset, alpha=0.2)
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Clean:      DP={results['clean']['dp']:.4f}")
        print(f"  Grad-PGD:   DP={results['grad_pgd']['dp']:.4f}")
        print(f"  Attack effect: {results['grad_pgd']['dp'] - results['clean']['dp']:+.4f}")

    print("\n\nDONE: All tests complete")