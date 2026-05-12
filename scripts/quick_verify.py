"""
Quick verification script - runs a subset of experiments to verify everything works.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json
import time
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


class RandomCorruptor:
    """Random corruption baseline (as in original paper)."""

    def __init__(self, alpha=0.2, random_state=None):
        self.alpha = alpha
        if random_state is not None:
            np.random.seed(random_state)

    def corrupt(self, X, y, a, device=None):
        n = len(X)
        n_corrupt = int(self.alpha * n)
        corrupt_idx = np.random.choice(n, n_corrupt, replace=False) if n_corrupt > 0 else np.array([], dtype=int)

        X_c = X.copy()
        y_c = y.copy()
        a_c = a.copy()

        if len(corrupt_idx) > 0:
            col_stds = np.std(X, axis=0, keepdims=True)
            col_stds[col_stds == 0] = 1.0
            X_c[corrupt_idx] = X[corrupt_idx] + 0.1 * col_stds.squeeze() * np.random.randn(len(corrupt_idx), X.shape[1])
            y_c[corrupt_idx] = 1 - y_c[corrupt_idx]
            a_c[corrupt_idx] = 1 - a_c[corrupt_idx]

        return X_c, y_c, a_c, corrupt_idx


def run_quick_test(dataset='adult', alpha=0.2, seed=42, device='cpu'):
    """Run a quick test of one dataset/alpha/seed."""
    print(f"\nQuick test: {dataset} α={alpha} seed={seed}")

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset, random_state=seed)

    tau = get_temperature(alpha)
    print(f"  Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")

    # Corrupt training data
    corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
    X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train)

    # Corrupt test data
    X_test_c, y_test_c, a_test_c, _ = corruptor.corrupt(X_test, y_test, a_test)

    input_dim = X_train.shape[1]

    # Naive-FAIR
    start = time.time()
    model_naive = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer_naive = NaiveFairTrainer(
        model_naive, device=device, lr_theta=1e-3, lr_lambda=5e-3, lambda_max=10.0,
        tau=tau, k=5, gamma=0.0, epochs=30, weight_decay=1e-4
    )
    trainer_naive.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    naive_time = time.time() - start

    preds_naive_clean = trainer_naive.predict(X_test)
    preds_naive_corrupt = trainer_naive.predict(X_test_c)

    naive_clean_acc = compute_accuracy(y_test, preds_naive_clean)
    naive_clean_dp = compute_dp_violation(preds_naive_clean, a_test)
    naive_clean_if = compute_if_violation(X_test, preds_naive_clean, a_test, k=5)

    naive_corrupt_acc = compute_accuracy(y_test_c, preds_naive_corrupt)
    naive_corrupt_dp = compute_dp_violation(preds_naive_corrupt, a_test_c)

    print(f"  Naive-FAIR ({naive_time:.1f}s):")
    print(f"    Clean test:  Acc={naive_clean_acc:.4f}, DP={naive_clean_dp:.4f}, IF={naive_clean_if:.4f}")
    print(f"    Corrupt test: Acc={naive_corrupt_acc:.4f}, DP={naive_corrupt_dp:.4f}")

    # DRO-FAIR
    start = time.time()
    model_dro = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=10.0,
        tau=tau, beta=5.0, k=5, gamma=0.0, K_inner=10, epochs=30, weight_decay=1e-4
    )
    trainer_dro.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    dro_time = time.time() - start

    preds_dro_clean = trainer_dro.predict(X_test)
    preds_dro_corrupt = trainer_dro.predict(X_test_c)

    dro_clean_acc = compute_accuracy(y_test, preds_dro_clean)
    dro_clean_dp = compute_dp_violation(preds_dro_clean, a_test)
    dro_clean_if = compute_if_violation(X_test, preds_dro_clean, a_test, k=5)

    dro_corrupt_acc = compute_accuracy(y_test_c, preds_dro_corrupt)
    dro_corrupt_dp = compute_dp_violation(preds_dro_corrupt, a_test_c)

    print(f"  DRO-FAIR ({dro_time:.1f}s):")
    print(f"    Clean test:  Acc={dro_clean_acc:.4f}, DP={dro_clean_dp:.4f}, IF={dro_clean_if:.4f}")
    print(f"    Corrupt test: Acc={dro_corrupt_acc:.4f}, DP={dro_corrupt_dp:.4f}")

    # Compute reductions
    dp_red_clean = (naive_clean_dp - dro_clean_dp) / naive_clean_dp * 100 if naive_clean_dp > 0 else 0
    dp_red_corrupt = (naive_corrupt_dp - dro_corrupt_dp) / naive_corrupt_dp * 100 if naive_corrupt_dp > 0 else 0

    print(f"\n  DRO-FAIR DP reduction:")
    print(f"    On clean test:    {dp_red_clean:.1f}%")
    print(f"    On corrupted test: {dp_red_corrupt:.1f}%")

    return {
        'dataset': dataset,
        'alpha': alpha,
        'seed': seed,
        'naive_time': naive_time,
        'dro_time': dro_time,
        'overhead': dro_time / naive_time if naive_time > 0 else 0,
        'dp_reduction_clean': dp_red_clean,
        'dp_reduction_corrupt': dp_red_corrupt
    }


def compare_corruption_types(dataset='adult', alpha=0.2, seed=42, device='cpu'):
    """Compare adversarial vs random corruption."""
    print(f"\n=== Corruption Comparison: {dataset} α={alpha} ===")

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset, random_state=seed)

    tau = get_temperature(alpha)

    results = {}

    for corr_type in ['adversarial', 'random']:
        if corr_type == 'adversarial':
            corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
        else:
            corruptor = RandomCorruptor(alpha=alpha, random_state=seed)

        X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train)

        model = MLPClassifier(X_train.shape[1], hidden_dims=[64, 32], dropout=0.1)
        trainer = DroFairTrainer(
            model, alpha=alpha, device=device, epochs=30, tau=tau, k=5,
            K_inner=10, lr_p=5e-3, use_dp=True, use_if=True
        )
        trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
        preds = trainer.predict(X_test)

        dp = compute_dp_violation(preds, a_test)
        acc = compute_accuracy(y_test, preds)

        results[corr_type] = {'dp': dp, 'acc': acc}
        print(f"  {corr_type:12s}: Acc={acc:.4f}, DP={dp:.4f}")

    if results['adversarial']['dp'] > results['random']['dp']:
        print(f"  Adversarial corruption is harder (higher DP violation)")
    else:
        print(f"  Random corruption is harder (higher DP violation)")

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test 1: Quick verification
    print("\n" + "="*60)
    print("TEST 1: Quick verification (Adult α=0.2)")
    print("="*60)
    result = run_quick_test('adult', 0.2, seed=42, device=device)

    # Test 2: Compare corruption types
    print("\n" + "="*60)
    print("TEST 2: Adversarial vs Random corruption")
    print("="*60)
    compare_corruption_types('adult', 0.2, seed=42, device=device)

    # Test 3: Check theoretical verification
    print("\n" + "="*60)
    print("TEST 3: Theoretical verification")
    print("="*60)
    import subprocess
    result = subprocess.run(['python3', 'experiments/verify_theory.py'],
                          capture_output=True, text=True, cwd='.')
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    print("\n" + "="*60)
    print("ALL QUICK TESTS COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()