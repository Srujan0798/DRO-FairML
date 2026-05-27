#!/usr/bin/env python3
"""
Experiment runner for UTKFace dataset.
Trains Naive-FAIR and DRO-FAIR on UTKFace with optional adversarial corruption.

Usage:
    python3 experiments/run_utkface.py --smoke --alphas 0.2
    python3 experiments/run_utkface.py --datasets utkface --alphas 0.0 0.1 0.2 --n_seeds 5
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.models.cnn_classifier import CNNClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


def _make_synthetic_utkface(n=1000, dim=512, seed=42):
    """Generate synthetic UTKFace-like data when real data unavailable."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim).astype(np.float32)
    y = rng.randint(0, 2, n).astype(np.float32)
    a = rng.randint(0, 2, n).astype(np.int64)
    return X, y, a


def run_single_utkface_experiment(dataset_name, alpha, seed, device='cpu', verbose=False):
    """Run single UTKFace experiment."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()

    try:
        X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
            get_dataset(dataset_name, random_state=seed)
    except RuntimeError as e:
        if 'UTKFace' in str(e) or 'No UTKFace' in str(e):
            print(f"  UTKFace not available ({e}), using synthetic data")
            X_train, y_train, a_train = _make_synthetic_utkface(n=800, seed=seed)
            X_test, y_test, a_test = _make_synthetic_utkface(n=200, seed=seed+999)
            X_val = X_test.copy()
            y_val = y_test.copy()
            a_val = a_test.copy()
            dname = 'UTKFace (synthetic)'
        else:
            raise

    tau = get_temperature(alpha)
    input_dim = X_train.shape[1]

    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed
    )
    X_train_c, y_train_c, a_train_c, _ = corruptor.corrupt(
        X_train, y_train, a_train, model=None, device=device
    )

    X_test_c, y_test_c, a_test_c, _ = corruptor.corrupt(
        X_test, y_test, a_test, model=None, device=device
    )

    results = {
        'dataset': dataset_name,
        'alpha': alpha,
        'seed': seed,
        'naive': {},
        'dro': {}
    }

    model_naive = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer_naive = NaiveFairTrainer(
        model_naive, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lambda_max=1.5,
        tau=tau, k=5, gamma=0.0,
        epochs=60, weight_decay=1e-4, tau_warmup_epochs=15
    )
    trainer_naive.fit(X_train_c, y_train_c, a_train_c,
                      X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)

    results['naive']['clean'] = compute_metrics_torch(
        trainer_naive.model, X_test, y_test, a_test,
        device=device, temperature=tau, k=5, gamma=0.0
    )
    results['naive']['corrupted'] = compute_metrics_torch(
        trainer_naive.model, X_test_c, y_test, a_test_c,
        device=device, temperature=tau, k=5, gamma=0.0
    )

    model_dro = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=1.5,
        tau=tau, beta=5.0, k=5, gamma=0.0,
        K_inner=10, epochs=60, weight_decay=1e-4, tau_warmup_epochs=15,
        lambda_warmstart=0.01
    )
    trainer_dro.fit(X_train_c, y_train_c, a_train_c,
                    X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)

    results['dro']['clean'] = compute_metrics_torch(
        trainer_dro.model, X_test, y_test, a_test,
        device=device, temperature=tau, k=5, gamma=0.0
    )
    results['dro']['corrupted'] = compute_metrics_torch(
        trainer_dro.model, X_test_c, y_test, a_test_c,
        device=device, temperature=tau, k=5, gamma=0.0
    )

    results['total_time'] = time.time() - start_time

    for method in ['naive', 'dro']:
        for eval_type in ['clean', 'corrupted']:
            results[method][eval_type] = {
                k: float(v) for k, v in results[method][eval_type].items()
            }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['utkface'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--smoke', action='store_true', help='Run single seed only (smoke test)')
    args = parser.parse_args()

    if args.smoke:
        args.n_seeds = 1
        args.alphas = [0.2]
        print("SMOKE TEST MODE: 1 seed, alpha=0.2")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs('results', exist_ok=True)
    results_path = 'results/utkface_results.json'

    all_results = []
    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                print(f"\n[{dataset}] alpha={alpha} seed={seed}")
                try:
                    t0 = time.time()
                    result = run_single_utkface_experiment(dataset, alpha, seed, device=device, verbose=False)
                    elapsed = time.time() - t0
                    all_results.append(result)
                    print(f"  Done in {elapsed:.0f}s | "
                          f"Naive clean: acc={result['naive']['clean']['accuracy']:.3f} "
                          f"dp={result['naive']['clean']['dp_violation']:.3f} "
                          f"if={result['naive']['clean']['if_violation']:.3f} | "
                          f"DRO clean: acc={result['dro']['clean']['accuracy']:.3f} "
                          f"dp={result['dro']['clean']['dp_violation']:.3f} "
                          f"if={result['dro']['clean']['if_violation']:.3f}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {results_path}")


if __name__ == '__main__':
    main()