"""
Run a limited experiment to get new-format results with test-time evaluation.
Runs 3 seeds per dataset/alpha to get a quick sample.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json
import pickle
import time
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


def corrupt_test_data(X_test, y_test, a_test, alpha, seed, device='cpu'):
    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1, pgd_steps=5, pgd_step_size=0.02,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed + 1000
    )
    return corruptor.corrupt(X_test, y_test, a_test, device=device)


def run_single_experiment(dataset_name, alpha, seed, device='cpu'):
    start_time = time.time()

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)

    tau = get_temperature(alpha)

    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1, pgd_steps=5, pgd_step_size=0.02,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed
    )

    X_train_c, y_train_c, a_train_c, _ = corruptor.corrupt(X_train, y_train, a_train, device=device)
    X_test_c, y_test_c, a_test_c, _ = corrupt_test_data(X_test, y_test, a_test, alpha, seed, device)

    input_dim = X_train.shape[1]
    results = {'dataset': dataset_name, 'alpha': alpha, 'seed': seed, 'naive': {}, 'dro': {}}

    # Naive-FAIR
    naive_start = time.time()
    model_naive = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer_naive = NaiveFairTrainer(
        model_naive, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lambda_max=10.0,
        tau=tau, k=5, gamma=0.0, epochs=30, weight_decay=1e-4
    )
    trainer_naive.fit(X_train_c, y_train_c, a_train_c, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)

    naive_time = time.time() - naive_start

    preds_naive_clean = trainer_naive.predict(X_test)
    results['naive']['time'] = naive_time
    results['naive']['clean'] = {
        'accuracy': float(compute_accuracy(y_test, preds_naive_clean)),
        'dp_violation': float(compute_dp_violation(preds_naive_clean, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds_naive_clean, a_test, k=5, gamma=0.0))
    }

    preds_naive_corrupt = trainer_naive.predict(X_test_c)
    results['naive']['corrupted'] = {
        'accuracy': float(compute_accuracy(y_test_c, preds_naive_corrupt)),
        'dp_violation': float(compute_dp_violation(preds_naive_corrupt, a_test_c)),
        'if_violation': float(compute_if_violation(X_test_c, preds_naive_corrupt, a_test_c, k=5, gamma=0.0))
    }

    # DRO-FAIR
    dro_start = time.time()
    model_dro = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=10.0,
        tau=tau, beta=5.0, k=5, gamma=0.0, K_inner=10, epochs=30, weight_decay=1e-4
    )
    trainer_dro.fit(X_train_c, y_train_c, a_train_c, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)

    dro_time = time.time() - dro_start

    preds_dro_clean = trainer_dro.predict(X_test)
    results['dro']['time'] = dro_time
    results['dro']['clean'] = {
        'accuracy': float(compute_accuracy(y_test, preds_dro_clean)),
        'dp_violation': float(compute_dp_violation(preds_dro_clean, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds_dro_clean, a_test, k=5, gamma=0.0))
    }

    preds_dro_corrupt = trainer_dro.predict(X_test_c)
    results['dro']['corrupted'] = {
        'accuracy': float(compute_accuracy(y_test_c, preds_dro_corrupt)),
        'dp_violation': float(compute_dp_violation(preds_dro_corrupt, a_test_c)),
        'if_violation': float(compute_if_violation(X_test_c, preds_dro_corrupt, a_test_c, k=5, gamma=0.0))
    }

    results['total_time'] = time.time() - start_time

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.2, 0.4]
    n_seeds = 3

    all_results = []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        for alpha in alphas:
            print(f"\n  Alpha = {alpha}")
            seed_results = []

            for seed in range(n_seeds):
                print(f"    Seed {seed}...", end=" ", flush=True)
                result = run_single_experiment(dataset, alpha, seed, device)
                seed_results.append(result)
                all_results.append(result)
                print(f"Done (DRO DRO-time: {result['dro']['time']:.1f}s)")

            # Print summary
            for method in ['naive', 'dro']:
                for eval_type in ['clean', 'corrupted']:
                    accs = [r[method][eval_type]['accuracy'] for r in seed_results]
                    dps = [r[method][eval_type]['dp_violation'] for r in seed_results]
                    print(f"      {method.upper()} ({eval_type}): Acc={np.mean(accs):.4f}, DP={np.mean(dps):.4f}")

    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/all_results_new_format.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Runtime summary
    naive_times = [r['naive']['time'] for r in all_results]
    dro_times = [r['dro']['time'] for r in all_results]

    runtime_data = {
        'naive_mean': float(np.mean(naive_times)),
        'naive_std': float(np.std(naive_times)),
        'dro_mean': float(np.mean(dro_times)),
        'dro_std': float(np.std(dro_times)),
        'overhead': float(np.mean(dro_times) / np.mean(naive_times)) if np.mean(naive_times) > 0 else 0,
        'n_experiments': len(naive_times)
    }
    with open('results/runtimes.json', 'w') as f:
        json.dump(runtime_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Runtime overhead: {runtime_data['overhead']:.2f}x (DRO vs Naive)")
    print(f"Saved {len(all_results)} results to results/all_results_new_format.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()