#!/usr/bin/env python3
"""
Robust batch runner for Fairness-PGD experiments.
Saves results after EACH experiment, so partial runs are preserved.

Usage:
    python3 experiments/run_fairness_pgd_batch.py --datasets adult --alphas 0.1 0.2 --n_seeds 3
    python3 experiments/run_fairness_pgd_batch.py --datasets adult credit lsac --alphas 0.1 0.2 0.3 --n_seeds 5
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import argparse
import numpy as np
import torch

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


def get_lambda_max(dataset, alpha):
    if dataset == 'adult' and alpha >= 0.2:
        return 0.5
    return 1.5


def run_single(dataset_name, alpha, seed, attack, method, device='cpu', epochs=60, k_inner=10):
    """Run single experiment and return result dict."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)

    tau = get_temperature(alpha)
    input_dim = X_train.shape[1]

    attack_obj = FairnessTargetedPGD(
        alpha=alpha, target_metric=attack, pgd_steps=5,
        coordinated=True, random_state=seed
    )
    X_train_att, y_train_att, a_train_att, _ = attack_obj.corrupt(
        X_train, y_train, a_train
    )

    result = {
        'dataset': dataset_name,
        'alpha': alpha,
        'seed': seed,
        'attack': attack,
        'method': method,
    }

    if method == 'naive':
        model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
        trainer = NaiveFairTrainer(
            model, device=device, lr_theta=1e-3, lr_lambda=5e-3, lambda_max=1.5,
            tau=tau, k=5, gamma=0.0, epochs=epochs, weight_decay=1e-4, tau_warmup_epochs=15
        )
        trainer.fit(X_train_att, y_train_att, a_train_att,
                    X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
        metrics = compute_metrics_torch(
            trainer.model, X_test, y_test, a_test,
            device=device, temperature=tau, k=5, gamma=0.0
        )
    else:
        model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
        lambda_max = get_lambda_max(dataset_name, alpha)
        trainer = DroFairTrainer(
            model, alpha=alpha, device=device,
            lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=lambda_max,
            tau=tau, beta=5.0, k=5, gamma=0.0,
            K_inner=k_inner, epochs=epochs, weight_decay=1e-4, tau_warmup_epochs=15,
            lambda_warmstart=0.01
        )
        trainer.fit(X_train_att, y_train_att, a_train_att,
                    X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
        metrics = compute_metrics_torch(
            trainer.model, X_test, y_test, a_test,
            device=device, temperature=tau, k=5, gamma=0.0
        )

    result['acc_clean'] = float(metrics['accuracy'])
    result['dp_clean'] = float(metrics['dp_violation'])
    result['if_clean'] = float(metrics['if_violation'])
    result['total_time'] = time.time() - start_time

    return result


def load_existing_results(path='results/fairness_pgd_results.json'):
    """Load existing results to resume from."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def already_done(existing, dataset, alpha, seed, attack, method):
    """Check if experiment already in results."""
    for r in existing:
        if (r['dataset'] == dataset and r['alpha'] == alpha and
            r['seed'] == seed and r['attack'] == attack and r['method'] == method):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--attacks', nargs='+', default=['dp', 'if', 'combined'])
    parser.add_argument('--methods', nargs='+', default=['naive', 'dro'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    if args.smoke:
        args.datasets = ['adult']
        args.alphas = [0.2]
        args.n_seeds = 1
        epochs = 10
        k_inner = 3
    else:
        epochs = 60
        k_inner = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs('results', exist_ok=True)
    results_path = 'results/fairness_pgd_results.json'

    all_results = load_existing_results(results_path)
    print(f"Loaded {len(all_results)} existing results")

    total = len(args.datasets) * len(args.alphas) * args.n_seeds * len(args.attacks) * len(args.methods)
    done_count = 0

    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                for attack in args.attacks:
                    for method in args.methods:
                        done_count += 1
                        if already_done(all_results, dataset, alpha, seed, attack, method):
                            print(f"[{done_count}/{total}] SKIP (already done): {dataset} α={alpha} seed={seed} {attack}/{method}")
                            continue

                        print(f"[{done_count}/{total}] RUN: {dataset} α={alpha} seed={seed} {attack}/{method}")
                        try:
                            t0 = time.time()
                            result = run_single(dataset, alpha, seed, attack, method,
                                                device=device, epochs=epochs, k_inner=k_inner)
                            elapsed = time.time() - t0
                            all_results.append(result)

                            # SAVE AFTER EVERY EXPERIMENT
                            with open(results_path, 'w') as f:
                                json.dump(all_results, f, indent=2)

                            print(f"  → acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} "
                                  f"if={result['if_clean']:.4f} ({elapsed:.0f}s) [SAVED]")

                        except Exception as e:
                            print(f"  → FAILED: {e}")
                            import traceback
                            traceback.print_exc()

    print(f"\nDone. Total results: {len(all_results)} → {results_path}")


if __name__ == '__main__':
    main()
