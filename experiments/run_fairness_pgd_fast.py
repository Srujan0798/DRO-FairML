#!/usr/bin/env python3
"""
FAST batch runner for Fairness-PGD experiments.
Uses epochs=30 and K_inner=5 for speed (vs 60/10 in full run).
Saves after EACH experiment.

Usage:
    screen -dmS fpgd bash -c 'cd /Users/srujansai/Desktop/DRO-FairML && venv/bin/python3 experiments/run_fairness_pgd_fast.py 2>&1 | tee logs/fast_fpgd.log'
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


def run_single(dataset_name, alpha, seed, attack, method, device='cpu'):
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
        'fast_mode': True,
    }

    if method == 'naive':
        model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
        trainer = NaiveFairTrainer(
            model, device=device, lr_theta=1e-3, lr_lambda=5e-3, lambda_max=1.5,
            tau=tau, k=5, gamma=0.0, epochs=30, weight_decay=1e-4, tau_warmup_epochs=10
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
            K_inner=5, epochs=30, weight_decay=1e-4, tau_warmup_epochs=10,
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


def load_existing(path='results/fairness_pgd_results.json'):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def done(existing, dataset, alpha, seed, attack, method):
    for r in existing:
        if (r['dataset'] == dataset and r['alpha'] == alpha and
            r['seed'] == seed and r['attack'] == attack and r['method'] == method):
            return True
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--attacks', nargs='+', default=['dp', 'if', 'combined'])
    parser.add_argument('--methods', nargs='+', default=['naive', 'dro'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--n_seeds', type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"FAST MODE: epochs=30, K_inner=5")
    print(f"Device: {device}")

    os.makedirs('results', exist_ok=True)
    results_path = 'results/fairness_pgd_results.json'

    all_results = load_existing(results_path)
    print(f"Loaded {len(all_results)} existing results")

    total = len(args.datasets) * len(args.alphas) * args.n_seeds * len(args.attacks) * len(args.methods)
    count = 0

    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                for attack in args.attacks:
                    for method in args.methods:
                        count += 1
                        if done(all_results, dataset, alpha, seed, attack, method):
                            print(f"[{count}/{total}] SKIP: {dataset} α={alpha} s={seed} {attack}/{method}")
                            continue

                        print(f"[{count}/{total}] RUN: {dataset} α={alpha} s={seed} {attack}/{method}")
                        try:
                            t0 = time.time()
                            result = run_single(dataset, alpha, seed, attack, method, device=device)
                            elapsed = time.time() - t0
                            all_results.append(result)

                            with open(results_path, 'w') as f:
                                json.dump(all_results, f, indent=2)

                            print(f"  → acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} "
                                  f"if={result['if_clean']:.4f} ({elapsed:.0f}s) [SAVED]")
                        except Exception as e:
                            print(f"  → FAILED: {e}")
                            import traceback
                            traceback.print_exc()

    print(f"\nDone. Total: {len(all_results)} results → {results_path}")


if __name__ == '__main__':
    main()
