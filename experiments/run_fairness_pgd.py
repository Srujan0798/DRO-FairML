#!/usr/bin/env python3
"""
Experiment driver for Fairness-Targeted PGD attacks on tabular datasets.
Trains Naive-FAIR and DRO-FAIR on clean data, applies attacks, retrains, evaluates.

Usage:
    python3 experiments/run_fairness_pgd.py --smoke
    python3 experiments/run_fairness_pgd.py --datasets adult --alphas 0.1 0.2 0.3
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


def run_single_experiment(dataset_name, alpha, seed, attack, method, device='cpu', verbose=False, epochs=60, k_inner=10, pgd_steps=5):
    """Run single (dataset, alpha, seed, attack, method) experiment."""
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
        alpha=alpha,
        target_metric=attack,
        pgd_steps=pgd_steps,
        coordinated=True,
        random_state=seed
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
            model, device=device,
            lr_theta=1e-3, lr_lambda=5e-3, lambda_max=1.5,
            tau=tau, k=5, gamma=0.0,
            epochs=epochs, weight_decay=1e-4, tau_warmup_epochs=15
        )
        trainer.fit(X_train_att, y_train_att, a_train_att,
                     X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)
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
                     X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)
        metrics = compute_metrics_torch(
            trainer.model, X_test, y_test, a_test,
            device=device, temperature=tau, k=5, gamma=0.0
        )

    result['acc_clean'] = float(metrics['accuracy'])
    result['dp_clean'] = float(metrics['dp_violation'])
    result['if_clean'] = float(metrics['if_violation'])
    result['total_time'] = time.time() - start_time

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--attacks', nargs='+', default=['dp', 'if', 'combined'])
    parser.add_argument('--methods', nargs='+', default=['naive', 'dro'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--smoke', action='store_true', help='1 seed, 1 dataset, 1 alpha')
    args = parser.parse_args()

    if args.smoke:
        args.datasets = ['adult']
        args.alphas = [0.2]
        args.n_seeds = 1
        smoke_epochs = 10
        smoke_k_inner = 3
        smoke_pgd_steps = 2
        print("SMOKE TEST MODE: 1 dataset, 1 alpha, 1 seed")
        print("Attacks: dp, if, combined | Methods: naive, dro")
        print("Expected: 6 rows\n")
        print("NOTE: smoke uses epochs=10, K_inner=3, pgd_steps=2 for speed")
    else:
        smoke_epochs = 60
        smoke_k_inner = 10
        smoke_pgd_steps = 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs('results', exist_ok=True)
    results_path = 'results/fairness_pgd_results.json'

    all_results = []

    total = len(args.datasets) * len(args.alphas) * args.n_seeds * len(args.attacks) * len(args.methods)
    count = 0

    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                for attack in args.attacks:
                    for method in args.methods:
                        count += 1
                        label = f"[{count}/{total}] {dataset} α={alpha} seed={seed} attack={attack} method={method}"
                        print(label)

                        try:
                            t0 = time.time()
                            result = run_single_experiment(
                                dataset, alpha, seed, attack, method, device=device, verbose=False,
                                epochs=smoke_epochs, k_inner=smoke_k_inner, pgd_steps=smoke_pgd_steps
                            )
                            elapsed = time.time() - t0
                            all_results.append(result)

                            print(f"  → acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} "
                                  f"if={result['if_clean']:.4f} ({elapsed:.0f}s)")

                        except Exception as e:
                            print(f"  → FAILED: {e}")
                            import traceback
                            traceback.print_exc()

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} results to {results_path}")

    if args.smoke and len(all_results) > 0:
        print("\nSMOKE TEST JSON:")
        print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()