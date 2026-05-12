"""
Run ablation studies comparing:
- Standard ML (no fairness)
- Naive-FAIR (DP + IF on corrupted data)
- DRO-FAIR joint (DP + IF with robust reweighting)
- DRO-FAIR DP-only
- DRO-FAIR IF-only

Also compares random vs adversarial corruption.
Matches corrected implementation with K=10, τ tuning, Adam for p.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json
from tqdm import tqdm
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor, RandomCorruptor
from src.training.standard_ml import StandardMLTrainer
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
        corrupt_idx = np.random.choice(n, n_corrupt, replace=False)

        corrupt_mask = np.zeros(n, dtype=bool)
        corrupt_mask[corrupt_idx] = True

        X_c = X.copy()
        y_c = y.copy()
        a_c = a.copy()

        if len(corrupt_idx) > 0:
            # Feature noise: random Gaussian perturbation
            col_stds = np.std(X, axis=0, keepdims=True)
            col_stds[col_stds == 0] = 1.0
            X_c[corrupt_idx] = X[corrupt_idx] + 0.1 * col_stds.squeeze() * np.random.randn(len(corrupt_idx), X.shape[1])

            # Random label flips
            y_c[corrupt_idx] = 1 - y_c[corrupt_idx]

            # Random attribute flips
            a_c[corrupt_idx] = 1 - a_c[corrupt_idx]

        return X_c, y_c, a_c, corrupt_mask


def run_ablation(dataset_name, alpha, seed, corruption_type='adversarial', device='cpu'):
    """Run all methods on one dataset/alpha/seed with specified corruption type."""
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)

    tau = get_temperature(alpha)

    if corruption_type == 'adversarial':
        corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
    else:
        corruptor = RandomCorruptor(alpha=alpha, random_state=seed)

    X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train)

    input_dim = X_train.shape[1]
    results = {'dataset': dataset_name, 'alpha': alpha, 'seed': seed, 'corruption': corruption_type}

    methods = {}

    # 1. Standard ML
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = StandardMLTrainer(model, device=device, epochs=30, lr=1e-3)
    trainer.fit(X_tr, y_tr, X_val, y_val, verbose=False)
    preds = trainer.predict(X_test)
    methods['standard_ml'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    # 2. Naive-FAIR
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = NaiveFairTrainer(model, device=device, epochs=30, tau=tau, k=5)
    trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    preds = trainer.predict(X_test)
    methods['naive'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    # 3. DRO-FAIR joint
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = DroFairTrainer(model, alpha=alpha, device=device, epochs=30, tau=tau, k=5,
                             K_inner=10, lr_p=5e-3, use_dp=True, use_if=True)
    trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    preds = trainer.predict(X_test)
    methods['dro_joint'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    # 4. DRO-FAIR DP-only
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = DroFairTrainer(model, alpha=alpha, device=device, epochs=30, tau=tau, k=5,
                             K_inner=10, lr_p=5e-3, use_dp=True, use_if=False)
    trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    preds = trainer.predict(X_test)
    methods['dro_dp_only'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    # 5. DRO-FAIR IF-only
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = DroFairTrainer(model, alpha=alpha, device=device, epochs=30, tau=tau, k=5,
                             K_inner=10, lr_p=5e-3, use_dp=False, use_if=True)
    trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    preds = trainer.predict(X_test)
    methods['dro_if_only'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    results['methods'] = methods
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    datasets = ['adult', 'credit']
    alphas = [0.2, 0.3]
    n_seeds = 10
    corruption_types = ['adversarial', 'random']

    all_results = []

    for corruption_type in corruption_types:
        print(f"\n{'='*60}")
        print(f"CORRUPTION TYPE: {corruption_type.upper()}")
        print(f"{'='*60}")

        for dataset in datasets:
            for alpha in alphas:
                print(f"\n  {dataset.upper()} α={alpha}")
                for seed in tqdm(range(n_seeds), desc=f"    {dataset} α={alpha}"):
                    result = run_ablation(dataset, alpha, seed, corruption_type=corruption_type, device=device)
                    all_results.append(result)

                # Print aggregates for this setting
                for method in ['standard_ml', 'naive', 'dro_joint', 'dro_dp_only', 'dro_if_only']:
                    accs = [r['methods'][method]['accuracy'] for r in all_results
                            if r['dataset'] == dataset and r['alpha'] == alpha and r['corruption'] == corruption_type]
                    dps = [r['methods'][method]['dp_violation'] for r in all_results
                           if r['dataset'] == dataset and r['alpha'] == alpha and r['corruption'] == corruption_type]
                    ifs = [r['methods'][method]['if_violation'] for r in all_results
                           if r['dataset'] == dataset and r['alpha'] == alpha and r['corruption'] == corruption_type]
                    if accs:
                        print(f"    {method:15s}: Acc={np.mean(accs):.4f}±{np.std(accs)/np.sqrt(len(accs)):.4f}, "
                              f"DP={np.mean(dps):.4f}±{np.std(dps)/np.sqrt(len(dps)):.4f}, "
                              f"IF={np.mean(ifs):.4f}±{np.std(ifs)/np.sqrt(len(ifs)):.4f}")

    os.makedirs('results', exist_ok=True)
    with open('results/ablation_full.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate comparison summary
    summary = {}
    for corruption_type in corruption_types:
        summary[corruption_type] = {}
        for dataset in datasets:
            for alpha in alphas:
                key = f"{dataset}_a{int(alpha*10)}"
                summary[corruption_type][key] = {}
                for method in ['naive', 'dro_joint']:
                    accs = [r['methods'][method]['accuracy'] for r in all_results
                            if r['dataset'] == dataset and r['alpha'] == alpha and r['corruption'] == corruption_type]
                    dps = [r['methods'][method]['dp_violation'] for r in all_results
                           if r['dataset'] == dataset and r['alpha'] == alpha and r['corruption'] == corruption_type]
                    if accs:
                        summary[corruption_type][key][method] = {
                            'acc': float(np.mean(accs)),
                            'dp': float(np.mean(dps)),
                        }

    with open('results/ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to results/ablation_full.json and results/ablation_summary.json")


if __name__ == '__main__':
    main()