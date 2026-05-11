"""
Run ablation studies comparing:
- Standard ML (no fairness)
- Naive-FAIR (DP + IF on corrupted data)
- DRO-FAIR joint (DP + IF with robust reweighting)
- DRO-FAIR DP-only
- DRO-FAIR IF-only

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
from src.corruption.adversarial import AdversarialCorruptor
from src.training.standard_ml import StandardMLTrainer
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


def run_ablation(dataset_name, alpha, seed, device='cpu'):
    """Run all methods on one dataset/alpha/seed."""
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)

    tau = get_temperature(alpha)

    corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
    X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train)
    # Val stays clean

    input_dim = X_train.shape[1]
    results = {'dataset': dataset_name, 'alpha': alpha, 'seed': seed}

    # 1. Standard ML
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = StandardMLTrainer(model, device=device, epochs=30, lr=1e-3, batch_size=512)
    trainer.fit(X_tr, y_tr, X_val, y_val, verbose=False)
    preds = trainer.predict(X_test)
    results['standard_ml'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    # 2. Naive-FAIR
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = NaiveFairTrainer(model, device=device, epochs=30, tau=tau, k=5, batch_size=512)
    trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    preds = trainer.predict(X_test)
    results['naive'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    # 3. DRO-FAIR joint
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = DroFairTrainer(model, alpha=alpha, device=device, epochs=30, tau=tau, k=5,
                             K_inner=10, batch_size=512, lr_p=5e-3, use_dp=True, use_if=True, adam_p=True)
    trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    preds = trainer.predict(X_test)
    results['dro_joint'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    # 4. DRO-FAIR DP-only
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = DroFairTrainer(model, alpha=alpha, device=device, epochs=30, tau=tau, k=5,
                             K_inner=10, batch_size=512, lr_p=5e-3, use_dp=True, use_if=False, adam_p=True)
    trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    preds = trainer.predict(X_test)
    results['dro_dp_only'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    # 5. DRO-FAIR IF-only
    model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer = DroFairTrainer(model, alpha=alpha, device=device, epochs=30, tau=tau, k=5,
                             K_inner=10, batch_size=512, lr_p=5e-3, use_dp=False, use_if=True, adam_p=True)
    trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    preds = trainer.predict(X_test)
    results['dro_if_only'] = {
        'accuracy': float(compute_accuracy(y_test, preds)),
        'dp_violation': float(compute_dp_violation(preds, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds, a_test, k=5))
    }

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    datasets = ['adult', 'credit']
    alphas = [0.2, 0.3]
    n_seeds = 10

    all_results = []
    for dataset in datasets:
        for alpha in alphas:
            print(f"\n{'='*60}")
            print(f"{dataset.upper()} α={alpha}")
            print(f"{'='*60}")
            for seed in tqdm(range(n_seeds), desc=f"{dataset} α={alpha}"):
                result = run_ablation(dataset, alpha, seed, device)
                all_results.append(result)

            # Print aggregates
            for method in ['standard_ml', 'naive', 'dro_joint', 'dro_dp_only', 'dro_if_only']:
                accs = [r[method]['accuracy'] for r in all_results if r['dataset'] == dataset and r['alpha'] == alpha]
                dps = [r[method]['dp_violation'] for r in all_results if r['dataset'] == dataset and r['alpha'] == alpha]
                ifs = [r[method]['if_violation'] for r in all_results if r['dataset'] == dataset and r['alpha'] == alpha]
                print(f"  {method:15s}: Acc={np.mean(accs):.4f}±{np.std(accs)/np.sqrt(len(accs)):.4f}, "
                      f"DP={np.mean(dps):.4f}±{np.std(dps)/np.sqrt(len(dps)):.4f}, "
                      f"IF={np.mean(ifs):.4f}±{np.std(ifs)/np.sqrt(len(ifs)):.4f}")

    os.makedirs('results', exist_ok=True)
    with open('results/ablation_full.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/ablation_full.json")


if __name__ == '__main__':
    main()
