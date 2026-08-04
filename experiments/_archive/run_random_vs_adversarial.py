#!/usr/bin/env python3
"""
Random vs Adversarial corruption comparison.
Runs Naive-FAIR on clean data, then with random corruption, then with adversarial.
Measures DP increase from each corruption type.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.naive_fair import NaiveFairTrainer
from src.evaluation.metrics import compute_metrics_torch
from src.temperature import get_temperature


def run_comparison(dataset_name, alpha, seed, device='cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)

    tau = get_temperature(alpha)
    input_dim = X_train.shape[1]

    results = {'dataset': dataset_name, 'alpha': alpha, 'seed': seed}

    # 1. Clean baseline
    model = MLPClassifier(input_dim, [128, 64], dropout=0.1)
    trainer = NaiveFairTrainer(model, device=device, lr_theta=1e-3, lr_lambda=5e-3,
                               lambda_max=1.5, tau=tau, k=5, gamma=0.0, epochs=60,
                               weight_decay=1e-4, tau_warmup_epochs=15)
    trainer.fit(X_train, y_train, a_train, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
    m_clean = compute_metrics_torch(trainer.model, X_test, y_test, a_test,
                                     device=device, temperature=tau, k=5, gamma=0.0)
    results['clean'] = {'acc': float(m_clean['accuracy']), 'dp': float(m_clean['dp_violation'])}

    # 2. Random corruption
    n = len(X_train)
    n_corrupt = int(alpha * n)
    rng = np.random.RandomState(seed)
    corrupt_idx = rng.choice(n, n_corrupt, replace=False)
    X_rand = X_train.copy()
    y_rand = y_train.copy()
    a_rand = a_train.copy()
    for idx in corrupt_idx:
        y_rand[idx] = 1 - y_rand[idx]
        a_rand[idx] = 1 - a_rand[idx]
        noise = rng.randn(X_train.shape[1]) * 0.3
        X_rand[idx] = X_train[idx] + noise

    model = MLPClassifier(input_dim, [128, 64], dropout=0.1)
    trainer = NaiveFairTrainer(model, device=device, lr_theta=1e-3, lr_lambda=5e-3,
                               lambda_max=1.5, tau=tau, k=5, gamma=0.0, epochs=60,
                               weight_decay=1e-4, tau_warmup_epochs=15)
    trainer.fit(X_rand, y_rand, a_rand, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
    m_rand = compute_metrics_torch(trainer.model, X_test, y_test, a_test,
                                    device=device, temperature=tau, k=5, gamma=0.0)
    results['random'] = {'acc': float(m_rand['accuracy']), 'dp': float(m_rand['dp_violation'])}

    # 3. Adversarial corruption (fixed attack)
    attack = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=20,
                                  epsilon=0.3, pgd_step_size=0.02, coordinated=False,
                                  random_state=seed)
    X_adv, y_adv, a_adv, _ = attack.corrupt(X_train, y_train, a_train)

    model = MLPClassifier(input_dim, [128, 64], dropout=0.1)
    trainer = NaiveFairTrainer(model, device=device, lr_theta=1e-3, lr_lambda=5e-3,
                               lambda_max=1.5, tau=tau, k=5, gamma=0.0, epochs=60,
                               weight_decay=1e-4, tau_warmup_epochs=15)
    trainer.fit(X_adv, y_adv, a_adv, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
    m_adv = compute_metrics_torch(trainer.model, X_test, y_test, a_test,
                                   device=device, temperature=tau, k=5, gamma=0.0)
    results['adversarial'] = {'acc': float(m_adv['accuracy']), 'dp': float(m_adv['dp_violation'])}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--n_seeds', type=int, default=3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_path = 'results/random_vs_adversarial_new.json'
    all_results = []

    total = len(args.datasets) * len(args.alphas) * args.n_seeds
    count = 0
    for ds in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                count += 1
                print(f"[{count}/{total}] {ds} α={alpha} seed={seed}")
                try:
                    r = run_comparison(ds, alpha, seed, device)
                    all_results.append(r)
                    with open(out_path, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    dp_rand = r['random']['dp'] - r['clean']['dp']
                    dp_adv = r['adversarial']['dp'] - r['clean']['dp']
                    print(f"  clean={r['clean']['dp']:.4f} random→+{dp_rand:.4f} adv→+{dp_adv:.4f} (ratio={dp_adv/max(dp_rand,1e-8):.1f}x)")
                except Exception as e:
                    print(f"  FAILED: {e}")

    print(f"\nSaved {len(all_results)} comparisons to {out_path}")


if __name__ == '__main__':
    main()
