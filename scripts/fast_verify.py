"""
Fast verification script for key cells.
Runs 3 seeds per dataset/alpha to quickly verify CYCLE 3 fixes.
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.training.standard_ml import StandardMLTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation

def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0

def train_warm_start(X_train, y_train, input_dim, device='cpu'):
    model = MLPClassifier(input_dim, [128, 64], 0.1)
    StandardMLTrainer(model, device=device, epochs=10, lr=1e-3).fit(X_train, y_train, verbose=False)
    return model

def run_cell(dataset, alpha, seed):
    print(f"  {dataset} α={alpha} s={seed}", end=' ', flush=True)
    t0 = time.time()

    X_tr, y_tr, a_tr, X_val, y_val, a_val, X_te, y_te, a_te, _ = get_dataset(dataset, data_dir='data/raw', random_state=seed)
    tau = get_temperature(alpha)
    dim = X_tr.shape[1]

    warm = train_warm_start(X_tr, y_tr, dim)
    corr = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
    X_c, y_c, a_c, _ = corr.corrupt(X_tr, y_tr, a_tr, model=warm)

    mp = MLPClassifier(dim, [128, 64], 0.1)
    StandardMLTrainer(mp, device='cpu', epochs=15, lr=1e-3).fit(X_c, y_c, verbose=False)

    # Naive
    mn = MLPClassifier(dim, [128, 64], 0.1)
    mn.load_state_dict(mp.state_dict())
    trainer_naive = NaiveFairTrainer(mn, device='cpu', tau=tau, epochs=15, tau_warmup_epochs=0)
    trainer_naive.fit(X_c, y_c, a_c, X_val, y_val, a_val, verbose=False)
    pn = trainer_naive.predict(X_te)

    # DRO
    md = MLPClassifier(dim, [128, 64], 0.1)
    md.load_state_dict(mp.state_dict())
    trainer_dro = DroFairTrainer(md, alpha=alpha, device='cpu', tau=tau, epochs=15, tau_warmup_epochs=0)
    trainer_dro.fit(X_c, y_c, a_c, X_val, y_val, a_val, verbose=False)
    pd = trainer_dro.predict(X_te)

    elapsed = time.time() - t0
    result = {
        'dataset': dataset, 'alpha': alpha, 'seed': seed,
        'naive_acc': compute_accuracy(y_te, pn),
        'naive_dp': compute_dp_violation(pn, a_te),
        'dro_acc': compute_accuracy(y_te, pd),
        'dro_dp': compute_dp_violation(pd, a_te),
        'time': elapsed, 'tau': tau
    }
    print(f"N_DP={result['naive_dp']:.4f} D_DP={result['dro_dp']:.4f} ({elapsed:.0f}s)")
    return result

def main():
    cells = [
        ('adult', 0.2), ('adult', 0.1), ('adult', 0.3),
        ('credit', 0.4), ('credit', 0.2),
        ('lsac', 0.2), ('lsac', 0.3),
    ]

    results = []
    for dataset, alpha in cells:
        print(f"\n{dataset.upper()} α={alpha}:")
        for seed in [42, 43, 44]:
            r = run_cell(dataset, alpha, seed)
            results.append(r)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for dataset in ['adult', 'credit', 'lsac']:
        print(f"\n{dataset.upper()}:")
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
            subset = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
            if not subset:
                continue
            n_dps = [r['naive_dp'] for r in subset]
            d_dps = [r['dro_dp'] for r in subset]
            n_accs = [r['naive_acc'] for r in subset]
            d_accs = [r['dro_acc'] for r in subset]
            wins = sum(1 for d in d_dps if d < [n for n, d in zip(n_dps, d_dps)][0])
            print(f"  α={alpha}: N_DP={np.mean(n_dps):.4f}, D_DP={np.mean(d_dps):.4f}, "
                  f"N_Acc={np.mean(n_accs):.4f}, D_Acc={np.mean(d_accs):.4f}, "
                  f"DP_win={sum(1 for nd,dd in zip(n_dps,d_dps) if dd<nd)}/{len(subset)}")

    # Save results
    with open('results/fast_verify.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/fast_verify.json")

if __name__ == '__main__':
    main()