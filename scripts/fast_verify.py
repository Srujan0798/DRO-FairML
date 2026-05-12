"""
Fast verification: Adult α=0.2 DP and Credit α=0.4 τ=1 accuracy.
Uses only 3 seeds to quickly verify the fix.
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


def train_warm_start(X_train, y_train, a_train, input_dim, device='cpu', epochs=10):
    model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer = StandardMLTrainer(model, device=device, epochs=epochs, lr=1e-3)
    trainer.fit(X_train, y_train, verbose=False)
    return model


def run_single(dataset, alpha, seed):
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset(dataset, random_state=seed)
    tau_train = get_temperature(alpha)
    input_dim = X_train.shape[1]

    warm = train_warm_start(X_train, y_train, a_train, input_dim, 'cpu', 10)

    corr = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
    X_tr, y_tr, a_tr, _ = corr.corrupt(X_train, y_train, a_train, model=warm, device='cpu')

    mp = MLPClassifier(input_dim, [128, 64], 0.1)
    StandardMLTrainer(mp, device='cpu', epochs=15, lr=1e-3).fit(X_tr, y_tr, verbose=False)

    # Naive
    mn = MLPClassifier(input_dim, [128, 64], 0.1)
    mn.load_state_dict(mp.state_dict())
    tn = NaiveFairTrainer(mn, device='cpu', tau=tau_train, epochs=15, tau_warmup_epochs=5)
    tn.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    pn = tn.predict(X_test)

    # DRO
    md = MLPClassifier(input_dim, [128, 64], 0.1)
    md.load_state_dict(mp.state_dict())
    tr = DroFairTrainer(md, alpha=alpha, device='cpu', tau=tau_train, epochs=15, tau_warmup_epochs=5)
    tr.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    pd = tr.predict(X_test)

    return {
        'naive_acc': compute_accuracy(y_test, pn),
        'naive_dp': compute_dp_violation(pn, a_test),
        'dro_acc': compute_accuracy(y_test, pd),
        'dro_dp': compute_dp_violation(pd, a_test)
    }


if __name__ == '__main__':
    cells = [
        ('adult', 0.2, 42), ('adult', 0.2, 43), ('adult', 0.2, 44),
        ('adult', 0.1, 42), ('adult', 0.3, 42),
        ('credit', 0.4, 42),
    ]

    results = []
    for dataset, alpha, seed in cells:
        print(f"Running {dataset} α={alpha} seed={seed}...", end=' ', flush=True)
        start = time.time()
        r = run_single(dataset, alpha, seed)
        r['dataset'] = dataset
        r['alpha'] = alpha
        r['seed'] = seed
        results.append(r)
        print(f"Done in {time.time()-start:.0f}s: N_DP={r['naive_dp']:.4f}, D_DP={r['dro_dp']:.4f}")

    print("\n=== SUMMARY ===")
    adult_02 = [r for r in results if r['dataset'] == 'adult' and r['alpha'] == 0.2]
    adult_02_wins = sum(1 for r in adult_02 if r['dro_dp'] < r['naive_dp'])
    print(f"Adult α=0.2 DP wins: {adult_02_wins}/3")

    credit_04 = [r for r in results if r['dataset'] == 'credit' and r['alpha'] == 0.4][0]
    print(f"Credit α=0.4 DRO acc: {credit_04['dro_acc']:.4f} (need ≥0.60)")

    all_wins = sum(1 for r in results if r['dro_dp'] < r['naive_dp'])
    print(f"Total DP wins: {all_wins}/{len(results)}")

    for r in results:
        win = r['dro_dp'] < r['naive_dp']
        print(f"  {r['dataset']} α={r['alpha']} s={r['seed']}: N={r['naive_dp']:.4f}, D={r['dro_dp']:.4f}, win={win}")