#!/usr/bin/env python3
"""
FULL hyperparameter lambda-trajectory diagnostic.

The quick diagnostic (run_lambda_diagnostic.py) used K_inner=3 and epochs=20
for speed. This version uses the PRODUCTION settings: K_inner=10, epochs=60.
If lambda_DP runaway only appears at full settings, this will catch it.

Run: python3 experiments/run_lambda_diagnostic_full.py
Expected time: ~45-60 min on CPU for 12 runs (4 configs x 3 seeds).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch
from src.temperature import get_temperature


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_one(dataset_name, alpha, seed, lambda_max, tag, device='cpu'):
    set_seed(seed)
    t0 = time.time()
    X_tr, y_tr, a_tr, X_v, y_v, a_v, X_te, y_te, a_te, _ = \
        get_dataset(dataset_name, random_state=seed)

    # Use the ORIGINAL AdversarialCorruptor (not FairnessTargetedPGD)
    # because the runaway was originally observed with this corruption.
    attack = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed)
    X_a, y_a, a_a, _ = attack.corrupt(X_tr, y_tr, a_tr, model=None, device=device)

    tau = get_temperature(alpha)
    model = MLPClassifier(X_a.shape[1], hidden_dims=[128, 64], dropout=0.1)
    trainer = DroFairTrainer(
        model, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=lambda_max,
        tau=tau, beta=5.0, k=5, gamma=0.0,
        K_inner=10, epochs=60, weight_decay=1e-4, tau_warmup_epochs=15,
        lambda_warmstart=0.01
    )
    hist = trainer.fit(X_a, y_a, a_a, X_val=X_v, y_val=y_v, a_val=a_v, verbose=False)
    metrics = compute_metrics_torch(
        trainer.model, X_te, y_te, a_te,
        device=device, temperature=tau, k=5, gamma=0.0
    )
    elapsed = time.time() - t0
    print(f"  [{tag}] seed={seed} acc={metrics['accuracy']:.3f} "
          f"dp={metrics['dp_violation']:.4f} if={metrics['if_violation']:.4f} "
          f"lambda_dp_final={hist['lambda_dp'][-1]:.3f} ({elapsed:.0f}s)")
    return {
        'tag': tag,
        'dataset': dataset_name,
        'alpha': alpha,
        'seed': seed,
        'lambda_max': lambda_max,
        'history': hist,
        'final': {
            'acc': float(metrics['accuracy']),
            'dp': float(metrics['dp_violation']),
            'if_v': float(metrics['if_violation']),
        },
        'elapsed': elapsed,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("WARNING: This uses FULL production settings (K_inner=10, epochs=60).")
    print("Expected wall time: ~45-60 min on CPU for 12 runs.")
    seeds = [0, 1, 2]
    runs = []

    configs = [
        ('adult',  0.2, 1.5, 'adult_lmax1.5'),
        ('adult',  0.2, 0.5, 'adult_lmax0.5'),
        ('credit', 0.2, 1.5, 'credit_lmax1.5'),
        ('lsac',   0.2, 1.5, 'lsac_lmax1.5'),
    ]

    for ds, alpha, lmax, tag in configs:
        print(f"\n[{tag}] dataset={ds} alpha={alpha} lambda_max={lmax}")
        for s in seeds:
            try:
                runs.append(run_one(ds, alpha, s, lmax, tag, device=device))
            except Exception as e:
                print(f"  seed={s} FAILED: {e}")
                import traceback
                traceback.print_exc()

    os.makedirs('results', exist_ok=True)
    out = 'results/lambda_diagnostic_full.json'
    with open(out, 'w') as f:
        json.dump(runs, f, indent=2)
    print(f"\nSaved {len(runs)} runs -> {out}")


if __name__ == '__main__':
    main()
