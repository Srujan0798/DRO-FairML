#!/usr/bin/env python3
"""
Comprehensive lambda grid search: Adult + Credit, DP + IF + Combined attacks.
Extends run_lambda_lr_grid.py (Adult DP only) to cover all main claims.

Grid:
  - datasets: adult, credit
  - attacks: dp, if, combined
  - alphas: 0.1, 0.2, 0.3, 0.4
  - lambda_inits: 0.0, 0.001, 0.01, 0.1, 1.0
  - lr_lambdas: 0.001, 0.005
  - seeds: 0, 1, 2

Output: results/lambda_grid_comprehensive.json
Resume-safe, incremental save.

Usage:
    python experiments/run_lambda_grid_comprehensive.py
    python experiments/run_lambda_grid_comprehensive.py --datasets credit --attacks dp
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch

TAU = 1.0


def run_one(dataset, alpha, seed, attack, lambda_init, lr_lambda):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    X_tr, y_tr, a_tr, X_v, y_v, a_v, X_te, y_te, a_te, _ = get_dataset(dataset, random_state=seed)

    atk = FairnessTargetedPGD(alpha=alpha, target_metric=attack, pgd_steps=20,
                              epsilon=0.3, pgd_step_size=0.02,
                              coordinated=False, random_state=seed)
    X_att, y_att, a_att, _ = atk.corrupt(X_tr, y_tr, a_tr)

    use_dp = attack in ('dp', 'combined')
    use_if = attack in ('if', 'combined')

    model = MLPClassifier(X_tr.shape[1], hidden_dims=[128, 64], dropout=0.1)
    trainer = DroFairTrainer(
        model, alpha=alpha, lr_theta=1e-3, lr_lambda=lr_lambda, lr_p=5e-3,
        lambda_max=1.5, tau=TAU, beta=5.0, k=5, gamma=0.0,
        K_inner=10, epochs=60, weight_decay=1e-4, tau_warmup_epochs=0,
        lambda_init=lambda_init,
        radii_mode='uniform',
        use_dp=use_dp, use_if=use_if,
    )
    trainer.fit(X_att, y_att, a_att, X_val=X_v, y_val=y_v, a_val=a_v, verbose=False)
    m = compute_metrics_torch(model, X_te, y_te, a_te, temperature=TAU, k=5, gamma=0.0)
    return {'acc': float(m['accuracy']), 'dp': float(m['dp_violation']), 'if': float(m['if_violation'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit'])
    parser.add_argument('--attacks', nargs='+', default=['dp', 'if', 'combined'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--n_seeds', type=int, default=3)
    args = parser.parse_args()

    lambda_inits = [0.0, 0.001, 0.01, 0.1, 1.0]
    lr_lambdas = [0.001, 0.005]

    os.makedirs('results', exist_ok=True)
    out = 'results/lambda_grid_comprehensive.json'
    results = json.load(open(out)) if os.path.exists(out) else []
    done = {(str(r['dataset']), float(r['alpha']), float(r['seed']),
             str(r['attack']), float(r['lambda_init']), float(r['lr_lambda']))
            for r in results}

    total = (len(args.datasets) * len(args.attacks) * len(args.alphas) *
             args.n_seeds * len(lambda_inits) * len(lr_lambdas))
    n = 0; skipped = 0

    for dataset in args.datasets:
        for attack in args.attacks:
            for alpha in args.alphas:
                for li in lambda_inits:
                    for lr in lr_lambdas:
                        for seed in range(args.n_seeds):
                            n += 1
                            key = (dataset, float(alpha), float(seed), attack, float(li), float(lr))
                            if key in done:
                                skipped += 1
                                continue
                            print(f"[{n}/{total}] {dataset} α={alpha} s={seed} {attack} λ0={li} lr={lr}", flush=True)
                            t0 = time.time()
                            try:
                                r = run_one(dataset, alpha, seed, attack, li, lr)
                                r.update({
                                    'dataset': dataset, 'attack': attack, 'tau': TAU,
                                    'alpha': alpha, 'seed': seed,
                                    'lambda_init': li, 'lr_lambda': lr,
                                    'time': time.time() - t0,
                                    'k_inner': 10, 'radii_mode': 'uniform',
                                    'coordinated': False, 'pgd_steps': 20,
                                    'n_seeds_planned': args.n_seeds, 'epochs': 60,
                                })
                                results.append(r)
                                json.dump(results, open(out, 'w'), indent=2)
                                print(f"   -> acc={r['acc']:.3f} dp={r['dp']:.4f} if={r['if']:.4f} ({r['time']:.0f}s)", flush=True)
                            except Exception as e:
                                import traceback; traceback.print_exc()
                                print(f"   -> FAILED: {e}", flush=True)

    print(f"\nDone. {len(results)} total results ({skipped} skipped). Output: {out}")


if __name__ == '__main__':
    main()
