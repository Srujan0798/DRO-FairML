#!/usr/bin/env python3
"""
Kuldeep Q1 hyperparameter grid search (DRO, Adult).

Kuldeep: "Can we try different initial value of lambdas, learning rates ... to
relax accuracy and tighten DP? If accuracy drops and DP drops, that fits our setup."

Grid: lambda_init x lr_lambda for DRO, DP attack, at the new best temperature tau=1
(the tau ablation showed fixed tau=1 makes DRO beat Naive on DP at every alpha).
Adult only, per Kuldeep "start by testing on the Adult dataset."

Incremental save + resume. Output: results/lambda_lr_grid.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch

TAU = 1.0            # fixed, the tau-ablation winner
ATTACK = 'dp'        # Kuldeep Q1 is about tightening DP
DATASET = 'adult'    # Kuldeep: start with Adult


def run_one(alpha, seed, lambda_init, lr_lambda):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    X_tr, y_tr, a_tr, X_v, y_v, a_v, X_te, y_te, a_te, _ = get_dataset(DATASET, random_state=seed)

    attack = FairnessTargetedPGD(alpha=alpha, target_metric=ATTACK, pgd_steps=20,
                                 epsilon=0.3, pgd_step_size=0.02,
                                 coordinated=False, random_state=seed)
    X_att, y_att, a_att, _ = attack.corrupt(X_tr, y_tr, a_tr)

    model = MLPClassifier(X_tr.shape[1], hidden_dims=[128, 64], dropout=0.1)
    trainer = DroFairTrainer(
        model, alpha=alpha, lr_theta=1e-3, lr_lambda=lr_lambda, lr_p=5e-3,
        lambda_max=1.5, tau=TAU, beta=5.0, k=5, gamma=0.0,
        K_inner=10, epochs=60, weight_decay=1e-4, tau_warmup_epochs=15,
        lambda_init=lambda_init,
        radii_mode='uniform',
        use_dp=True, use_if=False,  # DP-targeted attack → no IF constraint needed
    )
    trainer.fit(X_att, y_att, a_att, X_val=X_v, y_val=y_v, a_val=a_v, verbose=False)
    m = compute_metrics_torch(model, X_te, y_te, a_te, temperature=TAU, k=5, gamma=0.0)
    return {'acc': float(m['accuracy']), 'dp': float(m['dp_violation']), 'if': float(m['if_violation'])}


def main():
    alphas = [0.2, 0.3]
    seeds = [0, 1, 2]
    lambda_inits = [0.0, 0.01, 0.1]   # useful subset per task (0.0 paper spec)
    lr_lambdas = [0.001, 0.005]  # 1e-3,5e-3 ; note: current json has 1-row from reduced-epoch local fill; re-run with 60ep on server for full

    os.makedirs('results', exist_ok=True)
    out = 'results/lambda_lr_grid.json'
    results = json.load(open(out)) if os.path.exists(out) else []
    done = {(r['alpha'], r['seed'], r['lambda_init'], r['lr_lambda']) for r in results}

    total = len(alphas)*len(seeds)*len(lambda_inits)*len(lr_lambdas)
    n = 0
    for alpha in alphas:
        for li in lambda_inits:
            for lr in lr_lambdas:
                for seed in seeds:
                    n += 1
                    key = (alpha, seed, li, lr)
                    if key in done:
                        print(f"[{n}/{total}] SKIP α={alpha} s={seed} λ0={li} lr={lr}")
                        continue
                    print(f"[{n}/{total}] α={alpha} s={seed} λ0={li} lr={lr}", flush=True)
                    t0 = time.time()
                    try:
                        r = run_one(alpha, seed, li, lr)
                        r.update({'dataset': DATASET, 'attack': ATTACK, 'tau': TAU,
                                  'alpha': alpha, 'seed': seed,
                                  'lambda_init': li, 'lr_lambda': lr,
                                  'time': time.time()-t0,
                                  # full provenance per hard constraint §0/§1 and §4:
                                  'k_inner': 10,
                                  'radii_mode': 'uniform',
                                  'coordinated': False,
                                  'pgd_steps': 20,
                                  'n_seeds_planned': len(seeds),
                                  'epochs': 60,
                                  'use_if': False,  # for this Q1 speed optimization (DP only)
                                  })
                        results.append(r)
                        json.dump(results, open(out, 'w'), indent=2)
                        print(f"   -> acc={r['acc']:.3f} dp={r['dp']:.4f} ({r['time']:.0f}s)", flush=True)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"   -> FAILED: {e}", flush=True)
    print(f"Done. {len(results)} results in {out}")


if __name__ == '__main__':
    main()
