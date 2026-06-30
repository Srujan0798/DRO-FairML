#!/usr/bin/env python3
"""
Canonical empirical companion (DRO-only, radii_mode='empirical').

Per MASTER_PLAN.md §4,§8:
- 270 rows: 3 datasets x 5 alphas x 3 attacks x 1 method(dro) x 6 seeds
- Fixed: tau=1.0, K_inner=10, epochs=60, pgd_steps=20, coordinated=True, lambda_max=1.5,
  radii_mode='empirical', lambda_init=0.0
  (coordinated=True is REQUIRED: the empirical radii inversion assumes the 70/30
   minority-targeting structure; see _empirical_pi_clean in src/training/dro_fair.py)
- Output: results/canonical_tau1_empirical.json
- Resume-safe, incremental append+save.
- Full provenance per row.

Usage:
    python experiments/run_canonical_empirical.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import argparse
import numpy as np
import torch

from experiments.run_fairness_pgd import run_single_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--attacks', nargs='+', default=['dp', 'if', 'combined'])
    parser.add_argument('--methods', nargs='+', default=['dro'])
    parser.add_argument('--n_seeds', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--k_inner', type=int, default=10)
    parser.add_argument('--pgd_steps', type=int, default=20)
    args = parser.parse_args()

    TAU = 1.0
    LAMBDA_INIT = 0.0
    # Q5 empirical companion: coordinated=True is REQUIRED so the 70/30 minority
    # targeting matches the assumption inside _empirical_pi_clean (the inversion
    # pi_clean[min] = pi_obs[min] + 0.4*alpha is only exact for the coordinated attack).
    # This is a standalone Q5 study (known coordinated attack -> empirical radii,
    # no clean validation), distinct from the coordinated=False main canonical.
    COORDINATED = True
    LAMBDA_MAX = 1.5
    RADII_MODE = 'empirical'
    N_SEEDS_PLANNED = args.n_seeds

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Empirical companion config: tau={TAU}, k_inner={args.k_inner}, epochs={args.epochs}, "
          f"pgd_steps={args.pgd_steps}, coordinated={COORDINATED}, lambda_init={LAMBDA_INIT}, "
          f"radii_mode={RADII_MODE}, n_seeds_planned={N_SEEDS_PLANNED}")
    print(f"Grid: datasets={args.datasets} alphas={args.alphas} attacks={args.attacks} methods={args.methods} seeds=0..{args.n_seeds-1}")

    os.makedirs('results', exist_ok=True)
    results_path = 'results/canonical_tau1_empirical.json'

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {results_path} (resume)")
    else:
        all_results = []
        print(f"Starting fresh: {results_path}")

    def save_incremental():
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    completed_keys = {
        (r.get('dataset'), r.get('alpha'), r.get('seed'), r.get('attack'), r.get('method'))
        for r in all_results
    }

    total = len(args.datasets) * len(args.alphas) * args.n_seeds * len(args.attacks) * len(args.methods)
    count = 0
    skipped = 0

    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                for attack in args.attacks:
                    for method in args.methods:
                        count += 1
                        key = (dataset, alpha, seed, attack, method)
                        if key in completed_keys:
                            skipped += 1
                            print(f"[{count}/{total}] SKIP (already done): {dataset} a={alpha} seed={seed} attack={attack} method={method}")
                            continue

                        label = f"[{count}/{total}] {dataset} a={alpha} seed={seed} attack={attack} method={method}"
                        print(label)

                        try:
                            t0 = time.time()
                            result = run_single_experiment(
                                dataset, alpha, seed, attack, method, device=device, verbose=False,
                                epochs=args.epochs, k_inner=args.k_inner, pgd_steps=args.pgd_steps,
                                tau=TAU,
                                lambda_init=LAMBDA_INIT,
                                radii_mode=RADII_MODE,
                                coordinated=COORDINATED,
                                n_seeds_planned=N_SEEDS_PLANNED
                            )
                            elapsed = time.time() - t0
                            all_results.append(result)
                            completed_keys.add(key)
                            save_incremental()

                            print(f"  -> acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} "
                                  f"if={result['if_clean']:.4f} ({elapsed:.0f}s)")

                        except Exception as e:
                            print(f"  -> FAILED: {e}")
                            import traceback
                            traceback.print_exc()

    print(f"\nSkipped {skipped} already-completed experiments")
    print(f"Saved {len(all_results)} results to {results_path}")

    if all_results:
        last = all_results[-1]
        prov_keys = ['k_inner', 'tau', 'radii_mode', 'lambda_init', 'coordinated', 'pgd_steps', 'n_seeds_planned', 'epochs']
        missing = [k for k in prov_keys if k not in last]
        if missing:
            print(f"WARNING: last row missing provenance: {missing}")
        else:
            print(f"Provenance verified on sample row: " + ", ".join(f"{k}={last[k]}" for k in prov_keys))


if __name__ == '__main__':
    main()
