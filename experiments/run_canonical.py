#!/usr/bin/env python3
"""
Canonical experiment driver (Agent A sole owner).

Per MASTER_PLAN.md §3:
- Produces results/canonical_tau1.json (540 rows: 3ds x 5 alphas x 3 attacks x 2 methods x 6 seeds)
- Fixed: tau=1.0, K_inner=10, epochs=60, pgd_steps=20, coordinated=False, lambda_max=1.5,
  radii_mode='uniform' (main), lambda_init=0.0 (paper spec; ablation-only elsewhere)
- Every row records full provenance: k_inner, tau, radii_mode, lambda_init, coordinated,
  pgd_steps, n_seeds_planned, epochs + the usual keys + acc/dp/if_clean, total_time.
- Resume-safe, incremental append+save.
- Companion: use --radii_mode empirical to target results/canonical_tau1_empirical.json
  (only after B posts "src frozen"; see §4,§8).

Usage (after src frozen):
    python experiments/run_canonical.py                 # uniform -> canonical_tau1.json , 6 seeds full grid
    python experiments/run_canonical.py --radii_mode empirical
    python experiments/run_canonical.py --smoke         # tiny test grid to results/canonical_smoke_tau1.json

Agent A: sole launcher. Always ps aux | grep run_ before starting a writer for a file.
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
    parser.add_argument('--methods', nargs='+', default=['naive', 'dro'])
    parser.add_argument('--n_seeds', type=int, default=6)
    parser.add_argument('--radii_mode', type=str, default='uniform', choices=['uniform', 'empirical'],
                        help=" 'uniform' (main canonical_tau1) or 'empirical' (companion for Q5)")
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke: 1 dataset, 1 alpha=0.2, n_seeds=1, small epochs/k for quick test (writes separate _smoke file)')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--k_inner', type=int, default=10)
    parser.add_argument('--pgd_steps', type=int, default=20)
    args = parser.parse_args()

    # Hard constants per §0 GLOBAL HARD CONSTRAINTS + §3
    TAU = 1.0
    LAMBDA_INIT = 0.0
    COORDINATED = False
    LAMBDA_MAX = 1.5  # used inside run_single / trainer
    N_SEEDS_PLANNED = args.n_seeds

    if args.smoke:
        args.datasets = ['adult']
        args.alphas = [0.2]
        args.n_seeds = 1
        args.epochs = 5
        args.k_inner = 3
        args.pgd_steps = 3
        N_SEEDS_PLANNED = 1
        print("SMOKE MODE for canonical runner test (tiny grid, reduced steps/epochs).")
        print("NOTE: real canonical uses full 60ep/10K/20pgd/6seeds per spec. Smoke writes to separate file.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Canonical config (fixed): tau={TAU}, k_inner={args.k_inner}, epochs={args.epochs}, "
          f"pgd_steps={args.pgd_steps}, coordinated={COORDINATED}, lambda_init={LAMBDA_INIT}, "
          f"radii_mode={args.radii_mode}, n_seeds_planned={N_SEEDS_PLANNED}")
    print(f"Grid: datasets={args.datasets} alphas={args.alphas} attacks={args.attacks} methods={args.methods} seeds=0..{args.n_seeds-1}")

    os.makedirs('results', exist_ok=True)
    if args.radii_mode == 'empirical':
        base = 'canonical_tau1_empirical'
    else:
        base = 'canonical_tau1'
    if args.smoke:
        results_path = f'results/{base}_smoke.json'
    else:
        results_path = f'results/{base}.json'

    # Resume support: load existing
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

    # Completed keys (robust even if old rows lack some fields)
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
                            print(f"[{count}/{total}] SKIP (already done): {dataset} α={alpha} seed={seed} attack={attack} method={method}")
                            continue

                        label = f"[{count}/{total}] {dataset} α={alpha} seed={seed} attack={attack} method={method}"
                        print(label)

                        try:
                            t0 = time.time()
                            result = run_single_experiment(
                                dataset, alpha, seed, attack, method, device=device, verbose=False,
                                epochs=args.epochs, k_inner=args.k_inner, pgd_steps=args.pgd_steps,
                                tau=TAU,  # FIXED tau=1.0 per verified headline and hard constraints (ignore stepped get_temperature)
                                lambda_init=LAMBDA_INIT,
                                radii_mode=args.radii_mode,
                                coordinated=COORDINATED,
                                n_seeds_planned=N_SEEDS_PLANNED
                            )
                            elapsed = time.time() - t0
                            all_results.append(result)
                            completed_keys.add(key)
                            save_incremental()

                            print(f"  → acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} "
                                  f"if={result['if_clean']:.4f} ({elapsed:.0f}s)")

                        except Exception as e:
                            print(f"  → FAILED: {e}")
                            import traceback
                            traceback.print_exc()

    print(f"\nSkipped {skipped} already-completed experiments")
    print(f"Saved {len(all_results)} results to {results_path}")

    # Evidence: confirm provenance on last row if any
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
