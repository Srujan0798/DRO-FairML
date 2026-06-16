#!/usr/bin/env python3
"""Tau ablation: test fixed tau values across all alphas."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from experiments.run_fairness_pgd import run_single_experiment

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, required=True, help='Fixed temperature value')
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--attacks', nargs='+', default=['dp', 'if', 'combined'])
    parser.add_argument('--methods', nargs='+', default=['naive', 'dro'])
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--k_inner', type=int, default=5)
    parser.add_argument('--pgd_steps', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_path = f'results/tau_ablation_tau{args.tau:.0f}.json'

    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {out_path}")

    completed = {(r['dataset'], r['alpha'], r['seed'], r['attack'], r['method']) for r in results}

    total = len(args.datasets) * len(args.alphas) * args.n_seeds * len(args.attacks) * len(args.methods)
    count = 0

    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                for attack in args.attacks:
                    for method in args.methods:
                        count += 1
                        key = (dataset, alpha, seed, attack, method)
                        if key in completed:
                            print(f"[{count}/{total}] SKIP: {dataset} α={alpha} s={seed} {attack} {method}")
                            continue

                        print(f"[{count}/{total}] {dataset} α={alpha} s={seed} {attack} {method} (tau={args.tau})")
                        try:
                            t0 = time.time()
                            result = run_single_experiment(
                                dataset, alpha, seed, attack, method,
                                device=device, verbose=False,
                                epochs=args.epochs, k_inner=args.k_inner, pgd_steps=args.pgd_steps,
                                tau=args.tau,  # explicit fixed tau (bypasses stepped get_temperature)
                                lambda_init=0.0,
                                radii_mode='uniform',
                                coordinated=False,
                                n_seeds_planned=args.n_seeds
                            )
                            # provenance now injected inside run_single_experiment via _add_provenance
                            elapsed = time.time() - t0
                            results.append(result)
                            completed.add(key)
                            with open(out_path, 'w') as f:
                                json.dump(results, f, indent=2)
                            print(f"  → acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} ({elapsed:.0f}s)")
                        except Exception as e:
                            print(f"  -> FAILED: {e}")

    print(f"\nDone. Saved {len(results)} results to {out_path}")

if __name__ == '__main__':
    main()
