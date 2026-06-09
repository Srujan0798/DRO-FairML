#!/usr/bin/env python3
"""
Server batch script for UTKFace experiments.
Runs FairnessTargetedPGD attacks (dp, if, combined) across alphas and seeds.
Designed for GPU server execution.

Usage:
    python3 experiments/run_utkface_server.py --attack dp --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 5
    python3 experiments/run_utkface_server.py --attack combined --device cuda
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import argparse
import torch
from experiments.run_utkface import run_single_utkface_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['utkface'])
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--attacks', nargs='+',
                        default=['dp', 'if', 'combined'],
                        help='FairnessTargetedPGD attack modes to run')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lambda_max', type=float, default=1.5)
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print("WARNING: CUDA not available, falling back to CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(args.datasets) * len(args.alphas) * args.n_seeds * len(args.attacks)
    completed = 0
    failed = 0

    for dataset in args.datasets:
        for attack in args.attacks:
            out_path = os.path.join(
                args.output_dir,
                f'utkface_{attack}_server.json'
            )

            # Load existing progress
            all_results = []
            completed_keys = set()
            if os.path.exists(out_path):
                with open(out_path) as f:
                    all_results = json.load(f)
                completed_keys = {
                    (r['dataset'], r['alpha'], r['seed'], r['attack'])
                    for r in all_results
                }
                print(f"[{dataset}/{attack}] Loaded {len(all_results)} existing results")

            def save():
                with open(out_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

            for alpha in args.alphas:
                for seed in range(args.n_seeds):
                    key = (dataset, alpha, seed, attack)
                    if key in completed_keys:
                        completed += 1
                        continue

                    label = f"[{dataset}/{attack}] α={alpha} seed={seed}"
                    print(f"\n{label}")
                    try:
                        t0 = time.time()
                        result = run_single_utkface_experiment(
                            dataset, alpha, seed, device=device, verbose=False,
                            lambda_max=args.lambda_max, attack=attack,
                        )
                        elapsed = time.time() - t0
                        all_results.append(result)
                        completed_keys.add(key)
                        save()
                        completed += 1
                        print(f"  Done in {elapsed:.0f}s | "
                              f"Naive clean: acc={result['naive']['clean']['accuracy']:.3f} "
                              f"dp={result['naive']['clean']['dp_violation']:.3f} | "
                              f"DRO clean: acc={result['dro']['clean']['accuracy']:.3f} "
                              f"dp={result['dro']['clean']['dp_violation']:.3f}")
                    except Exception as e:
                        failed += 1
                        print(f"  FAILED: {e}")
                        import traceback
                        traceback.print_exc()

            print(f"[{dataset}/{attack}] Saved {len(all_results)} results to {out_path}")

    print(f"\n{'='*60}")
    print(f"SERVER RUN COMPLETE")
    print(f"Total: {total} | Completed: {completed} | Failed: {failed}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
