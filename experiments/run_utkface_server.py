#!/usr/bin/env python3
"""
Server batch script for UTKFace experiments (FairnessTargetedPGD focus).
Runs dp / if / combined attacks across alphas and seeds on GPU (flair2 ready).

CANONICAL CONFIG (per May 19 task + tau=1 / K_inner=10):
- tau=1.0 (FIXED, not stepped)
- K_inner=10
- pgd_steps=20, epochs=60
- n_seeds=6 (for Wilcoxon power)
- alphas 0.0-0.4 , attacks dp/if/combined

Turnkey on flair2 (after git pull + venv):
    python3 experiments/run_utkface_server.py --n_seeds 6 --tau 1.0 --k_inner 10

Resumes automatically per-attack json. Records full provenance (k_inner/tau/pgd_steps/epochs/...).

See UTKFACE_SERVER_COMMANDS.txt for nohup / tmux / slurm copy-paste lines.
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
    parser = argparse.ArgumentParser(
        description="Turnkey server runner for UTKFace + FairnessTargetedPGD (dp/if/combined). "
                    "Canonical: tau=1 fixed + K_inner=10 + full provenance. Resume-safe.",
        epilog="""Canonical examples (copy-paste ready for flair2 after access):
  # Full grid (recommended: 6 seeds, tau=1, K=10) — one attack at a time for easy monitoring
  python3 experiments/run_utkface_server.py --attack dp --n_seeds 6 --tau 1.0 --k_inner 10 --alphas 0.0 0.1 0.2 0.3 0.4
  python3 experiments/run_utkface_server.py --attack if --n_seeds 6 --tau 1.0 --k_inner 10 --alphas 0.0 0.1 0.2 0.3 0.4
  python3 experiments/run_utkface_server.py --attack combined --n_seeds 6 --tau 1.0 --k_inner 10 --alphas 0.0 0.1 0.2 0.3 0.4

  # All three attacks in one invocation (will write utkface_dp_server.json etc.)
  python3 experiments/run_utkface_server.py --attacks dp if combined --n_seeds 6 --tau 1.0 --k_inner 10

  # Smoke / quick CPU validation (matches the 2-row fairness_pgd smoke schema)
  python3 experiments/run_utkface_server.py --attack dp --n_seeds 1 --alphas 0.2 --tau 1.0 --k_inner 3 --epochs 5 --pgd_steps 3 --device cpu

See also: UTKFACE_SERVER_COMMANDS.txt for nohup/tmux/slurm wrappers.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--datasets', nargs='+', default=['utkface'])
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--attacks', nargs='+',
                        default=['dp', 'if', 'combined'],
                        help='FairnessTargetedPGD attack modes to run (canonical server grid)')
    parser.add_argument('--n_seeds', type=int, default=6,
                        help='Number of seeds (canonical target: 6 for statistical power)')
    parser.add_argument('--device', default='auto',
                        help='auto|cuda|mps|cpu (auto picks cuda>mps>cpu)')
    parser.add_argument('--lambda_max', type=float, default=1.5)
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Fixed temperature (canonical=1.0 for all alphas; do not use stepped)')
    parser.add_argument('--k_inner', type=int, default=10,
                        help='K_inner for DRO (canonical=10)')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--pgd_steps', type=int, default=20)
    parser.add_argument('--coordinated', action='store_true', default=False)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--output', type=str, default=None,
                        help='If set, write ALL attacks into this single JSON (canonical aggregate)')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to auto")
        device = 'mps' if (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()) else 'cpu'
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"UTKFACE SERVER RUNNER (canonical mode)")
    print(f"  tau={args.tau} (fixed)  k_inner={args.k_inner}  epochs={args.epochs}  pgd_steps={args.pgd_steps}")
    print(f"  attacks={args.attacks}  alphas={args.alphas}  n_seeds={args.n_seeds}  device={device}")

    total = len(args.datasets) * len(args.alphas) * args.n_seeds * len(args.attacks)
    completed = 0
    failed = 0

    for dataset in args.datasets:
        for attack in args.attacks:
            if args.output:
                out_path = args.output
            else:
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
                    (r.get('dataset'), r.get('alpha'), r.get('seed'), r.get('attack'))
                    for r in all_results
                }
                print(f"[{dataset}/{attack}] Loaded {len(all_results)} existing results (resume)")

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
                            tau=args.tau, k_inner=args.k_inner, epochs=args.epochs,
                            pgd_steps=args.pgd_steps, coordinated=args.coordinated,
                            n_seeds_planned=args.n_seeds,
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

    # Final provenance sanity on last written result (if any)
    if all_results:
        last = all_results[-1]
        prov_keys = ['k_inner', 'tau', 'pgd_steps', 'epochs', 'n_seeds_planned']
        print("Last row provenance sample:", {k: last.get(k) for k in prov_keys})


if __name__ == '__main__':
    main()
