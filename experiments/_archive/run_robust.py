#!/usr/bin/env python3
"""
Crash-resilient experiment runner.
Runs ONE experiment at a time, saves to individual JSON files.
If killed, only loses the current experiment — all completed ones are safe.

Usage:
    python3 experiments/run_robust.py
    # Or for a specific dataset:
    python3 experiments/run_robust.py --datasets adult
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.run_experiments import run_single_experiment

DATASETS = ['adult', 'credit', 'lsac']
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_SEEDS = 10
RESULTS_DIR = 'results/individual'


def get_completed():
    """Scan individual result files to find what's done."""
    completed = set()
    if not os.path.exists(RESULTS_DIR):
        return completed
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('.json'):
            key = fname.replace('.json', '')
            completed.add(key)
    return completed


def save_result(result):
    """Save one result to its own JSON file (atomic)."""
    key = f"{result['dataset']}_{result['alpha']}_{result['seed']}"
    path = os.path.join(RESULTS_DIR, f"{key}.json")
    # Write to temp file first, then rename (atomic on most filesystems)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(result, f, indent=2)
    os.rename(tmp_path, path)
    return key


def merge_all():
    """Merge all individual results into all_results.json."""
    all_results = []
    if not os.path.exists(RESULTS_DIR):
        return all_results
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                all_results.append(json.load(f))
    # Save merged
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--alphas', type=float, nargs='+', default=ALPHAS)
    parser.add_argument('--n_seeds', type=int, default=N_SEEDS)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    completed = get_completed()

    total = len(args.datasets) * len(args.alphas) * args.n_seeds
    remaining = total - len(completed)
    print(f"Completed: {len(completed)}/{total}, Remaining: {remaining}")

    count = 0
    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                key = f"{dataset}_{alpha}_{seed}"
                if key in completed:
                    continue

                count += 1
                print(f"\n[{len(completed)+count}/{total}] {dataset} α={alpha} seed={seed}")

                try:
                    t0 = time.time()
                    result = run_single_experiment(dataset, alpha, seed, device='cpu', verbose=False)
                    elapsed = time.time() - t0

                    saved_key = save_result(result)

                    n_dp = result['naive']['clean']['dp_violation']
                    d_dp = result['dro']['clean']['dp_violation']
                    d_acc = result['dro']['clean']['accuracy']
                    win = "WIN" if d_dp < n_dp else "LOSS"
                    print(f"  Done in {elapsed:.0f}s | Naive DP={n_dp:.4f} DRO DP={d_dp:.4f} DRO Acc={d_acc:.4f} {win}")

                    # Force garbage collection to prevent memory buildup
                    gc.collect()

                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # Final merge
    print("\nMerging all results...")
    all_results = merge_all()
    print(f"Total: {len(all_results)} experiments saved to results/all_results.json")


if __name__ == '__main__':
    main()