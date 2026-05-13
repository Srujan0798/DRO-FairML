#!/usr/bin/env python3
"""
Monitor Experiment Progress
============================

Shows how many experiments are completed, which datasets/alphas are done,
and estimates time remaining.

Usage:
    python3 experiments/monitor_progress.py
"""

import os
import sys
import pickle
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    checkpoint_path = 'results/checkpoint.pkl'
    
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found.")
        print("   Experiments may not have started yet.")
        print("   Run: python3 experiments/run_experiments.py --n_seeds 10")
        return 1
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    results = checkpoint['results']
    completed_keys = checkpoint['completed_keys']
    total = 150
    done = len(completed_keys)
    remaining = total - done
    
    # Parse completed experiments
    dataset_counts = Counter()
    alpha_counts = Counter()
    dataset_alpha_counts = defaultdict(Counter)
    
    for key in completed_keys:
        parts = key.split('_')
        if len(parts) >= 3:
            dataset = parts[0]
            alpha = parts[1]
            dataset_counts[dataset] += 1
            alpha_counts[alpha] += 1
            dataset_alpha_counts[dataset][alpha] += 1
    
    # Progress bar
    bar_len = 40
    filled = int(bar_len * done / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    
    print("=" * 60)
    print("EXPERIMENT PROGRESS")
    print("=" * 60)
    print(f"\nProgress: [{bar}] {done}/{total} ({100*done/total:.1f}%)")
    print(f"Remaining: {remaining} experiments")
    
    # Estimate time remaining (very rough)
    if done > 0:
        # Average experiment takes ~120s for DRO + ~15s for Naive = ~135s
        avg_time = 135  # seconds
        est_remaining = remaining * avg_time
        print(f"Est. time remaining: {format_time(est_remaining)}")
    
    print(f"\n{'='*60}")
    print("BY DATASET")
    print(f"{'='*60}")
    for ds in ['adult', 'credit', 'lsac']:
        count = dataset_counts.get(ds, 0)
        expected = 50  # 5 alphas × 10 seeds
        print(f"  {ds:8s}: {count:3d}/{expected}  ({100*count/expected:.0f}%)")
    
    print(f"\n{'='*60}")
    print("BY ALPHA (across all datasets)")
    print(f"{'='*60}")
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
        count = alpha_counts.get(str(alpha), 0)
        expected = 30  # 3 datasets × 10 seeds
        print(f"  α={alpha}: {count:3d}/{expected}  ({100*count/expected:.0f}%)")
    
    print(f"\n{'='*60}")
    print("DETAILED MATRIX")
    print(f"{'='*60}")
    print(f"{'Dataset':8s} {'α=0.0':>6s} {'α=0.1':>6s} {'α=0.2':>6s} {'α=0.3':>6s} {'α=0.4':>6s}")
    print("-" * 50)
    for ds in ['adult', 'credit', 'lsac']:
        row = f"{ds:8s}"
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
            count = dataset_alpha_counts[ds].get(str(alpha), 0)
            expected = 10
            status = f"{count}/{expected}"
            if count == expected:
                status = "✅"
            elif count == 0:
                status = "⬜"
            row += f" {status:>5s}"
        print(row)
    
    # Check if all_results.json exists
    if os.path.exists('results/all_results.json'):
        import json
        with open('results/all_results.json') as f:
            all_results = json.load(f)
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"  results/all_results.json: {len(all_results)} entries")
        if len(all_results) == 150:
            print("  🎉 ALL EXPERIMENTS COMPLETE!")
    
    print(f"\n{'='*60}")
    if done == total:
        print("✅ READY FOR DELIVERABLES")
        print("   Run: python3 experiments/generate_all_deliverables.py")
    elif done == 0:
        print("⏳ NOT STARTED")
        print("   Run: python3 experiments/run_experiments.py --n_seeds 10")
    else:
        print(f"⏳ IN PROGRESS ({done}/{total})")
        print("   Keep waiting, or re-run if process died:")
        print("   python3 experiments/run_experiments.py --n_seeds 10")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
