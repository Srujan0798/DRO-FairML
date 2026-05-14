#!/usr/bin/env python3
"""
Validation script (Stream A4 from completion plan).
Checks that DRO-FAIR beats Naive-FAIR on DP in >= 6/9 cells.
Run after experiments complete and results are merged.
"""
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def validate():
    # Try merging first if individual results exist
    if os.path.exists('results/individual') and os.listdir('results/individual'):
        from experiments.run_robust import merge_all
        merge_all()

    if not os.path.exists('results/all_results.json'):
        print("ERROR: results/all_results.json not found. Run experiments first.")
        return False

    results = json.load(open('results/all_results.json'))
    print(f"Total: {len(results)} experiments")

    if len(results) < 150:
        print(f"WARNING: Expected 150, got {len(results)}")

    wins = {"dp": 0, "if": 0, "total": 0}
    acc_drops = []

    for ds in ['adult', 'credit', 'lsac']:
        for a in [0.1, 0.2, 0.3]:
            sub = [r for r in results if r['dataset'] == ds and r['alpha'] == a]
            if not sub:
                print(f"{ds} a={a}: NO DATA")
                continue

            n_dp = np.mean([r['naive']['clean']['dp_violation'] for r in sub])
            d_dp = np.mean([r['dro']['clean']['dp_violation'] for r in sub])
            n_if = np.mean([r['naive']['clean']['if_violation'] for r in sub])
            d_if = np.mean([r['dro']['clean']['if_violation'] for r in sub])
            n_acc = np.mean([r['naive']['clean']['accuracy'] for r in sub])
            d_acc = np.mean([r['dro']['clean']['accuracy'] for r in sub])

            dp_win = d_dp < n_dp
            if_win = d_if < n_if
            if dp_win:
                wins["dp"] += 1
            if if_win:
                wins["if"] += 1
            wins["total"] += 1
            acc_drops.append(n_acc - d_acc)

            dp_red = (n_dp - d_dp) / n_dp * 100 if n_dp > 0 else 0
            if_red = (n_if - d_if) / n_if * 100 if n_if > 0 else 0

            print(f"{ds} a={a}: DP {n_dp:.4f}->{d_dp:.4f} ({dp_red:+.1f}%) {'WIN' if dp_win else 'LOSS'}, "
                  f"IF {n_if:.4f}->{d_if:.4f} ({if_red:+.1f}%) {'WIN' if if_win else 'LOSS'}, "
                  f"Acc {n_acc:.4f}->{d_acc:.4f} ({(n_acc-d_acc)*100:.1f}% drop)")

    # Check Credit alpha=0.4 accuracy
    sub04 = [r for r in results if r['dataset'] == 'credit' and r['alpha'] == 0.4]
    if sub04:
        d_acc04 = np.mean([r['dro']['clean']['accuracy'] for r in sub04])
        print(f"\nCredit a=0.4 DRO acc: {d_acc04:.4f} {'OK' if d_acc04 >= 0.60 else 'WARNING: low'}")

    # Runtime check
    naive_times = [r['naive']['time'] for r in results if 'time' in r.get('naive', {})]
    dro_times = [r['dro']['time'] for r in results if 'time' in r.get('dro', {})]
    if naive_times and dro_times:
        overhead = np.mean(dro_times) / np.mean(naive_times)
        print(f"\nRuntime: Naive={np.mean(naive_times):.1f}s, DRO={np.mean(dro_times):.1f}s, "
              f"Overhead={overhead:.1f}x")

    print(f"\n{'='*50}")
    print(f"DP wins: {wins['dp']}/{wins['total']} (need >= 6/9)")
    print(f"IF wins: {wins['if']}/{wins['total']} (need >= 6/9)")
    print(f"Avg accuracy drop: {np.mean(acc_drops)*100:.2f}% (target: 1-4%)")

    passed = wins['dp'] >= 6
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*50}")
    return passed


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
