#!/usr/bin/env python3
"""
Validation script (Stream A4 from completion plan).
Checks that DRO-FAIR beats Naive-FAIR on DP in >= 6/9 cells
using Wilcoxon signed-rank test (the official criterion).

CRITICAL FIX: Previous version used mean-based comparison only.
This version uses Wilcoxon p<0.05 AND mean(DRO) < mean(Naive)
to match the report's "Wilcoxon p<0.05" claim.
"""
import json
import sys
import os
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def validate():
    # Try merging first if individual results exist
    if os.path.exists('results/individual') and os.listdir('results/individual'):
        try:
            from experiments.run_robust import merge_all
            merge_all()
        except Exception as e:
            print(f"Note: merge_all failed ({e}), using existing all_results.json")

    if not os.path.exists('results/all_results.json'):
        print("ERROR: results/all_results.json not found. Run experiments first.")
        return False

    results = json.load(open('results/all_results.json'))
    print(f"Total: {len(results)} experiments")

    if len(results) < 150:
        print(f"WARNING: Expected 150, got {len(results)}")

    # ========================================================================
    # WILCOXON-BASED VALIDATION (matches report caption)
    # ========================================================================
    wins = {"dp": 0, "if": 0, "total": 0}
    mean_wins = {"dp": 0, "if": 0}  # For reference only
    acc_drops = []

    print("\n" + "="*70)
    print("WILCOXON VALIDATION (one-sided H1: Naive > DRO, n=10 paired seeds)")
    print("="*70)

    for ds in ['adult', 'credit', 'lsac']:
        for a in [0.1, 0.2, 0.3]:
            sub = [r for r in results if r['dataset'] == ds and abs(r['alpha'] - a) < 1e-6]
            if not sub:
                print(f"{ds} a={a}: NO DATA")
                continue

            n_dp = np.array([r['naive']['clean']['dp_violation'] for r in sub])
            d_dp = np.array([r['dro']['clean']['dp_violation'] for r in sub])
            n_if = np.array([r['naive']['clean']['if_violation'] for r in sub])
            d_if = np.array([r['dro']['clean']['if_violation'] for r in sub])
            n_acc = np.mean([r['naive']['clean']['accuracy'] for r in sub])
            d_acc = np.mean([r['dro']['clean']['accuracy'] for r in sub])

            # Wilcoxon: test if DRO < Naive
            diff_dp = n_dp - d_dp
            diff_if = n_if - d_if

            try:
                _, p_dp = wilcoxon(diff_dp, alternative='greater')
            except Exception:
                p_dp = 1.0
            try:
                _, p_if = wilcoxon(diff_if, alternative='greater')
            except Exception:
                p_if = 1.0

            dp_sig = (p_dp < 0.05) and (np.mean(d_dp) < np.mean(n_dp))
            if_sig = (p_if < 0.05) and (np.mean(d_if) < np.mean(n_if))
            dp_mean_win = np.mean(d_dp) < np.mean(n_dp)
            if_mean_win = np.mean(d_if) < np.mean(n_if)

            if dp_sig:
                wins["dp"] += 1
            if if_sig:
                wins["if"] += 1
            if dp_mean_win:
                mean_wins["dp"] += 1
            if if_mean_win:
                mean_wins["if"] += 1
            wins["total"] += 1
            acc_drops.append(n_acc - d_acc)

            dp_red = (np.mean(n_dp) - np.mean(d_dp)) / np.mean(n_dp) * 100 if np.mean(n_dp) > 0 else 0
            if_red = (np.mean(n_if) - np.mean(d_if)) / np.mean(n_if) * 100 if np.mean(n_if) > 0 else 0

            dp_status = f"SIG_WIN(p={p_dp:.3f})" if dp_sig else (f"mean_win(p={p_dp:.3f})" if dp_mean_win else f"NOT_SIG(p={p_dp:.3f})")
            if_status = f"SIG_WIN(p={p_if:.3f})" if if_sig else (f"mean_win(p={p_if:.3f})" if if_mean_win else f"NOT_SIG(p={p_if:.3f})")

            print(f"{ds:6s} a={a}: DP {np.mean(n_dp):.4f}->{np.mean(d_dp):.4f} ({dp_red:+.1f}%) {dp_status:20s} | "
                  f"IF {np.mean(n_if):.4f}->{np.mean(d_if):.4f} ({if_red:+.1f}%) {if_status:20s} | "
                  f"Acc {n_acc:.4f}->{d_acc:.4f} ({(n_acc-d_acc)*100:.1f}% drop)")

    # Check Credit alpha=0.4 accuracy
    sub04 = [r for r in results if r['dataset'] == 'credit' and abs(r['alpha'] - 0.4) < 1e-6]
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

    print(f"\n{'='*70}")
    print(f"DP WINS (Wilcoxon p<0.05):  {wins['dp']}/{wins['total']}  (need >= 6/9)")
    print(f"IF WINS (Wilcoxon p<0.05):  {wins['if']}/{wins['total']}  (report claims 5/9)")
    print(f"DP WINS (mean-based only):  {mean_wins['dp']}/{wins['total']}  (for reference)")
    print(f"IF WINS (mean-based only):  {mean_wins['if']}/{wins['total']}  (for reference)")
    print(f"Avg accuracy drop: {np.mean(acc_drops)*100:.2f}%")
    print(f"{'='*70}")

    passed = wins['dp'] >= 6
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*70}")
    return passed


if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
