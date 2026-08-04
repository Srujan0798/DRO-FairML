#!/usr/bin/env python3
"""Detailed analysis of complete LSAC dataset (90/90 results)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict


def load_results():
    with open('results/fairness_pgd_lsac.json') as f:
        return json.load(f)


def analyze(results):
    groups = defaultdict(list)
    for r in results:
        key = (r['alpha'], r['attack'], r['method'])
        groups[key].append(r)
    
    print("=" * 70)
    print("LSAC Complete Analysis (90/90 results)")
    print("=" * 70)
    
    alphas = sorted(set(a for a, _, _ in groups.keys()))
    attacks = ['dp', 'if', 'combined']
    
    for alpha in alphas:
        print(f"\n--- α = {alpha} ---")
        print(f"{'Attack':>8} | {'Naive Acc':>10} | {'Naive DP':>10} | {'DRO Acc':>10} | {'DRO DP':>10} | {'Δ DP':>8} | {'p (t)':>8}")
        print("-" * 80)
        
        for attack in attacks:
            n_key = (alpha, attack, 'naive')
            d_key = (alpha, attack, 'dro')
            
            if n_key not in groups or d_key not in groups:
                continue
            
            naive = groups[n_key]
            dro = groups[d_key]
            
            n_acc = [r['acc_clean'] for r in naive]
            n_dp = [r['dp_clean'] for r in naive]
            d_acc = [r['acc_clean'] for r in dro]
            d_dp = [r['dp_clean'] for r in dro]
            
            # Paired t-test (same seeds)
            from scipy import stats
            t_stat, p_val = stats.ttest_rel(n_dp, d_dp)
            
            delta = np.mean(d_dp) - np.mean(n_dp)
            
            print(f"{attack:>8} | {np.mean(n_acc):>10.4f} | {np.mean(n_dp):>10.6f} | "
                  f"{np.mean(d_acc):>10.4f} | {np.mean(d_dp):>10.6f} | "
                  f"{delta:>+8.6f} | {p_val:>8.4f}")
    
    print("\n" + "=" * 70)
    print("Summary: DRO wins when Δ DP < 0 and p < 0.05")
    print("=" * 70)


if __name__ == '__main__':
    results = load_results()
    analyze(results)
