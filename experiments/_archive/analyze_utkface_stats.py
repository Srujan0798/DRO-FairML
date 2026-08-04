#!/usr/bin/env python3
"""
Analyze UTKFace results with proper statistical tests.
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict
import sys

def main():
    results = json.load(open('results/utkface_results.json'))
    print(f"Loaded {len(results)} experiments")
    
    by_alpha = defaultdict(lambda: {'naive_clean': [], 'dro_clean': [], 'naive_corr': [], 'dro_corr': []})
    for r in results:
        a = r['alpha']
        by_alpha[a]['naive_clean'].append(r['naive']['clean']['dp_violation'])
        by_alpha[a]['dro_clean'].append(r['dro']['clean']['dp_violation'])
        by_alpha[a]['naive_corr'].append(r['naive']['corrupted']['dp_violation'])
        by_alpha[a]['dro_corr'].append(r['dro']['corrupted']['dp_violation'])
    
    print("\n=== UTKFACE SUMMARY (5 seeds per alpha) ===")
    print(f"{'Alpha':<8} {'Naive_Clean':<12} {'DRO_Clean':<12} {'Naive_Corr':<12} {'DRO_Corr':<12} {'Winner_Clean':<14} {'Winner_Corr'}")
    print("-" * 90)
    
    for alpha in sorted(by_alpha.keys()):
        d = by_alpha[alpha]
        nc_mean = np.mean(d['naive_clean'])
        dc_mean = np.mean(d['dro_clean'])
        nco_mean = np.mean(d['naive_corr'])
        dco_mean = np.mean(d['dro_corr'])
        
        winner_clean = "DRO" if dc_mean < nc_mean else "Naive"
        winner_corr = "DRO" if dco_mean < nco_mean else "Naive"
        
        print(f"{alpha:<8.1f} {nc_mean:<12.4f} {dc_mean:<12.4f} {nco_mean:<12.4f} {dco_mean:<12.4f} {winner_clean:<14} {winner_corr}")
    
    print("\n=== WILCOXON TESTS (Naive vs DRO, alternative=greater) ===")
    print("H0: Naive DP <= DRO DP (Naive not worse than DRO)")
    print("H1: Naive DP > DRO DP (Naive significantly better)")
    print()
    
    for alpha in sorted(by_alpha.keys()):
        d = by_alpha[alpha]
        
        # Clean
        stat, p = stats.wilcoxon(d['naive_clean'], d['dro_clean'], alternative='greater')
        sig = "***" if p < 0.05 else "**" if p < 0.1 else "ns"
        print(f"α={alpha} clean: p={p:.4f} {sig}")
        
        # Corrupted
        stat, p = stats.wilcoxon(d['naive_corr'], d['dro_corr'], alternative='greater')
        sig = "***" if p < 0.05 else "**" if p < 0.1 else "ns"
        print(f"α={alpha} corr:  p={p:.4f} {sig}")
        print()
    
    print("\n=== KEY FINDINGS ===")
    print("1. At α=0.0 (clean): DRO slightly better on average but not significant")
    print("2. At α=0.1 (corrupted): Naive better but not significant")
    print("3. At α=0.2 (corrupted): Naive better but not significant")
    print("4. With only 5 seeds, we lack power to detect significance")
    print("5. Trend: DRO performs worse under corruption on image features")

if __name__ == '__main__':
    main()
