#!/usr/bin/env python3
"""Quick analysis of partial fairness PGD results."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


def load_all_results():
    """Load and merge per-dataset result files."""
    all_results = []
    for dataset in ['adult', 'credit', 'lsac']:
        path = f'results/fairness_pgd_{dataset}.json'
        if os.path.exists(path):
            with open(path) as f:
                all_results.extend(json.load(f))
    return all_results


def summarize(results):
    """Print summary table grouped by dataset, alpha, attack, method."""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for r in results:
        key = (r['dataset'], r['alpha'], r['attack'], r['method'])
        groups[key].append(r)
    
    print("\n" + "="*80)
    print("Partial Results Summary")
    print("="*80)
    print(f"Total results: {len(results)}")
    print(f"Unique configs: {len(groups)}")
    
    # Count by dataset
    by_ds = defaultdict(int)
    for r in results:
        by_ds[r['dataset']] += 1
    print(f"By dataset: {dict(by_ds)}")
    
    print("\n" + "-"*80)
    print(f"{'Dataset':<8} {'Alpha':<5} {'Attack':<9} {'Method':<6} {'N':<3} {'Acc':<8} {'DP':<10} {'IF':<10}")
    print("-"*80)
    
    for key in sorted(groups.keys()):
        dataset, alpha, attack, method = key
        entries = groups[key]
        n = len(entries)
        accs = [e['acc_clean'] for e in entries]
        dps = [e['dp_clean'] for e in entries]
        ifs = [e['if_clean'] for e in entries]
        
        acc_mean = np.mean(accs)
        dp_mean = np.mean(dps)
        if_mean = np.mean(ifs)
        
        print(f"{dataset:<8} {alpha:<5.1f} {attack:<9} {method:<6} {n:<3} "
              f"{acc_mean:.4f}   {dp_mean:.6f}   {if_mean:.6f}")
    
    print("-"*80)
    
    # DRO vs Naive comparison where both exist
    print("\nDRO vs Naive (where both have data):")
    print("-"*80)
    print(f"{'Dataset':<8} {'Alpha':<5} {'Attack':<9} {'Naive DP':<12} {'DRO DP':<12} {'Change':<10}")
    print("-"*80)
    
    for key in sorted(groups.keys()):
        dataset, alpha, attack, method = key
        naive_key = (dataset, alpha, attack, 'naive')
        dro_key = (dataset, alpha, attack, 'dro')
        
        if naive_key in groups and dro_key in groups:
            naive_dp = np.mean([e['dp_clean'] for e in groups[naive_key]])
            dro_dp = np.mean([e['dp_clean'] for e in groups[dro_key]])
            change = dro_dp - naive_dp
            pct = (change / naive_dp * 100) if naive_dp > 0.001 else 0
            print(f"{dataset:<8} {alpha:<5.1f} {attack:<9} {naive_dp:.6f}     {dro_dp:.6f}     {pct:+.1f}%")
    
    print("-"*80)


def main():
    results = load_all_results()
    if not results:
        print("No results found yet.")
        return
    summarize(results)


if __name__ == '__main__':
    main()
