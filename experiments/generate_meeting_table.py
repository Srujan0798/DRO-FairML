#!/usr/bin/env python3
"""Generate meeting summary table from fairness PGD results."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict


def load_results():
    path = 'results/fairness_pgd_results.json'
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def generate_table(results):
    groups = defaultdict(list)
    for r in results:
        key = (r['dataset'], r['alpha'], r['attack'], r['method'])
        groups[key].append(r)
    
    datasets = ['adult', 'credit', 'lsac']
    alphas = sorted(set(r['alpha'] for r in results))
    attacks = ['dp', 'if', 'combined']
    
    print("\n## Fairness PGD Results Summary\n")
    print("| Dataset | α | Attack | Naive Acc | Naive DP | DRO Acc | DRO DP | Δ DP | N |")
    print("|---------|---|--------|-----------|----------|---------|--------|------|---|")
    
    for ds in datasets:
        for alpha in alphas:
            for attack in attacks:
                n_key = (ds, alpha, attack, 'naive')
                d_key = (ds, alpha, attack, 'dro')
                
                if n_key not in groups or d_key not in groups:
                    continue
                
                naive = groups[n_key]
                dro = groups[d_key]
                
                n_acc = np.mean([r['acc_clean'] for r in naive])
                n_dp = np.mean([r['dp_clean'] for r in naive])
                n_se = np.std([r['dp_clean'] for r in naive]) / np.sqrt(len(naive)) if len(naive) > 1 else 0
                
                d_acc = np.mean([r['acc_clean'] for r in dro])
                d_dp = np.mean([r['dp_clean'] for r in dro])
                d_se = np.std([r['dp_clean'] for r in dro]) / np.sqrt(len(dro)) if len(dro) > 1 else 0
                
                delta = d_dp - n_dp
                pct = (delta / n_dp * 100) if n_dp > 0.001 else 0
                
                n_total = len(naive) + len(dro)
                
                print(f"| {ds} | {alpha:.1f} | {attack} | {n_acc:.3f} | {n_dp:.4f}±{n_se:.4f} | {d_acc:.3f} | {d_dp:.4f}±{d_se:.4f} | {pct:+.1f}% | {n_total} |")
    
    print()


def main():
    results = load_results()
    if not results:
        print("No results found yet.")
        return
    
    print(f"Results: {len(results)} total")
    generate_table(results)


if __name__ == '__main__':
    main()
