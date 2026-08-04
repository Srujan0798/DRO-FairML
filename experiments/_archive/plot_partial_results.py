#!/usr/bin/env python3
"""Plot partial results showing DRO vs Naive across alphas per dataset."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def load_results():
    from experiments.loaders import load_fairness_pgd_results
    return load_fairness_pgd_results()


def plot_dataset(ax, results, dataset_name, metric='dp_clean'):
    """Plot DRO vs Naive for one dataset."""
    groups = defaultdict(list)
    for r in results:
        if r['dataset'] == dataset_name:
            key = (r['alpha'], r['attack'], r['method'])
            groups[key].append(r)
    
    alphas = sorted(set(a for a, _, _ in groups.keys()))
    attacks = ['dp', 'if', 'combined']
    colors = {'dp': 'tab:blue', 'if': 'tab:orange', 'combined': 'tab:green'}
    markers = {'naive': 'o', 'dro': 's'}
    
    for attack in attacks:
        naive_vals = []
        dro_vals = []
        valid_alphas = []
        
        for alpha in alphas:
            n_key = (alpha, attack, 'naive')
            d_key = (alpha, attack, 'dro')
            if n_key in groups and d_key in groups:
                n_vals = [x[metric] for x in groups[n_key]]
                d_vals = [x[metric] for x in groups[d_key]]
                naive_vals.append(np.mean(n_vals))
                dro_vals.append(np.mean(d_vals))
                valid_alphas.append(alpha)
        
        if valid_alphas:
            ax.plot(valid_alphas, naive_vals, marker='o', linestyle='--', 
                   color=colors[attack], label=f'{attack} naive', alpha=0.7)
            ax.plot(valid_alphas, dro_vals, marker='s', linestyle='-', 
                   color=colors[attack], label=f'{attack} DRO', alpha=0.9)
    
    ax.set_xlabel('α')
    ax.set_ylabel('DP Violation')
    ax.set_title(dataset_name.upper())
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def main():
    results = load_results()
    if not results:
        print("No results found.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    datasets = ['adult', 'credit', 'lsac']
    
    for ax, ds in zip(axes, datasets):
        plot_dataset(ax, results, ds)
    
    plt.tight_layout()
    fig.savefig('figures/partial_results_dp.png', dpi=150)
    fig.savefig('figures/partial_results_dp.pdf')
    print("Saved figures/partial_results_dp.{png,pdf}")


if __name__ == '__main__':
    main()
