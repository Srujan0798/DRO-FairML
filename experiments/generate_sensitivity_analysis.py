"""Generate sensitivity analysis figure: DRO vs Naive across all alphas and attacks."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

# Load tabular results
with open('results/fairness_pgd_results.json') as f:
    results = json.load(f)

# Group by (dataset, attack, alpha, method)
from collections import defaultdict
by_config = defaultdict(list)
for r in results:
    key = (r['dataset'], r['attack'], r['alpha'], r['method'])
    by_config[key].append(r['dp_clean'])

datasets = ['adult', 'credit', 'lsac']
attacks = ['dp', 'if', 'combined']
alphas = [0.1, 0.2, 0.3]

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

colors = {'naive': '#c44e2b', 'dro': '#1a7a3a'}
markers = {'naive': 'o', 'dro': 's'}

for i, attack in enumerate(attacks):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]
        for method in ['naive', 'dro']:
            means = []
            stds = []
            for alpha in alphas:
                key = (dataset, attack, alpha, method)
                if key in by_config:
                    vals = by_config[key]
                    means.append(np.mean(vals))
                    stds.append(np.std(vals) / np.sqrt(len(vals)))
                else:
                    means.append(np.nan)
                    stds.append(0)
            valid = [(a, m, s) for a, m, s in zip(alphas, means, stds) if not np.isnan(m)]
            if valid:
                a_vals, m_vals, s_vals = zip(*valid)
                ax.errorbar(a_vals, m_vals, yerr=s_vals, label=method,
                           color=colors[method], marker=markers[method],
                           linewidth=2, markersize=7, capsize=4)
        
        ax.set_title(f'{dataset.upper()} — {attack.upper()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('α', fontsize=10)
        ax.set_ylabel('DP Violation', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0 and j == 2:
            ax.legend(fontsize=9)

plt.suptitle('Sensitivity Analysis: DRO vs Naive across All Configurations', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/sensitivity_all_configs.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/sensitivity_all_configs.pdf', bbox_inches='tight', facecolor='white')
print('Saved sensitivity_all_configs.png/pdf')
