"""Analyze UTKFace results and generate figures."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
with open('results/utkface_results.json') as f:
    results = json.load(f)

# Group by alpha
from collections import defaultdict
by_alpha = defaultdict(list)
for r in results:
    by_alpha[r['alpha']].append(r)

# Extract stats
alphas = sorted(by_alpha.keys())
naive_clean_dp = []
dro_clean_dp = []
naive_corr_dp = []
dro_corr_dp = []
naive_clean_acc = []
dro_clean_acc = []
naive_corr_acc = []
dro_corr_acc = []

for alpha in alphas:
    items = by_alpha[alpha]
    naive_clean_dp.append(np.mean([x['naive']['clean']['dp_violation'] for x in items]))
    dro_clean_dp.append(np.mean([x['dro']['clean']['dp_violation'] for x in items]))
    naive_corr_dp.append(np.mean([x['naive']['corrupted']['dp_violation'] for x in items]))
    dro_corr_dp.append(np.mean([x['dro']['corrupted']['dp_violation'] for x in items]))
    naive_clean_acc.append(np.mean([x['naive']['clean']['accuracy'] for x in items]))
    dro_clean_acc.append(np.mean([x['dro']['clean']['accuracy'] for x in items]))
    naive_corr_acc.append(np.mean([x['naive']['corrupted']['accuracy'] for x in items]))
    dro_corr_acc.append(np.mean([x['dro']['corrupted']['accuracy'] for x in items]))

# Figure 1: DP violation comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(alphas))
width = 0.35

# Clean
ax = axes[0]
ax.bar(x - width/2, naive_clean_dp, width, label='Naive-FAIR', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, dro_clean_dp, width, label='DRO-FAIR', color='#2ecc71', alpha=0.8)
ax.set_xlabel('Corruption Level (α)', fontsize=12)
ax.set_ylabel('DP Violation', fontsize=12)
ax.set_title('UTKFace: Clean Data', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{a:.1f}' for a in alphas])
ax.legend()
ax.set_ylim(0, 0.15)

# Corrupted
ax = axes[1]
ax.bar(x - width/2, naive_corr_dp, width, label='Naive-FAIR', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, dro_corr_dp, width, label='DRO-FAIR', color='#2ecc71', alpha=0.8)
ax.set_xlabel('Corruption Level (α)', fontsize=12)
ax.set_ylabel('DP Violation', fontsize=12)
ax.set_title('UTKFace: Corrupted Data (α budget)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{a:.1f}' for a in alphas])
ax.legend()
ax.set_ylim(0, 0.15)

plt.tight_layout()
plt.savefig('figures/fig_utkface_dp_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_utkface_dp_comparison.pdf', bbox_inches='tight')
print('Saved figures/fig_utkface_dp_comparison.png')

# Figure 2: Accuracy vs Fairness tradeoff
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(naive_corr_acc, naive_corr_dp, s=200, marker='o', label='Naive-FAIR', color='#e74c3c', alpha=0.7, edgecolors='black')
ax.scatter(dro_corr_acc, dro_corr_dp, s=200, marker='s', label='DRO-FAIR', color='#2ecc71', alpha=0.7, edgecolors='black')

for i, alpha in enumerate(alphas):
    ax.annotate(f'α={alpha}', (naive_corr_acc[i], naive_corr_dp[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax.annotate(f'α={alpha}', (dro_corr_acc[i], dro_corr_dp[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

ax.set_xlabel('Accuracy', fontsize=12)
ax.set_ylabel('DP Violation', fontsize=12)
ax.set_title('UTKFace: Accuracy vs Fairness Tradeoff (Corrupted)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_utkface_tradeoff.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_utkface_tradeoff.pdf', bbox_inches='tight')
print('Saved figures/fig_utkface_tradeoff.png')

# Print summary table
print('\n' + '='*70)
print('UTKFACE SUMMARY TABLE')
print('='*70)
print(f"{'α':<6} {'Naive Clean DP':<15} {'DRO Clean DP':<15} {'Naive Corr DP':<15} {'DRO Corr DP':<15} {'Winner':<10}")
print('-'*70)
for i, alpha in enumerate(alphas):
    winner = 'DRO' if dro_corr_dp[i] < naive_corr_dp[i] else 'Naive'
    print(f"{alpha:<6.1f} {naive_clean_dp[i]:<15.4f} {dro_clean_dp[i]:<15.4f} {naive_corr_dp[i]:<15.4f} {dro_corr_dp[i]:<15.4f} {winner:<10}")
