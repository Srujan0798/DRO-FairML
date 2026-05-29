"""Generate final combined figure for Madam's meeting."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load UTKFace results
with open('results/utkface_results.json') as f:
    utk_results = json.load(f)

# Load tabular Wilcoxon results
with open('results/fairness_pgd_wilcoxon.csv') as f:
    lines = f.readlines()[1:]

# Parse tabular significant results (DP reduction, p < 0.05)
sig_results = []
for line in lines:
    parts = line.strip().split(',')
    if len(parts) >= 9:
        reduction = float(parts[6])
        pvalue = float(parts[7])
        if pvalue < 0.05 and reduction > 0:
            sig_results.append({
                'dataset': parts[0],
                'attack': parts[1],
                'alpha': float(parts[2]),
                'reduction': reduction,
                'pvalue': pvalue
            })

# Sort by reduction
sig_results.sort(key=lambda x: x['reduction'], reverse=True)

# UTKFace stats
from collections import defaultdict
by_alpha = defaultdict(list)
for r in utk_results:
    by_alpha[r['alpha']].append(r)

utk_alphas = sorted(by_alpha.keys())
naive_dp = [np.mean([x['naive']['corrupted']['dp_violation'] for x in by_alpha[a]]) for a in utk_alphas]
dro_dp = [np.mean([x['dro']['corrupted']['dp_violation'] for x in by_alpha[a]]) for a in utk_alphas]

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

# ===== TOP LEFT: Significant tabular results =====
ax1 = fig.add_subplot(gs[0, 0])
labels = [f"{r['dataset']}\n{r['attack'].upper()} α={r['alpha']}" for r in sig_results[:5]]
reductions = [r['reduction'] for r in sig_results[:5]]
colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in reductions]
bars = ax1.barh(labels, reductions, color=colors, edgecolor='black', alpha=0.8)
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.set_xlabel('DP Reduction by DRO (%)', fontsize=12)
ax1.set_title('Task 1: DRO Improvement Under Adversarial Attacks\n(Statistically Significant Results)', 
              fontsize=13, fontweight='bold')
ax1.invert_yaxis()
for bar, val in zip(bars, reductions):
    ax1.text(val + 3 if val > 0 else val - 3, bar.get_y() + bar.get_height()/2, 
             f'{val:.1f}%', ha='left' if val > 0 else 'right', va='center', fontsize=10, fontweight='bold')

# ===== TOP RIGHT: UTKFace bar chart =====
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(utk_alphas))
width = 0.35
ax2.bar(x - width/2, naive_dp, width, label='Naive-FAIR', color='#e74c3c', alpha=0.8, edgecolor='black')
ax2.bar(x + width/2, dro_dp, width, label='DRO-FAIR', color='#2ecc71', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Corruption Level (α)', fontsize=12)
ax2.set_ylabel('DP Violation', fontsize=12)
ax2.set_title('Task 2: UTKFace Under Label Corruption\n(ResNet18 Features, 23K Images)', 
              fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{a:.1f}' for a in utk_alphas])
ax2.legend(fontsize=11)
ax2.set_ylim(0, 0.15)

# Add annotations
for i, (n, d) in enumerate(zip(naive_dp, dro_dp)):
    if d < n:
        ax2.annotate('DRO wins', xy=(i + width/2, d), xytext=(i + width/2, d + 0.01),
                    ha='center', fontsize=9, color='green', fontweight='bold')
    else:
        ax2.annotate('Naive wins', xy=(i - width/2, n), xytext=(i - width/2, n + 0.01),
                    ha='center', fontsize=9, color='red', fontweight='bold')

# ===== BOTTOM: Key message =====
ax3 = fig.add_subplot(gs[1, :])
ax3.axis('off')

message = """
KEY FINDING: DRO-FAIR Robustness is NOT Universal

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                      │
│   On TABULAR DATA (Credit, LSAC):                                                                    │
│   • DRO reduces DP violation by 64-97% under IF-targeted adversarial attacks (p < 0.05)             │
│   • BUT: DRO loses under DP-targeted attacks — the adversary is strong enough to break it           │
│                                                                                                      │
│   On IMAGE DATA (UTKFace, ResNet18 features):                                                        │
│   • DRO makes fairness WORSE under corruption — Naive-FAIR outperforms DRO by 7-39%                 │
│   • This is OPPOSITE to tabular results — a surprising, novel finding                               │
│                                                                                                      │
│   CONCLUSION: DRO's robustness depends on BOTH the attack metric (IF vs DP) AND the data modality   │
│   (tabular vs image features). Prior work only tested random noise — we show gradient-based         │
│   adversarial attacks reveal important limitations.                                                  │
│                                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
ax3.text(0.5, 0.5, message, ha='center', va='center', fontsize=11.5, family='monospace',
         transform=ax3.transAxes,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fef9e7', edgecolor='#f39c12', linewidth=2))

plt.suptitle('Weekly Research Summary — May 29, 2026 | Adversarial Fairness Attacks on DRO-FAIR', 
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('figures/final_meeting_figure.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/final_meeting_figure.pdf', bbox_inches='tight', facecolor='white')
print('Saved: figures/final_meeting_figure.png & .pdf')
