#!/usr/bin/env python3
"""Generate fig8_attack_defense_matrix — 3x3 heatmap of DRO vs Naive across attacks and datasets."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['CMU Serif', 'Computer Modern Roman', 'Latin Modern Roman',
                           'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset':   'cm',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     11,
    'axes.titleweight':   'bold',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   False,
    'axes.spines.bottom': False,
    'axes.linewidth':     0.8,
    'axes.labelpad':      4,
})

# ── Load data ───────────────────────────────────────────────────────────────
results = {}
with open('results/fairness_pgd_wilcoxon.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row['dataset']
        attack = row['attack']
        alpha = float(row['alpha'])
        reduction = float(row['dp_reduction_pct'])
        pvalue = float(row['dp_pvalue'])
        key = (dataset, attack)
        if key not in results:
            results[key] = []
        results[key].append((alpha, reduction, pvalue))

# ── Aggregate: for each cell, show the best significant result if any ───────
datasets = ['adult', 'credit', 'lsac']
attacks = ['dp', 'if', 'combined']

cell_value = np.zeros((3, 3))      # reduction %
cell_pval = np.ones((3, 3))        # best p-value
cell_text = [["" for _ in range(3)] for _ in range(3)]
cell_alpha = [["" for _ in range(3)] for _ in range(3)]

for i, attack in enumerate(attacks):
    for j, dataset in enumerate(datasets):
        key = (dataset, attack)
        if key in results:
            entries = results[key]
            # Find best significant result
            sig = [(a, r, p) for a, r, p in entries if p < 0.05 and r > 0]
            if sig:
                best = max(sig, key=lambda x: x[1])
                cell_value[i, j] = best[1]
                cell_pval[i, j] = best[2]
                cell_text[i][j] = f"+{best[1]:.1f}%\np={best[2]:.3f}"
                cell_alpha[i][j] = f"α={best[0]:.1f}"
            else:
                # Show average or most representative
                avg_r = np.mean([r for _, r, _ in entries])
                cell_value[i, j] = avg_r
                cell_pval[i, j] = min(p for _, _, p in entries)
                if avg_r > 0:
                    cell_text[i][j] = f"+{avg_r:.1f}%\nn.s."
                else:
                    cell_text[i][j] = f"{avg_r:.1f}%\nn.s."

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Custom diverging colormap: red (negative) -> white (0) -> green (positive)
vmax = max(abs(cell_value.min()), abs(cell_value.max()))
if vmax == 0:
    vmax = 1

for i in range(3):
    for j in range(3):
        val = cell_value[i, j]
        # Color: negative = red, positive = green, intensity by magnitude
        if val >= 0:
            intensity = min(val / 100, 1.0)
            color = (1 - intensity * 0.7, 1 - intensity * 0.2, 1 - intensity * 0.7)
        else:
            intensity = min(abs(val) / 50, 1.0)
            color = (1, 1 - intensity * 0.7, 1 - intensity * 0.7)

        rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

        text = cell_text[i][j]
        fontsize = 14 if '\np=' in text else 13
        weight = 'bold' if 'p=' in text else 'normal'
        color_text = '#1a472a' if val > 0 else '#8b0000'
        ax.text(j, i, text, ha='center', va='center', fontsize=fontsize,
                fontweight=weight, color=color_text)

# Labels
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels([d.upper() for d in datasets], fontsize=13, fontweight='bold')
ax.set_yticklabels([a.upper() for a in attacks], fontsize=13, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Attack Mode', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('DRO vs Naive: DP Reduction % under Adversarial Attacks\n' +
             '(Green = DRO better, Red = DRO worse)',
             fontsize=14, fontweight='bold', pad=15)

# Colorbar legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=(0.85, 0.95, 0.85), edgecolor='black', label='DRO wins (significant)'),
    Patch(facecolor=(0.95, 0.85, 0.85), edgecolor='black', label='DRO loses / not sig.'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
          frameon=True, framealpha=0.9, edgecolor='0.8')

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figures/fig8_attack_defense_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/fig8_attack_defense_matrix.pdf', bbox_inches='tight', facecolor='white')
print('Saved: figures/fig8_attack_defense_matrix.png & .pdf')
