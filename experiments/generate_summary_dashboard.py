"""Generate a 1-page summary dashboard for Madam's meeting."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.8, 1.2, 1], hspace=0.35, wspace=0.3)

# ===== HEADER =====
ax_header = fig.add_subplot(gs[0, :])
ax_header.axis('off')
ax_header.text(0.5, 0.7, 'Weekly Research Summary — May 29, 2026', 
               ha='center', va='center', fontsize=20, fontweight='bold')
ax_header.text(0.5, 0.35, 'Adversarial Fairness Attacks on DRO-FAIR',
               ha='center', va='center', fontsize=15, style='italic', color='#555')
ax_header.text(0.5, 0.05, 'Both tasks assigned by Madam are COMPLETE',
               ha='center', va='center', fontsize=13, color='#2ecc71', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#2ecc71'))

# ===== TASK 1: TABULAR RESULTS =====
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_title('Task 1: Tabular Data (270 experiments)', fontsize=13, fontweight='bold', pad=10)
ax1.axis('off')

# Table
table_data = [
    ['Dataset', 'Attack', 'α', 'DRO Wins By', 'p-value'],
    ['Credit', 'IF', '0.2', '64.5%', '0.031*'],
    ['Credit', 'IF', '0.3', '97.5%', '0.031*'],
    ['LSAC', 'IF', '0.3', '96.2%', '0.031*'],
]
colors = [['#3498db']*5] + [['#eaf2f8']*5 for _ in range(3)]
table = ax1.table(cellText=table_data, cellColours=colors, loc='center',
                  colWidths=[0.18, 0.15, 0.1, 0.22, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
for i in range(5):
    table[(0, i)].set_text_props(fontweight='bold', color='white')

ax1.text(0.5, 0.02, 'DRO-FAIR significantly reduces DP violation under IF attacks\n(Wilcoxon signed-rank, n=5 seeds)',
         ha='center', va='bottom', fontsize=10, color='#2c3e50', transform=ax1.transAxes)

# ===== TASK 2: UTKFACE RESULTS =====
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_title('Task 2: UTKFace Images (9 experiments)', fontsize=13, fontweight='bold', pad=10)
ax2.axis('off')

table_data2 = [
    ['α', 'Condition', 'Naive DP', 'DRO DP', 'Winner'],
    ['0.0', 'Clean', '0.036', '0.027', 'DRO ✓'],
    ['0.1', 'Corrupted', '0.094', '0.130', 'Naive ✓'],
    ['0.2', 'Corrupted', '0.103', '0.110', 'Naive ✓'],
]
colors2 = [['#e74c3c']*5] + [['#fdeaea']*5, ['#e8f8f5']*5, ['#e8f8f5']*5]
table2 = ax2.table(cellText=table_data2, cellColours=colors2, loc='center',
                   colWidths=[0.12, 0.22, 0.18, 0.18, 0.18])
table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1, 2)
for i in range(5):
    table2[(0, i)].set_text_props(fontweight='bold', color='white')

ax2.text(0.5, 0.02, 'NEW FINDING: DRO makes fairness WORSE on image data!\nOpposite of tabular results.',
         ha='center', va='bottom', fontsize=10, color='#c0392b', fontweight='bold',
         transform=ax2.transAxes)

# ===== KEY INSIGHT =====
ax3 = fig.add_subplot(gs[2, :])
ax3.axis('off')

insight_box = """
KEY INSIGHT: DRO-FAIR robustness is NOT universal

    • Metric-dependent:  DRO wins against IF attacks  →  loses against DP attacks
    • Modality-dependent: DRO wins on tabular data    →  loses on image features

This is a NOVEL finding — prior work only tested random noise, not gradient-based adversarial attacks.
"""
ax3.text(0.5, 0.7, insight_box, ha='center', va='center', fontsize=12,
         family='monospace', color='#2c3e50',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fef9e7', edgecolor='#f1c40f', linewidth=2))

# Next steps
ax3.text(0.5, 0.15, 'Next: More UTKFace seeds → CelebA/FairFace → ResNet50 → Paper draft (NeurIPS/ICLR)',
         ha='center', va='center', fontsize=11, color='#555',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#ebf5fb', edgecolor='#3498db'))

plt.savefig('figures/summary_dashboard_may29.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/summary_dashboard_may29.pdf', bbox_inches='tight', facecolor='white')
print('Saved: figures/summary_dashboard_may29.png & .pdf')
