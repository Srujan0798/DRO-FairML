#!/usr/bin/env python3
"""
Generate Figure 8: Attack-Defense Matrix heatmap from fairness_pgd_wilcoxon.csv.

Each cell corresponds to (attack type × dataset).
- Color: best significant positive DP reduction %, or average if none.
- Text: best significant result with α and p-value, or average + "n.s."
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Style ────────────────────────────────────────────────────────────────────
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
    'axes.linewidth':     0.8,
    'axes.labelpad':      4,
    'grid.alpha':         0.3,
    'grid.linewidth':     0.4,
    'grid.linestyle':     '--',
    'legend.frameon':     True,
    'legend.framealpha':  0.9,
    'legend.fontsize':    10,
    'legend.edgecolor':   '0.8',
    'savefig.dpi':        300,
    'figure.dpi':         150,
})

CSV_PATH = 'results/fairness_pgd_wilcoxon.csv'
OUT_DIR = 'figures'
OUT_NAME = 'fig8_attack_defense_matrix'

DATASETS = ['adult', 'credit', 'lsac']
DS_LABELS = {'adult': 'Adult', 'credit': 'Credit', 'lsac': 'LSAC'}
ATTACKS = ['dp', 'if', 'combined']
ATTACK_LABELS = {'dp': 'DP', 'if': 'IF', 'combined': 'Combined'}

SIG_THRESHOLD = 0.05
CLIP_MIN = -100
CLIP_MAX = 100


def main():
    df = pd.read_csv(CSV_PATH)

    n_rows = len(ATTACKS)
    n_cols = len(DATASETS)
    color_mat = np.zeros((n_rows, n_cols))
    text_mat = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for i, attack in enumerate(ATTACKS):
        for j, dataset in enumerate(DATASETS):
            sub = df[(df['attack'] == attack) & (df['dataset'] == dataset)]
            if sub.empty:
                color_mat[i, j] = 0.0
                text_mat[i][j] = "N/A"
                continue

            # Best significant positive result
            sig = sub[sub['dp_pvalue'] < SIG_THRESHOLD]
            if not sig.empty:
                best_idx = sig['dp_reduction_pct'].idxmax()
                best = sig.loc[best_idx]
                val = float(best['dp_reduction_pct'])
                alpha = float(best['alpha'])
                p = float(best['dp_pvalue'])
                text = f"{val:+.1f}%\nα={alpha:.1f}, p={p:.3f}"
                color_val = val
            else:
                avg = float(sub['dp_reduction_pct'].mean())
                text = f"{avg:+.1f}%\nn.s."
                color_val = avg

            color_mat[i, j] = color_val
            text_mat[i][j] = text

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = LinearSegmentedColormap.from_list(
        'rdwhgn', ['#e74c3c', '#ffffff', '#2ecc71'], N=256)

    im = ax.imshow(color_mat, cmap=cmap, vmin=CLIP_MIN, vmax=CLIP_MAX, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046, shrink=0.8)
    cbar.set_label('DP Reduction %', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Ticks & labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=12, fontweight='bold')
    ax.set_yticklabels([ATTACK_LABELS[a] for a in ATTACKS], fontsize=12, fontweight='bold')
    ax.tick_params(length=0)

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Attack Type', fontsize=11)
    ax.set_title('DRO vs Naive: DP Reduction % under Adversarial Attacks',
                 fontsize=13, fontweight='bold', pad=12)

    # Annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = color_mat[i, j]
            txt = text_mat[i][j]
            clipped = np.clip(val, CLIP_MIN, CLIP_MAX)
            tc = 'white' if abs(clipped) > 55 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=11, color=tc, fontweight='bold')

    ax.set_frame_on(False)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f'{OUT_NAME}.pdf')
    png_path = os.path.join(OUT_DIR, f'{OUT_NAME}.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    # Verify
    for p in (pdf_path, png_path):
        size = os.path.getsize(p)
        print(f"Saved {p} ({size:,} bytes)")
        if size < 10 * 1024:
            raise RuntimeError(f"{p} is too small ({size} bytes)")


if __name__ == '__main__':
    main()
