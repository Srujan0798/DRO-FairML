#!/usr/bin/env python3
"""
Publication-quality figure generation for DRO-FairML.
Produces 300 DPI figures with professional styling, significance markers,
and complete analysis for senior researcher / ICML-level presentation.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Professional style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'axes.titleweight':   'bold',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'legend.framealpha':  0.9,
    'legend.fontsize':    9,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.1,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
})

# Colorblind-friendly palette
NAIVE_COLOR  = '#E76F51'   # warm orange-red
DRO_COLOR    = '#2A9D8F'   # teal
CLEAN_COLOR  = '#264653'   # dark slate
CORRUPT_COLOR= '#F4A261'   # amber
ALPHAS       = [0.0, 0.1, 0.2, 0.3, 0.4]
DATASETS     = ['adult', 'credit', 'lsac']
DS_LABELS    = {'adult': 'Adult', 'credit': 'Credit', 'lsac': 'LSAC'}
METRIC_LABELS= {'accuracy': 'Accuracy ↑', 'dp_violation': 'DP Violation ↓', 'if_violation': 'IF Violation ↓'}


def load_results():
    with open('results/all_results.json') as f:
        return json.load(f)


def wilcoxon_p(naive_vals, dro_vals):
    try:
        if len(naive_vals) < 3:
            return 1.0
        _, p = wilcoxon(naive_vals, dro_vals)
        return p
    except Exception:
        return 1.0


def sig_marker(p, dro_mean, naive_mean):
    """Return significance star if DRO wins and p<0.05."""
    if p < 0.001 and dro_mean < naive_mean:
        return '***'
    if p < 0.01 and dro_mean < naive_mean:
        return '**'
    if p < 0.05 and dro_mean < naive_mean:
        return '*'
    return ''


def get_stats(results, dataset, alpha, method, metric, eval_type='clean'):
    subset = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
    if not subset:
        return np.nan, np.nan
    vals = [r[method][eval_type][metric] for r in subset]
    return np.mean(vals), np.std(vals) / np.sqrt(len(vals))


# ── Figure 1: Main Results (3×3, line plots with error bands) ───────────────
def fig1_main_results(results, out='figures'):
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle('DRO-FAIR vs Naive-FAIR: Main Results (Clean Test, mean ± SE, 10 seeds)',
                 fontsize=13, fontweight='bold', y=1.01)

    for row, ds in enumerate(DATASETS):
        for col, metric in enumerate(['accuracy', 'dp_violation', 'if_violation']):
            ax = axes[row, col]
            naive_m, naive_se, dro_m, dro_se = [], [], [], []
            sig_marks = []

            for alpha in ALPHAS:
                nm, nse = get_stats(results, ds, alpha, 'naive', metric)
                dm, dse = get_stats(results, ds, alpha, 'dro',   metric)
                naive_m.append(nm); naive_se.append(nse)
                dro_m.append(dm);   dro_se.append(dse)

                subset = [r for r in results if r['dataset']==ds and r['alpha']==alpha]
                nv = [r['naive']['clean'][metric] for r in subset]
                dv = [r['dro']['clean'][metric]   for r in subset]
                p  = wilcoxon_p(nv, dv)
                sig_marks.append(sig_marker(p, dm, nm))

            x = np.array(ALPHAS)
            ax.plot(x, naive_m, 'o-', color=NAIVE_COLOR, lw=2, ms=5, label='Naive-FAIR', zorder=3)
            ax.fill_between(x,
                            np.array(naive_m)-np.array(naive_se),
                            np.array(naive_m)+np.array(naive_se),
                            color=NAIVE_COLOR, alpha=0.15)
            ax.plot(x, dro_m, 's-', color=DRO_COLOR, lw=2, ms=5, label='DRO-FAIR', zorder=3)
            ax.fill_between(x,
                            np.array(dro_m)-np.array(dro_se),
                            np.array(dro_m)+np.array(dro_se),
                            color=DRO_COLOR, alpha=0.15)

            # Significance markers above DRO line
            ymax = ax.get_ylim()[1]
            for i, (xi, mark) in enumerate(zip(x, sig_marks)):
                if mark:
                    ypos = max(naive_m[i], dro_m[i]) + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    ax.text(xi, ypos, mark, ha='center', va='bottom', fontsize=8,
                            color=DRO_COLOR, fontweight='bold')

            ax.set_xlabel('Corruption Level α', fontsize=9)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_xticks(ALPHAS)
            ax.set_title(f'{DS_LABELS[ds]}', fontsize=11, fontweight='bold')
            if row == 0 and col == 2:
                ax.legend(loc='upper right')

            # Shade α=0 lightly (no corruption baseline)
            ax.axvspan(-0.02, 0.02, alpha=0.08, color='gray')

    # Add metric column labels
    for col, metric in enumerate(['accuracy', 'dp_violation', 'if_violation']):
        axes[0, col].set_title(
            f'{DS_LABELS[DATASETS[0]]} — {METRIC_LABELS[metric]}',
            fontsize=11, fontweight='bold')
    for row in range(1, 3):
        for col in range(3):
            axes[row, col].set_title(
                DS_LABELS[DATASETS[row]], fontsize=11, fontweight='bold')

    plt.tight_layout()
    os.makedirs(out, exist_ok=True)
    fig.savefig(f'{out}/fig1_main_results.png')
    fig.savefig(f'{out}/fig1_main_results.pdf')
    plt.close()
    print(f'  ✓ fig1_main_results.png')


# ── Figure 2: DP Reduction Heatmap ──────────────────────────────────────────
def fig2_dp_reduction_heatmap(results, out='figures'):
    dp_red  = np.zeros((3, 5))
    p_matrix= np.ones((3, 5))

    for i, ds in enumerate(DATASETS):
        for j, alpha in enumerate(ALPHAS):
            subset = [r for r in results if r['dataset']==ds and r['alpha']==alpha]
            if not subset:
                continue
            nv = [r['naive']['clean']['dp_violation'] for r in subset]
            dv = [r['dro']['clean']['dp_violation']   for r in subset]
            nm, dm = np.mean(nv), np.mean(dv)
            dp_red[i, j] = (nm - dm) / nm * 100 if nm > 1e-6 else 0.0
            p_matrix[i, j] = wilcoxon_p(nv, dv)

    fig, ax = plt.subplots(figsize=(9, 4))
    vmax = max(abs(dp_red).max(), 1)
    im = ax.imshow(dp_red, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('DP Reduction (%) — positive = DRO wins', fontsize=10)

    ax.set_xticks(range(5))
    ax.set_xticklabels([f'α={a}' for a in ALPHAS], fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels([DS_LABELS[d] for d in DATASETS], fontsize=11, fontweight='bold')
    ax.set_title('DRO-FAIR DP Reduction over Naive-FAIR (%)\n* p<0.05  ** p<0.01  *** p<0.001 (Wilcoxon signed-rank)',
                 fontsize=11, fontweight='bold')

    for i in range(3):
        for j in range(5):
            p = p_matrix[i, j]
            val = dp_red[i, j]
            stars = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
            color = 'white' if abs(val) > vmax*0.6 else 'black'
            ax.text(j, i, f'{val:+.0f}%\n{stars}',
                    ha='center', va='center', fontsize=10,
                    color=color, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{out}/fig2_dp_reduction_heatmap.png')
    fig.savefig(f'{out}/fig2_dp_reduction_heatmap.pdf')
    plt.close()
    print(f'  ✓ fig2_dp_reduction_heatmap.png')


# ── Figure 3: Clean vs Corrupted robustness ──────────────────────────────────
def fig3_robustness(results, out='figures'):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Robustness: Clean vs. Adversarially Corrupted Test Evaluation',
                 fontsize=13, fontweight='bold')

    for col, ds in enumerate(DATASETS):
        for row, method in enumerate(['naive', 'dro']):
            ax = axes[row, col]
            clean_m, clean_se, corr_m, corr_se = [], [], [], []

            for alpha in ALPHAS:
                cm, cse = get_stats(results, ds, alpha, method, 'dp_violation', 'clean')
                km, kse = get_stats(results, ds, alpha, method, 'dp_violation', 'corrupted')
                clean_m.append(cm); clean_se.append(cse)
                corr_m.append(km);  corr_se.append(kse)

            x = np.array(ALPHAS)
            ax.plot(x, clean_m, 'o-', color=CLEAN_COLOR, lw=2, ms=5, label='Clean test')
            ax.fill_between(x,
                            np.array(clean_m)-np.array(clean_se),
                            np.array(clean_m)+np.array(clean_se),
                            color=CLEAN_COLOR, alpha=0.15)
            ax.plot(x, corr_m, 's--', color=CORRUPT_COLOR, lw=2, ms=5, label='Corrupted test')
            ax.fill_between(x,
                            np.array(corr_m)-np.array(corr_se),
                            np.array(corr_m)+np.array(corr_se),
                            color=CORRUPT_COLOR, alpha=0.15)

            ax.set_xlabel('Corruption Level α', fontsize=9)
            ax.set_ylabel('DP Violation', fontsize=9)
            ax.set_xticks(ALPHAS)
            label = 'Naive-FAIR' if method == 'naive' else 'DRO-FAIR'
            ax.set_title(f'{DS_LABELS[ds]} — {label}', fontsize=11, fontweight='bold')
            if row == 0 and col == 0:
                ax.legend()

    plt.tight_layout()
    fig.savefig(f'{out}/fig3_robustness_clean_vs_corrupted.png')
    fig.savefig(f'{out}/fig3_robustness_clean_vs_corrupted.pdf')
    plt.close()
    print(f'  ✓ fig3_robustness_clean_vs_corrupted.png')


# ── Figure 4: Statistical significance matrix ────────────────────────────────
def fig4_significance_matrix(results, out='figures'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Wilcoxon Signed-Rank Test: DRO-FAIR vs Naive-FAIR (10 seeds)\np-values — green = DRO significantly better (p<0.05)',
                 fontsize=11, fontweight='bold')

    for ax_idx, metric in enumerate(['dp_violation', 'if_violation']):
        ax = axes[ax_idx]
        p_grid  = np.ones((3, 5))
        win_grid= np.zeros((3, 5))

        for i, ds in enumerate(DATASETS):
            for j, alpha in enumerate(ALPHAS):
                subset = [r for r in results if r['dataset']==ds and r['alpha']==alpha]
                nv = [r['naive']['clean'][metric] for r in subset]
                dv = [r['dro']['clean'][metric]   for r in subset]
                p  = wilcoxon_p(nv, dv)
                p_grid[i, j] = p
                win_grid[i, j] = 1 if (p < 0.05 and np.mean(dv) < np.mean(nv)) else (
                                 -1 if (p < 0.05 and np.mean(dv) > np.mean(nv)) else 0)

        color_mat = np.zeros((3, 5, 4))
        for i in range(3):
            for j in range(5):
                if win_grid[i, j] == 1:
                    color_mat[i, j] = [0.165, 0.612, 0.561, 0.85]   # teal = DRO wins
                elif win_grid[i, j] == -1:
                    color_mat[i, j] = [0.906, 0.435, 0.318, 0.85]   # red = DRO loses
                else:
                    color_mat[i, j] = [0.9, 0.9, 0.9, 0.7]           # grey = tie

        ax.imshow(color_mat, aspect='auto')
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'α={a}' for a in ALPHAS], fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels([DS_LABELS[d] for d in DATASETS], fontsize=11, fontweight='bold')

        metric_name = 'DP Violation' if metric == 'dp_violation' else 'IF Violation'
        ax.set_title(f'{metric_name}', fontsize=11, fontweight='bold')

        for i in range(3):
            for j in range(5):
                p = p_grid[i, j]
                stars = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
                result = 'DRO ↓' if win_grid[i,j]==1 else ('Naive ↓' if win_grid[i,j]==-1 else 'tie')
                ax.text(j, i, f'p={p:.3f}\n{stars}\n{result}',
                        ha='center', va='center', fontsize=8, fontweight='bold')

        teal_patch = mpatches.Patch(color=[0.165,0.612,0.561,0.85], label='DRO significantly better')
        red_patch  = mpatches.Patch(color=[0.906,0.435,0.318,0.85], label='Naive significantly better')
        grey_patch = mpatches.Patch(color=[0.9,0.9,0.9,0.7],        label='No significant difference')
        ax.legend(handles=[teal_patch, red_patch, grey_patch],
                  loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=3, fontsize=8)

    plt.tight_layout()
    fig.savefig(f'{out}/fig4_significance_matrix.png')
    fig.savefig(f'{out}/fig4_significance_matrix.pdf')
    plt.close()
    print(f'  ✓ fig4_significance_matrix.png')


# ── Figure 5: Accuracy-Fairness Tradeoff scatter ─────────────────────────────
def fig5_accuracy_fairness_tradeoff(results, out='figures'):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Accuracy–Fairness Tradeoff (Clean Test, each point = one α level, mean over 10 seeds)',
                 fontsize=12, fontweight='bold')

    markers_alpha = {0.0: 'o', 0.1: 's', 0.2: '^', 0.3: 'D', 0.4: 'X'}
    sizes_alpha   = {0.0: 60, 0.1: 70, 0.2: 80, 0.3: 90, 0.4: 100}

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        for alpha in ALPHAS:
            subset = [r for r in results if r['dataset']==ds and r['alpha']==alpha]
            if not subset:
                continue
            naive_acc = np.mean([r['naive']['clean']['accuracy']     for r in subset])
            naive_dp  = np.mean([r['naive']['clean']['dp_violation'] for r in subset])
            dro_acc   = np.mean([r['dro']['clean']['accuracy']       for r in subset])
            dro_dp    = np.mean([r['dro']['clean']['dp_violation']   for r in subset])

            ax.scatter(naive_acc, naive_dp, color=NAIVE_COLOR, s=sizes_alpha[alpha],
                       marker=markers_alpha[alpha], zorder=3, edgecolors='white', lw=0.5)
            ax.scatter(dro_acc,   dro_dp,   color=DRO_COLOR,   s=sizes_alpha[alpha],
                       marker=markers_alpha[alpha], zorder=3, edgecolors='white', lw=0.5)
            ax.annotate(f'α={alpha}', (dro_acc, dro_dp),
                        textcoords='offset points', xytext=(4, 3), fontsize=7, color=DRO_COLOR)

        ax.set_xlabel('Accuracy ↑', fontsize=10)
        ax.set_ylabel('DP Violation ↓', fontsize=10)
        ax.set_title(DS_LABELS[ds], fontsize=11, fontweight='bold')

    naive_patch = mpatches.Patch(color=NAIVE_COLOR, label='Naive-FAIR')
    dro_patch   = mpatches.Patch(color=DRO_COLOR,   label='DRO-FAIR')
    axes[2].legend(handles=[naive_patch, dro_patch], loc='upper right')
    plt.tight_layout()
    fig.savefig(f'{out}/fig5_accuracy_fairness_tradeoff.png')
    fig.savefig(f'{out}/fig5_accuracy_fairness_tradeoff.pdf')
    plt.close()
    print(f'  ✓ fig5_accuracy_fairness_tradeoff.png')


# ── Figure 6: Per-seed variance — Adult instability ──────────────────────────
def fig6_seed_stability(results, out='figures'):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Per-Seed DRO-FAIR Accuracy Distribution (10 seeds) — Stability Analysis',
                 fontsize=12, fontweight='bold')

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        data, labels, colors = [], [], []
        for alpha in [0.1, 0.2, 0.3, 0.4]:
            subset = [r for r in results if r['dataset']==ds and r['alpha']==alpha]
            if not subset:
                continue
            dro_accs = sorted([r['dro']['clean']['accuracy'] for r in subset])
            data.append(dro_accs)
            labels.append(f'α={alpha}')
            colors.append(DRO_COLOR)

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color='white', lw=2),
                        whiskerprops=dict(color=DRO_COLOR),
                        capprops=dict(color=DRO_COLOR),
                        flierprops=dict(marker='o', markerfacecolor=NAIVE_COLOR, markersize=4))
        for patch in bp['boxes']:
            patch.set_facecolor(DRO_COLOR)
            patch.set_alpha(0.7)

        ax.axhline(0.75, color='red', lw=1, ls='--', alpha=0.6, label='0.75 threshold')
        ax.set_ylabel('DRO-FAIR Accuracy', fontsize=10)
        ax.set_xlabel('Corruption Level', fontsize=10)
        ax.set_title(DS_LABELS[ds], fontsize=11, fontweight='bold')
        ax.set_ylim(0.1, 1.0)
        if col == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(f'{out}/fig6_seed_stability.png')
    fig.savefig(f'{out}/fig6_seed_stability.pdf')
    plt.close()
    print(f'  ✓ fig6_seed_stability.png')


# ── Figure 7: Summary win-rate bar chart ─────────────────────────────────────
def fig7_summary(results, out='figures'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('DRO-FAIR vs Naive-FAIR: Summary Win Rates (α=0.1,0.2,0.3)',
                 fontsize=12, fontweight='bold')

    for ax_idx, metric in enumerate(['dp_violation', 'if_violation']):
        ax = axes[ax_idx]
        wins = {'Adult': 0, 'Credit': 0, 'LSAC': 0}
        losses = {'Adult': 0, 'Credit': 0, 'LSAC': 0}
        ns = {'Adult': 0, 'Credit': 0, 'LSAC': 0}

        for ds in DATASETS:
            label = DS_LABELS[ds]
            for alpha in [0.1, 0.2, 0.3]:
                subset = [r for r in results if r['dataset']==ds and r['alpha']==alpha]
                nv = [r['naive']['clean'][metric] for r in subset]
                dv = [r['dro']['clean'][metric]   for r in subset]
                p  = wilcoxon_p(nv, dv)
                if p < 0.05 and np.mean(dv) < np.mean(nv):
                    wins[label] += 1
                elif p < 0.05 and np.mean(dv) > np.mean(nv):
                    losses[label] += 1
                else:
                    ns[label] += 1

        ds_labels_list = list(wins.keys())
        win_vals  = [wins[d] for d in ds_labels_list]
        loss_vals = [losses[d] for d in ds_labels_list]
        ns_vals   = [ns[d] for d in ds_labels_list]
        x = np.arange(len(ds_labels_list))

        ax.bar(x, win_vals,  color=DRO_COLOR,   label='DRO sig. better', alpha=0.85)
        ax.bar(x, ns_vals,   bottom=win_vals,    color='#ADB5BD', label='No sig. diff.', alpha=0.85)
        ax.bar(x, loss_vals, bottom=[w+n for w,n in zip(win_vals, ns_vals)],
               color=NAIVE_COLOR, label='Naive sig. better', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels_list, fontsize=11, fontweight='bold')
        ax.set_ylabel('# Comparisons (out of 3)', fontsize=10)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(0, 3.5)
        metric_name = 'DP Violation' if metric == 'dp_violation' else 'IF Violation'
        ax.set_title(f'{metric_name}', fontsize=11, fontweight='bold')
        if ax_idx == 1:
            ax.legend(fontsize=9)

        for i, (w, l) in enumerate(zip(win_vals, loss_vals)):
            ax.text(i, w + 0.05, f'{w}/3', ha='center', va='bottom',
                    fontsize=11, color=DRO_COLOR, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{out}/fig7_summary_win_rates.png')
    fig.savefig(f'{out}/fig7_summary_win_rates.pdf')
    plt.close()
    print(f'  ✓ fig7_summary_win_rates.png')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs('figures', exist_ok=True)
    results = load_results()
    print(f'Loaded {len(results)} results. Generating publication-quality figures...')
    fig1_main_results(results)
    fig2_dp_reduction_heatmap(results)
    fig3_robustness(results)
    fig4_significance_matrix(results)
    fig5_accuracy_fairness_tradeoff(results)
    fig6_seed_stability(results)
    fig7_summary(results)
    print(f'\n✅ All 7 figures saved to figures/ at 300 DPI.')


if __name__ == '__main__':
    main()
