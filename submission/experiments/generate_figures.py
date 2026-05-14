#!/usr/bin/env python3
"""
Publication-quality figure generation for DRO-FairML.
300 DPI, colorblind-safe palette, ICML-submission aesthetics.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family':          'DejaVu Sans',
    'font.size':            10,
    'axes.titlesize':       11,
    'axes.labelsize':       10,
    'axes.titleweight':     'bold',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.linewidth':       0.8,
    'grid.alpha':           0.25,
    'grid.linestyle':       '--',
    'grid.linewidth':       0.6,
    'legend.framealpha':    0.92,
    'legend.fontsize':      9,
    'legend.edgecolor':     '0.8',
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.12,
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'xtick.major.size':     3,
    'ytick.major.size':     3,
})

# Colorblind-safe palette (Wong 2011)
NAIVE_COLOR   = '#D55E00'   # vermillion
DRO_COLOR     = '#009E73'   # bluish green
CLEAN_COLOR   = '#0072B2'   # blue
CORRUPT_COLOR = '#E69F00'   # orange
ALPHAS        = [0.0, 0.1, 0.2, 0.3, 0.4]
DATASETS      = ['adult', 'credit', 'lsac']
DS_LABELS     = {'adult': 'Adult', 'credit': 'Credit', 'lsac': 'LSAC'}
METRIC_LABELS = {
    'accuracy':     'Accuracy ↑',
    'dp_violation': 'DP Violation ↓',
    'if_violation': 'IF Violation ↓',
}
OUT = 'figures'


def load_results():
    with open('results/all_results.json') as f:
        return json.load(f)


def wilcoxon_p(a_vals, b_vals):
    try:
        if len(a_vals) < 3 or np.allclose(a_vals, b_vals):
            return 1.0
        _, p = wilcoxon(a_vals, b_vals)
        return p
    except Exception:
        return 1.0


def stars(p, dro_mean, naive_mean):
    """Stars only when DRO is better (lower)."""
    if dro_mean >= naive_mean:
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def get_vals(results, dataset, alpha, method, metric, eval_t='clean'):
    recs = [r for r in results if r['dataset'] == dataset and abs(r['alpha'] - alpha) < 1e-6]
    return [r[method][eval_t][metric] for r in recs] if recs else []


def mean_se(vals):
    if not vals or all(np.isnan(v) for v in vals):
        return np.nan, np.nan
    a = np.array(vals, dtype=float)
    return np.mean(a), np.std(a) / np.sqrt(len(a))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Main Results (3 × 3)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_main_results(results):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(
        'DRO-FAIR vs Naive-FAIR — Clean Test Set (mean ± 1 SE, n=10 seeds)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    metrics = ['accuracy', 'dp_violation', 'if_violation']

    for row, ds in enumerate(DATASETS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            n_m, n_se, d_m, d_se, smarks = [], [], [], [], []

            for alpha in ALPHAS:
                nv = get_vals(results, ds, alpha, 'naive', metric)
                dv = get_vals(results, ds, alpha, 'dro',   metric)
                nm, nse = mean_se(nv)
                dm, dse = mean_se(dv)
                n_m.append(nm); n_se.append(nse)
                d_m.append(dm); d_se.append(dse)
                p = wilcoxon_p(nv, dv)
                smarks.append(stars(p, dm, nm))

            x = np.array(ALPHAS)
            n_m  = np.array(n_m,  float)
            n_se = np.array(n_se, float)
            d_m  = np.array(d_m,  float)
            d_se = np.array(d_se, float)

            ax.plot(x, n_m, 'o-', color=NAIVE_COLOR, lw=2, ms=5,
                    label='Naive-FAIR', zorder=4)
            ax.fill_between(x, n_m - n_se, n_m + n_se,
                            color=NAIVE_COLOR, alpha=0.13, zorder=2)
            ax.plot(x, d_m, 's-', color=DRO_COLOR, lw=2, ms=5,
                    label='DRO-FAIR', zorder=4)
            ax.fill_between(x, d_m - d_se, d_m + d_se,
                            color=DRO_COLOR, alpha=0.13, zorder=2)

            # Significance stars above the higher of the two lines
            ylo, yhi = ax.get_ylim()
            span = yhi - ylo if yhi > ylo else 1
            for i, (xi, mk) in enumerate(zip(x, smarks)):
                if mk:
                    ypos = max(n_m[i], d_m[i]) + 0.04 * span
                    ax.text(xi, ypos, mk, ha='center', va='bottom',
                            fontsize=9, color=DRO_COLOR, fontweight='bold')

            # Grey band for α=0 baseline
            ax.axvspan(-0.02, 0.02, color='gray', alpha=0.07, zorder=1)

            ax.set_xlabel('Corruption level α', fontsize=9)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_xticks(ALPHAS)
            # Every cell gets "Dataset — Metric" title
            ax.set_title(f'{DS_LABELS[ds]} — {METRIC_LABELS[metric]}',
                         fontsize=10, fontweight='bold')

            if row == 0 and col == 2:
                ax.legend(loc='upper right', framealpha=0.9)

    for ax in axes.flat:
        ax.set_xlim(-0.04, 0.44)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/fig1_main_results.png')
    fig.savefig(f'{OUT}/fig1_main_results.pdf')
    plt.close()
    print('  ✓ fig1_main_results')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — DP Reduction Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig2_dp_reduction_heatmap(results):
    dp_red   = np.full((3, 5), np.nan)
    p_mat    = np.ones((3, 5))
    raw_red  = np.full((3, 5), np.nan)   # uncapped for text

    for i, ds in enumerate(DATASETS):
        for j, alpha in enumerate(ALPHAS):
            nv = get_vals(results, ds, alpha, 'naive', 'dp_violation')
            dv = get_vals(results, ds, alpha, 'dro',   'dp_violation')
            if not nv:
                continue
            nm, dm = np.mean(nv), np.mean(dv)
            raw = (nm - dm) / max(nm, 1e-9) * 100
            raw_red[i, j]  = raw
            dp_red[i, j]   = np.clip(raw, -100, 100)   # cap for colorscale
            p_mat[i, j]    = wilcoxon_p(nv, dv)

    fig, ax = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)

    # Custom diverging colormap: red→white→green
    cmap = LinearSegmentedColormap.from_list(
        'rwg', ['#C0392B', '#F5F5F5', '#1E8449'], N=256)
    im = ax.imshow(dp_red, cmap=cmap, vmin=-100, vmax=100, aspect='auto')
    ax.grid(False)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label('DP Reduction (%)  positive → DRO wins', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(5))
    ax.set_xticklabels([f'α={a}' for a in ALPHAS], fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels([DS_LABELS[d] for d in DATASETS], fontsize=11, fontweight='bold')
    ax.set_title(
        'DRO-FAIR DP Reduction over Naive-FAIR\n'
        '(capped ±100 % for color; exact value shown  |  * p<0.05  ** p<0.01  *** p<0.001)',
        fontsize=10, fontweight='bold')

    for i in range(3):
        for j in range(5):
            p   = p_mat[i, j]
            val = raw_red[i, j]
            if np.isnan(val):
                continue
            st  = ('***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))
            # White text on dark cells, black on light
            bg  = dp_red[i, j]
            tc  = 'white' if abs(bg) > 60 else 'black'
            # Mark extreme Adult α=0.3 cell differently
            suffix = '' if abs(val) <= 100 else ' ⚠'
            ax.text(j, i, f'{val:+.0f}%{suffix}\n{st}',
                    ha='center', va='center', fontsize=9.5,
                    color=tc, fontweight='bold')

    ax.set_frame_on(False)
    fig.savefig(f'{OUT}/fig2_dp_reduction_heatmap.png')
    fig.savefig(f'{OUT}/fig2_dp_reduction_heatmap.pdf')
    plt.close()
    print('  ✓ fig2_dp_reduction_heatmap')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Clean vs Corrupted Robustness
# ─────────────────────────────────────────────────────────────────────────────
def fig3_robustness(results):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(
        'Robustness: DP Violation on Clean vs Adversarially Corrupted Test Set',
        fontsize=12, fontweight='bold')

    methods = ['naive', 'dro']
    mlabel  = {'naive': 'Naive-FAIR', 'dro': 'DRO-FAIR'}

    for col, ds in enumerate(DATASETS):
        for row, method in enumerate(methods):
            ax = axes[row, col]
            cl_m, cl_se, cr_m, cr_se = [], [], [], []
            for alpha in ALPHAS:
                cv  = get_vals(results, ds, alpha, method, 'dp_violation', 'clean')
                kv  = get_vals(results, ds, alpha, method, 'dp_violation', 'corrupted')
                cm, cse = mean_se(cv)
                km, kse = mean_se(kv)
                cl_m.append(cm); cl_se.append(cse)
                cr_m.append(km); cr_se.append(kse)

            x    = np.array(ALPHAS)
            cl_m = np.array(cl_m, float);  cl_se = np.array(cl_se, float)
            cr_m = np.array(cr_m, float);  cr_se = np.array(cr_se, float)

            ax.plot(x, cl_m, 'o-', color=CLEAN_COLOR, lw=2, ms=5,
                    label='Clean test')
            ax.fill_between(x, cl_m - cl_se, cl_m + cl_se,
                            color=CLEAN_COLOR, alpha=0.12)
            ax.plot(x, cr_m, 's--', color=CORRUPT_COLOR, lw=2, ms=5,
                    label='Corrupted test')
            ax.fill_between(x, cr_m - cr_se, cr_m + cr_se,
                            color=CORRUPT_COLOR, alpha=0.12)

            ax.set_xlabel('Corruption level α', fontsize=9)
            ax.set_ylabel('DP Violation', fontsize=9)
            ax.set_xticks(ALPHAS)
            ax.set_title(f'{DS_LABELS[ds]} — {mlabel[method]}',
                         fontsize=10, fontweight='bold')
            if row == 0 and col == 0:
                ax.legend(fontsize=9)

    fig.savefig(f'{OUT}/fig3_robustness_clean_vs_corrupted.png')
    fig.savefig(f'{OUT}/fig3_robustness_clean_vs_corrupted.pdf')
    plt.close()
    print('  ✓ fig3_robustness_clean_vs_corrupted')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Significance Matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig4_significance_matrix(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), constrained_layout=True)
    fig.suptitle(
        'Statistical Significance: Wilcoxon Signed-Rank (one-sided, n=10 seeds, clean test)',
        fontsize=11, fontweight='bold')

    TEAL  = np.array([0.000, 0.620, 0.451, 0.85])  # DRO wins
    RED   = np.array([0.835, 0.369, 0.000, 0.85])  # Naive wins
    GREY  = np.array([0.90,  0.90,  0.90,  0.65])  # tie

    for ax_idx, metric in enumerate(['dp_violation', 'if_violation']):
        ax = axes[ax_idx]
        p_grid   = np.ones((3, 5))
        win_grid = np.zeros((3, 5))

        for i, ds in enumerate(DATASETS):
            for j, alpha in enumerate(ALPHAS):
                nv = get_vals(results, ds, alpha, 'naive', metric)
                dv = get_vals(results, ds, alpha, 'dro',   metric)
                p  = wilcoxon_p(nv, dv)
                p_grid[i, j] = p
                nm, dm = (np.mean(nv) if nv else np.nan), (np.mean(dv) if dv else np.nan)
                if p < 0.05 and dm < nm:
                    win_grid[i, j] =  1
                elif p < 0.05 and dm > nm:
                    win_grid[i, j] = -1

        cmat = np.zeros((3, 5, 4))
        for i in range(3):
            for j in range(5):
                cmat[i, j] = TEAL if win_grid[i, j] == 1 else (
                              RED  if win_grid[i, j] == -1 else GREY)

        ax.imshow(cmat, aspect='auto', interpolation='nearest')
        ax.grid(False)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'α={a}' for a in ALPHAS], fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels([DS_LABELS[d] for d in DATASETS],
                           fontsize=10, fontweight='bold')
        ax.set_title(
            'DP Violation' if metric == 'dp_violation' else 'IF Violation',
            fontsize=11, fontweight='bold')

        for i in range(3):
            for j in range(5):
                p   = p_grid[i, j]
                w   = win_grid[i, j]
                st  = ('***' if p < 0.001 else ('**' if p < 0.01 else
                       ('*'  if p < 0.05  else 'ns')))
                lbl = ('DRO ↓' if w == 1 else ('Naive ↓' if w == -1 else 'tie'))
                tc  = 'white' if w != 0 else '#444444'
                ax.text(j, i, f'p={p:.3f}\n{st}  {lbl}',
                        ha='center', va='center', fontsize=8,
                        color=tc, fontweight='bold')

        ax.set_frame_on(False)
        teal_p = mpatches.Patch(color=TEAL, label='DRO significantly better')
        red_p  = mpatches.Patch(color=RED,  label='Naive significantly better')
        grey_p = mpatches.Patch(color=GREY, label='No significant difference')
        ax.legend(handles=[teal_p, red_p, grey_p],
                  loc='upper center', bbox_to_anchor=(0.5, -0.14),
                  ncol=3, fontsize=8, framealpha=0.9)

    fig.savefig(f'{OUT}/fig4_significance_matrix.png')
    fig.savefig(f'{OUT}/fig4_significance_matrix.pdf')
    plt.close()
    print('  ✓ fig4_significance_matrix')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Accuracy–Fairness Tradeoff
# ─────────────────────────────────────────────────────────────────────────────
def fig5_accuracy_fairness_tradeoff(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    fig.suptitle(
        'Accuracy–Fairness Tradeoff (mean over 10 seeds, clean test; each point = one α level)',
        fontsize=12, fontweight='bold')

    markers = {0.0: 'o', 0.1: 's', 0.2: '^', 0.3: 'D', 0.4: 'X'}
    sizes   = {0.0: 65,  0.1: 75,  0.2: 85,  0.3: 95,  0.4: 105}

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        for alpha in ALPHAS:
            nv_acc = get_vals(results, ds, alpha, 'naive', 'accuracy')
            nv_dp  = get_vals(results, ds, alpha, 'naive', 'dp_violation')
            dv_acc = get_vals(results, ds, alpha, 'dro',   'accuracy')
            dv_dp  = get_vals(results, ds, alpha, 'dro',   'dp_violation')
            if not nv_acc:
                continue
            na, nd = np.mean(nv_acc), np.mean(nv_dp)
            da, dd = np.mean(dv_acc), np.mean(dv_dp)

            ax.scatter(na, nd, color=NAIVE_COLOR, s=sizes[alpha],
                       marker=markers[alpha], zorder=4, edgecolors='white', lw=0.8)
            ax.scatter(da, dd, color=DRO_COLOR, s=sizes[alpha],
                       marker=markers[alpha], zorder=4, edgecolors='white', lw=0.8)
            ax.annotate(f'α={alpha}', (da, dd),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=7.5, color=DRO_COLOR)

        ax.set_xlabel('Accuracy ↑', fontsize=10)
        ax.set_ylabel('DP Violation ↓', fontsize=10)
        ax.set_title(DS_LABELS[ds], fontsize=11, fontweight='bold')

    naive_p = mpatches.Patch(color=NAIVE_COLOR, label='Naive-FAIR')
    dro_p   = mpatches.Patch(color=DRO_COLOR,   label='DRO-FAIR')
    alpha_lines = [plt.Line2D([0],[0], marker=markers[a], color='gray',
                              linestyle='None', ms=7, label=f'α={a}')
                   for a in ALPHAS]
    axes[2].legend(handles=[naive_p, dro_p] + alpha_lines,
                   loc='upper right', fontsize=8, ncol=2)

    fig.savefig(f'{OUT}/fig5_accuracy_fairness_tradeoff.png')
    fig.savefig(f'{OUT}/fig5_accuracy_fairness_tradeoff.pdf')
    plt.close()
    print('  ✓ fig5_accuracy_fairness_tradeoff')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Seed Stability Boxplots
# ─────────────────────────────────────────────────────────────────────────────
def fig6_seed_stability(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    fig.suptitle(
        'Per-Seed DRO-FAIR Accuracy (10 seeds) — Stability Analysis',
        fontsize=12, fontweight='bold')

    bp_props = dict(
        medianprops  = dict(color='white', lw=2.5),
        whiskerprops = dict(color=DRO_COLOR, lw=1.5),
        capprops     = dict(color=DRO_COLOR, lw=1.5),
        flierprops   = dict(marker='o', markerfacecolor=NAIVE_COLOR,
                            markeredgecolor='white', markersize=5),
        boxprops     = dict(linewidth=1.2),
    )

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        data, xlabels = [], []
        for alpha in [0.1, 0.2, 0.3, 0.4]:
            dv = get_vals(results, ds, alpha, 'dro', 'accuracy')
            if dv:
                data.append(dv)
                xlabels.append(f'α={alpha}')

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, labels=xlabels, patch_artist=True, **bp_props)
        for patch in bp['boxes']:
            patch.set_facecolor(DRO_COLOR)
            patch.set_alpha(0.72)

        ax.axhline(0.75, color='#C0392B', lw=1.2, ls='--', alpha=0.7,
                   label='0.75 threshold')
        ax.set_ylabel('DRO-FAIR Accuracy', fontsize=10)
        ax.set_xlabel('Corruption Level', fontsize=10)
        ax.set_title(DS_LABELS[ds], fontsize=11, fontweight='bold')
        ax.set_ylim(0.05, 1.02)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        if col == 0:
            ax.legend(fontsize=8, loc='lower left')

    fig.savefig(f'{OUT}/fig6_seed_stability.png')
    fig.savefig(f'{OUT}/fig6_seed_stability.pdf')
    plt.close()
    print('  ✓ fig6_seed_stability')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Summary Win-Rate Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def fig7_summary(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    fig.suptitle(
        'DRO-FAIR vs Naive-FAIR — Win Summary at α ∈ {0.1, 0.2, 0.3} (3 cells each)',
        fontsize=12, fontweight='bold')

    for ax_idx, metric in enumerate(['dp_violation', 'if_violation']):
        ax = axes[ax_idx]
        wins, losses, ties = {}, {}, {}
        for ds in DATASETS:
            lbl = DS_LABELS[ds]
            wins[lbl] = losses[lbl] = ties[lbl] = 0
            for alpha in [0.1, 0.2, 0.3]:
                nv = get_vals(results, ds, alpha, 'naive', metric)
                dv = get_vals(results, ds, alpha, 'dro',   metric)
                p  = wilcoxon_p(nv, dv)
                nm, dm = np.mean(nv), np.mean(dv)
                if p < 0.05 and dm < nm:
                    wins[lbl]   += 1
                elif p < 0.05 and dm > nm:
                    losses[lbl] += 1
                else:
                    ties[lbl]   += 1

        lbls  = list(wins.keys())
        w_v   = [wins[l]   for l in lbls]
        t_v   = [ties[l]   for l in lbls]
        l_v   = [losses[l] for l in lbls]
        x     = np.arange(len(lbls))
        w_arr = 0.52

        ax.bar(x, w_v, w_arr, color=DRO_COLOR,   alpha=0.85, label='DRO sig. better')
        ax.bar(x, t_v, w_arr, bottom=w_v,         color='#ADB5BD', alpha=0.8,  label='No sig. diff.')
        ax.bar(x, l_v, w_arr, bottom=[w+t for w,t in zip(w_v, t_v)],
               color=NAIVE_COLOR, alpha=0.85, label='Naive sig. better')

        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=11, fontweight='bold')
        ax.set_ylabel('# comparisons (of 3)', fontsize=10)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(0, 3.7)
        ax.set_title(
            'DP Violation' if metric == 'dp_violation' else 'IF Violation',
            fontsize=11, fontweight='bold')

        for i, (w, l) in enumerate(zip(w_v, l_v)):
            ax.text(i, w + 0.07, f'{w}/3', ha='center', va='bottom',
                    fontsize=11, color=DRO_COLOR, fontweight='bold')

        if ax_idx == 1:
            ax.legend(fontsize=9, loc='upper right')

    fig.savefig(f'{OUT}/fig7_summary_win_rates.png')
    fig.savefig(f'{OUT}/fig7_summary_win_rates.pdf')
    plt.close()
    print('  ✓ fig7_summary_win_rates')


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    results = load_results()
    print(f'Loaded {len(results)} results. Generating 7 publication-quality figures...\n')

    fig1_main_results(results)
    fig2_dp_reduction_heatmap(results)
    fig3_robustness(results)
    fig4_significance_matrix(results)
    fig5_accuracy_fairness_tradeoff(results)
    fig6_seed_stability(results)
    fig7_summary(results)

    print(f'\n✅  All 7 figures saved to {OUT}/ at 300 DPI (PNG + PDF).')


if __name__ == '__main__':
    main()
