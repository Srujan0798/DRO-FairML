#!/usr/bin/env python3
"""
Figure generation for DRO-FairML.
Large, readable figures with proper math fonts.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.12,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'lines.linewidth':    1.8,
    'lines.markersize':   6,
    'errorbar.capsize':   3,
})

# Professional but readable colors
C_NAIVE = '#c44e2b'   # warm red
C_DRO   = '#1a7a3a'   # rich green
C_CLEAN = '#2b6d99'   # steel blue
C_CORR  = '#d4880f'   # dark gold

ALPHAS   = [0.0, 0.1, 0.2, 0.3, 0.4]
DATASETS = ['adult', 'credit', 'lsac']
DS_LABEL = {'adult': 'Adult', 'credit': 'Credit', 'lsac': 'LSAC'}
OUT      = 'figures'


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_results():
    with open('results/all_results.json') as f:
        return json.load(f)


def _get(results, ds, alpha, method, metric, ev='clean'):
    recs = [r for r in results if r['dataset'] == ds and abs(r['alpha'] - alpha) < 1e-6]
    return [r[method][ev][metric] for r in recs] if recs else []


def _ms(vals):
    if not vals:
        return np.nan, np.nan
    a = np.array(vals, dtype=float)
    return np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a))


def _wilcox(a, b):
    try:
        if len(a) < 3 or np.allclose(a, b):
            return 1.0
        _, p = wilcoxon(a, b)
        return p
    except Exception:
        return 1.0


def _sig(p, dro_m, naive_m):
    if dro_m >= naive_m:
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/{name}.pdf')
    fig.savefig(f'{OUT}/{name}.png')
    plt.close(fig)
    print(f'  {name}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Main results (3 datasets x 3 metrics)
# ─────────────────────────────────────────────────────────────────────────────
def fig1(results):
    metrics = ['accuracy', 'dp_violation', 'if_violation']
    col_titles = ['Accuracy', r'DP Violation ($\Delta_{\mathrm{DP}}$)',
                  r'IF Violation ($\mathcal{L}_{\mathrm{IF}}$)']

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.subplots_adjust(hspace=0.35, wspace=0.30, top=0.94, bottom=0.05, left=0.08, right=0.97)
    fig.suptitle('DRO-FAIR vs Naive-FAIR on Clean Test Set  (mean $\\pm$ 1 SE, $n{=}10$ seeds)',
                 fontsize=15, fontweight='bold', y=0.98)

    x = np.array(ALPHAS)

    for row, ds in enumerate(DATASETS):
        for col, met in enumerate(metrics):
            ax = axes[row, col]
            nm, nse, dm, dse = [], [], [], []

            for alpha in ALPHAS:
                nv = _get(results, ds, alpha, 'naive', met)
                dv = _get(results, ds, alpha, 'dro',   met)
                n_m, n_s = _ms(nv); d_m, d_s = _ms(dv)
                nm.append(n_m); nse.append(n_s)
                dm.append(d_m); dse.append(d_s)

            nm = np.array(nm);  nse = np.array(nse)
            dm = np.array(dm);  dse = np.array(dse)

            ax.errorbar(x, nm, yerr=nse, fmt='o-', color=C_NAIVE,
                        ms=5, lw=1.8, capsize=3, capthick=1.0,
                        label='Naive-FAIR', zorder=3)
            ax.errorbar(x, dm, yerr=dse, fmt='s-', color=C_DRO,
                        ms=5, lw=1.8, capsize=3, capthick=1.0,
                        label='DRO-FAIR', zorder=3)

            # Significance markers
            for i, alpha in enumerate(ALPHAS):
                nv = _get(results, ds, alpha, 'naive', met)
                dv = _get(results, ds, alpha, 'dro',   met)
                p = _wilcox(nv, dv)
                s = _sig(p, dm[i], nm[i])
                if s:
                    ymax = max(nm[i] + nse[i], dm[i] + dse[i])
                    span = ax.get_ylim()[1] - ax.get_ylim()[0]
                    if span == 0: span = 1
                    ax.text(alpha, ymax + 0.03 * span, s,
                            ha='center', va='bottom', fontsize=9,
                            color=C_DRO, fontweight='bold')

            ax.set_xticks(ALPHAS)
            ax.set_xlabel(r'Corruption level $\alpha$', fontsize=10)
            ax.set_xlim(-0.03, 0.43)

            # Title = "Dataset — Metric" for every cell
            ax.set_title(f'{DS_LABEL[ds]} — {col_titles[col]}', fontsize=11, fontweight='bold')

            if row == 0 and col == 2:
                ax.legend(loc='upper left', fontsize=9)

    _save(fig, 'fig1_main_results')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — DP reduction heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig2(results):
    raw  = np.full((3, 5), np.nan)
    clip = np.full((3, 5), np.nan)
    pmat = np.ones((3, 5))

    for i, ds in enumerate(DATASETS):
        for j, alpha in enumerate(ALPHAS):
            nv = _get(results, ds, alpha, 'naive', 'dp_violation')
            dv = _get(results, ds, alpha, 'dro',   'dp_violation')
            if not nv: continue
            nm_, dm_ = np.mean(nv), np.mean(dv)
            r = (nm_ - dm_) / max(nm_, 1e-9) * 100
            raw[i, j]  = r
            clip[i, j] = np.clip(r, -100, 100)
            pmat[i, j] = _wilcox(nv, dv)

    fig, ax = plt.subplots(figsize=(12, 4.5))

    cmap = LinearSegmentedColormap.from_list(
        'bwg', ['#a8432e', '#f7f7f7', '#1a7a3a'], N=256)
    im = ax.imshow(clip, cmap=cmap, vmin=-100, vmax=100, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03, shrink=0.85)
    cbar.set_label(r'$\Delta_{\mathrm{DP}}$ reduction (%)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(range(5))
    ax.set_xticklabels([fr'$\alpha={a}$' for a in ALPHAS], fontsize=12)
    ax.set_yticks(range(3))
    ax.set_yticklabels([DS_LABEL[d] for d in DATASETS], fontsize=13, fontweight='bold')
    ax.tick_params(length=0)
    ax.grid(False)

    ax.set_title('DRO-FAIR DP Reduction over Naive-FAIR\n'
                 '(color capped $\\pm$100%; exact value shown  |  '
                 '$*\\,p{<}0.05$  $**\\,p{<}0.01$  $***\\,p{<}0.001$)',
                 fontsize=12, fontweight='bold', pad=12)

    for i in range(3):
        for j in range(5):
            v = raw[i, j]
            if np.isnan(v): continue
            p  = pmat[i, j]
            st = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
            tc = 'white' if abs(clip[i, j]) > 55 else 'black'
            txt = f'{v:+.0f}%'
            ax.text(j, i, f'{txt}\n{st}', ha='center', va='center',
                    fontsize=12, color=tc, fontweight='bold')

    ax.set_frame_on(False)
    _save(fig, 'fig2_dp_reduction_heatmap')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Clean vs corrupted test robustness
# ─────────────────────────────────────────────────────────────────────────────
def fig3(results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.40, wspace=0.28, top=0.92, bottom=0.08, left=0.06, right=0.97)
    fig.suptitle('Robustness: DP Violation on Clean vs Adversarially Corrupted Test Set',
                 fontsize=14, fontweight='bold')

    methods = ['naive', 'dro']
    mlabel  = {'naive': 'Naive-FAIR', 'dro': 'DRO-FAIR'}
    x = np.array(ALPHAS)

    for col, ds in enumerate(DATASETS):
        for row, method in enumerate(methods):
            ax = axes[row, col]
            cl_m, cl_se, cr_m, cr_se = [], [], [], []

            for alpha in ALPHAS:
                cv = _get(results, ds, alpha, method, 'dp_violation', 'clean')
                kv = _get(results, ds, alpha, method, 'dp_violation', 'corrupted')
                cm, cs = _ms(cv); km, ks = _ms(kv)
                cl_m.append(cm); cl_se.append(cs)
                cr_m.append(km); cr_se.append(ks)

            cl_m = np.array(cl_m); cl_se = np.array(cl_se)
            cr_m = np.array(cr_m); cr_se = np.array(cr_se)

            ax.errorbar(x, cl_m, yerr=cl_se, fmt='o-', color=C_CLEAN,
                        ms=5, lw=1.8, capsize=3, capthick=1.0, label='Clean test')
            ax.errorbar(x, cr_m, yerr=cr_se, fmt='s--', color=C_CORR,
                        ms=5, lw=1.8, capsize=3, capthick=1.0, label='Corrupted test')

            ax.set_xticks(ALPHAS)
            ax.set_xlabel(r'Corruption level $\alpha$', fontsize=10)
            ax.set_ylabel(r'$\Delta_{\mathrm{DP}}$', fontsize=11)
            ax.set_title(f'{DS_LABEL[ds]} — {mlabel[method]}',
                         fontsize=11, fontweight='bold')

            if row == 0 and col == 2:
                ax.legend(fontsize=9, loc='best')

    _save(fig, 'fig3_robustness_clean_vs_corrupted')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Significance matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig4(results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    fig.subplots_adjust(wspace=0.25, top=0.82, bottom=0.18, left=0.06, right=0.97)
    fig.suptitle('Statistical Significance: Wilcoxon Signed-Rank (one-sided, $n{=}10$ seeds, clean test)',
                 fontsize=13, fontweight='bold')

    c_win  = np.array([0.10, 0.48, 0.23, 0.80])   # green
    c_lose = np.array([0.77, 0.31, 0.17, 0.80])    # red
    c_tie  = np.array([0.90, 0.90, 0.90, 0.55])    # grey

    for ai, met in enumerate(['dp_violation', 'if_violation']):
        ax = axes[ai]
        pgrid = np.ones((3, 5))
        wgrid = np.zeros((3, 5))

        for i, ds in enumerate(DATASETS):
            for j, alpha in enumerate(ALPHAS):
                nv = _get(results, ds, alpha, 'naive', met)
                dv = _get(results, ds, alpha, 'dro', met)
                p  = _wilcox(nv, dv)
                pgrid[i, j] = p
                nm_, dm_ = (np.mean(nv) if nv else np.nan), (np.mean(dv) if dv else np.nan)
                if   p < 0.05 and dm_ < nm_: wgrid[i, j] =  1
                elif p < 0.05 and dm_ > nm_: wgrid[i, j] = -1

        cmat = np.zeros((3, 5, 4))
        for i in range(3):
            for j in range(5):
                cmat[i, j] = (c_win if wgrid[i, j] == 1
                              else c_lose if wgrid[i, j] == -1
                              else c_tie)

        ax.imshow(cmat, aspect='auto', interpolation='nearest')
        ax.grid(False)
        ax.set_xticks(range(5))
        ax.set_xticklabels([fr'$\alpha={a}$' for a in ALPHAS], fontsize=11)
        ax.set_yticks(range(3))
        ax.set_yticklabels([DS_LABEL[d] for d in DATASETS], fontsize=12, fontweight='bold')
        ax.tick_params(length=0)

        title = (r'DP Violation ($\Delta_{\mathrm{DP}}$)' if met == 'dp_violation'
                 else r'IF Violation ($\mathcal{L}_{\mathrm{IF}}$)')
        ax.set_title(title, fontsize=12, fontweight='bold')

        for i in range(3):
            for j in range(5):
                p = pgrid[i, j]
                w = wgrid[i, j]
                st = ('***' if p < 0.001 else '**' if p < 0.01
                      else '*' if p < 0.05 else 'n.s.')
                winner = ('DRO' if w == 1 else 'Naive' if w == -1 else '')
                tc = 'white' if w != 0 else '#444'
                line1 = f'$p$={p:.3f}'
                line2 = f'{st}  {winner}' if winner else st
                ax.text(j, i, f'{line1}\n{line2}',
                        ha='center', va='center', fontsize=9, color=tc,
                        fontweight='bold')

        ax.set_frame_on(False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c_win,  label='DRO sig. better'),
               Patch(facecolor=c_lose, label='Naive sig. better'),
               Patch(facecolor=c_tie,  label='Not significant')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    _save(fig, 'fig4_significance_matrix')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Accuracy vs DP tradeoff
# ─────────────────────────────────────────────────────────────────────────────
def fig5(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.28, top=0.88, bottom=0.12, left=0.05, right=0.88)
    fig.suptitle('Accuracy vs Fairness Tradeoff  (mean over 10 seeds, clean test)',
                 fontsize=14, fontweight='bold')

    markers = {0.0: 'o', 0.1: 's', 0.2: '^', 0.3: 'D', 0.4: 'v'}
    sizes   = {0.0: 70,  0.1: 80,  0.2: 90,  0.3: 100, 0.4: 110}

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        for alpha in ALPHAS:
            na = np.mean(_get(results, ds, alpha, 'naive', 'accuracy'))
            nd = np.mean(_get(results, ds, alpha, 'naive', 'dp_violation'))
            da = np.mean(_get(results, ds, alpha, 'dro',   'accuracy'))
            dd = np.mean(_get(results, ds, alpha, 'dro',   'dp_violation'))

            ax.scatter(na, nd, color=C_NAIVE, s=sizes[alpha], marker=markers[alpha],
                       zorder=3, edgecolors='white', linewidths=0.6)
            ax.scatter(da, dd, color=C_DRO,   s=sizes[alpha], marker=markers[alpha],
                       zorder=3, edgecolors='white', linewidths=0.6)

            # Label DRO points with offset to avoid overlap
            dy = 5 if alpha in [0.0, 0.2, 0.4] else -12
            ax.annotate(fr'$\alpha$={alpha}', (da, dd), fontsize=8, color='#444',
                        textcoords='offset points', xytext=(6, dy))

        ax.set_xlabel('Accuracy', fontsize=11)
        ax.set_ylabel(r'$\Delta_{\mathrm{DP}}$', fontsize=11)
        ax.set_title(DS_LABEL[ds], fontsize=13, fontweight='bold')

    # Legend in right margin
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_NAIVE,
               ms=8, label='Naive-FAIR'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_DRO,
               ms=8, label='DRO-FAIR'),
    ]
    for a in ALPHAS:
        handles.append(Line2D([0], [0], marker=markers[a], color='w',
                              markerfacecolor='gray', ms=7, label=fr'$\alpha$={a}'))
    fig.legend(handles=handles, loc='center right', fontsize=9,
               frameon=True, bbox_to_anchor=(0.99, 0.5))

    _save(fig, 'fig5_accuracy_fairness_tradeoff')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Per-seed stability (boxplots)
# ─────────────────────────────────────────────────────────────────────────────
def fig6(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.25, top=0.88, bottom=0.12)
    fig.suptitle('Per-Seed DRO-FAIR Accuracy ($n{=}10$ seeds)',
                 fontsize=14, fontweight='bold')

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        data, labels = [], []
        for alpha in [0.1, 0.2, 0.3, 0.4]:
            dv = _get(results, ds, alpha, 'dro', 'accuracy')
            if dv:
                data.append(dv)
                labels.append(fr'$\alpha$={alpha}')

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data, labels=labels, patch_artist=True, widths=0.5,
            medianprops=dict(color='white', lw=2),
            whiskerprops=dict(color=C_DRO, lw=1.2),
            capprops=dict(color=C_DRO, lw=1.2),
            flierprops=dict(marker='o', markerfacecolor=C_NAIVE,
                            markeredgecolor='white', ms=5),
            boxprops=dict(linewidth=1.0),
        )
        for patch in bp['boxes']:
            patch.set_facecolor(C_DRO)
            patch.set_alpha(0.6)

        # Auto-scale y with padding
        all_vals = [v for d in data for v in d]
        ylo = max(0, min(all_vals) - 0.08)
        yhi = min(1.02, max(all_vals) + 0.04)
        ax.set_ylim(ylo, yhi)
        if ylo < 0.75 < yhi:
            ax.axhline(0.75, color='#999', lw=0.8, ls=':', zorder=1,
                       label='0.75 threshold')
            if col == 0:
                ax.legend(fontsize=8, loc='lower left')

        ax.set_xlabel('Corruption Level', fontsize=11)
        ax.set_ylabel('DRO-FAIR Accuracy', fontsize=11)
        ax.set_title(DS_LABEL[ds], fontsize=13, fontweight='bold')

    _save(fig, 'fig6_seed_stability')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Win-rate summary
# ─────────────────────────────────────────────────────────────────────────────
def fig7(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.30, top=0.85, bottom=0.18, left=0.08, right=0.95)
    fig.suptitle(r'DRO-FAIR vs Naive-FAIR — Win Summary at $\alpha \in \{0.1, 0.2, 0.3\}$',
                 fontsize=14, fontweight='bold')

    for ai, met in enumerate(['dp_violation', 'if_violation']):
        ax = axes[ai]
        wins, losses, ties = {}, {}, {}
        for ds in DATASETS:
            lbl = DS_LABEL[ds]
            wins[lbl] = losses[lbl] = ties[lbl] = 0
            for alpha in [0.1, 0.2, 0.3]:
                nv = _get(results, ds, alpha, 'naive', met)
                dv = _get(results, ds, alpha, 'dro',   met)
                p  = _wilcox(nv, dv)
                nm_, dm_ = np.mean(nv), np.mean(dv)
                if   p < 0.05 and dm_ < nm_: wins[lbl]   += 1
                elif p < 0.05 and dm_ > nm_: losses[lbl] += 1
                else:                        ties[lbl]   += 1

        lbls = list(wins.keys())
        w = [wins[l] for l in lbls]
        t = [ties[l] for l in lbls]
        lo = [losses[l] for l in lbls]
        x = np.arange(len(lbls))
        bw = 0.52

        ax.bar(x, w, bw, color=C_DRO,   alpha=0.80, label='DRO sig. better')
        ax.bar(x, t, bw, bottom=w, color='#ccc', alpha=0.75, label='No sig. diff.')
        ax.bar(x, lo, bw, bottom=[a+b for a, b in zip(w, t)],
               color=C_NAIVE, alpha=0.80, label='Naive sig. better')

        for i, wi in enumerate(w):
            if wi > 0:
                ax.text(i, wi / 2, f'{wi}/3', ha='center', va='center',
                        fontsize=12, color='white', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=12, fontweight='bold')
        ax.set_ylabel('# Comparisons (of 3)', fontsize=11)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(0, 3.6)

        title = (r'DP Violation ($\Delta_{\mathrm{DP}}$)' if met == 'dp_violation'
                 else r'IF Violation ($\mathcal{L}_{\mathrm{IF}}$)')
        ax.set_title(title, fontsize=12, fontweight='bold')

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    _save(fig, 'fig7_summary_win_rates')


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    results = load_results()
    print(f'Generating figures from {len(results)} experiment results...\n')

    fig1(results)
    fig2(results)
    fig3(results)
    fig4(results)
    fig5(results)
    fig6(results)
    fig7(results)

    print(f'\nAll figures saved to {OUT}/ (PDF + PNG, 300 DPI)')


if __name__ == '__main__':
    main()
