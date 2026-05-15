#!/usr/bin/env python3
"""
Figure generation for DRO-FairML — ICML submission style.
Matches the typographic conventions of top ML venues:
  Computer Modern math, minimal chrome, proper error bars, restrained palette.
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

# ── ICML-style rcParams ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['CMU Serif', 'Computer Modern Roman', 'Latin Modern Roman',
                           'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset':   'cm',          # Computer Modern math
    'font.size':          8,
    'axes.titlesize':     9,
    'axes.labelsize':     8,
    'axes.titleweight':   'normal',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.6,
    'axes.labelpad':      3,
    'grid.alpha':         0.3,
    'grid.linewidth':     0.4,
    'grid.linestyle':     '--',
    'legend.frameon':     True,
    'legend.framealpha':  0.85,
    'legend.fontsize':    7,
    'legend.edgecolor':   '0.8',
    'legend.handlelength': 1.5,
    'legend.handletextpad': 0.4,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'xtick.minor.size':   1.5,
    'ytick.minor.size':   1.5,
    'xtick.major.pad':    2,
    'ytick.major.pad':    2,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'lines.linewidth':    1.2,
    'lines.markersize':   4,
    'errorbar.capsize':   2,
})

# Two-color palette (muted, print-safe, accessible)
C_NAIVE = '#c45232'   # muted red-brown
C_DRO   = '#2a6e3f'   # muted forest green
C_CLEAN = '#2b5c8a'   # steel blue
C_CORR  = '#d4880f'   # dark gold

ALPHAS   = [0.0, 0.1, 0.2, 0.3, 0.4]
DATASETS = ['adult', 'credit', 'lsac']
DS_LABEL = {'adult': 'Adult', 'credit': 'Credit', 'lsac': 'LSAC'}
OUT      = 'figures'

# ICML text width: 6.75 in (full), 3.25 in (single column)
FULL_W   = 6.75
COL_W    = 3.25


# ── Data helpers ─────────────────────────────────────────────────────────────
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


def _sig_marker(p, dro_m, naive_m):
    """Return significance marker only when DRO is better (lower)."""
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
    ylabels = {
        'accuracy':     'Accuracy',
        'dp_violation': r'$\Delta_{\mathrm{DP}}$',
        'if_violation': r'$\mathcal{L}_{\mathrm{IF}}$',
    }

    fig, axes = plt.subplots(3, 3, figsize=(FULL_W, 5.8))
    fig.subplots_adjust(hspace=0.48, wspace=0.40, top=0.93, bottom=0.07, left=0.10)

    x = np.array(ALPHAS)

    for row, ds in enumerate(DATASETS):
        for col, met in enumerate(metrics):
            ax = axes[row, col]
            nm, nse, dm, dse = [], [], [], []

            for alpha in ALPHAS:
                nv = _get(results, ds, alpha, 'naive', met)
                dv = _get(results, ds, alpha, 'dro',   met)
                n_m, n_s = _ms(nv)
                d_m, d_s = _ms(dv)
                nm.append(n_m); nse.append(n_s)
                dm.append(d_m); dse.append(d_s)

            nm  = np.array(nm);  nse = np.array(nse)
            dm  = np.array(dm);  dse = np.array(dse)

            # Error bars with caps (not shaded bands)
            ax.errorbar(x, nm, yerr=nse, fmt='o-', color=C_NAIVE,
                        ms=3.5, lw=1.1, capsize=2, capthick=0.8,
                        label='Naive-Fair', zorder=3)
            ax.errorbar(x, dm, yerr=dse, fmt='s-', color=C_DRO,
                        ms=3.5, lw=1.1, capsize=2, capthick=0.8,
                        label='DRO-Fair', zorder=3)

            # Significance markers
            for i, alpha in enumerate(ALPHAS):
                nv = _get(results, ds, alpha, 'naive', met)
                dv = _get(results, ds, alpha, 'dro',   met)
                p = _wilcox(nv, dv)
                s = _sig_marker(p, dm[i], nm[i])
                if s:
                    ymax = max(nm[i] + nse[i], dm[i] + dse[i])
                    span = ax.get_ylim()[1] - ax.get_ylim()[0]
                    if span == 0: span = 1
                    ax.text(alpha, ymax + 0.02 * span, s,
                            ha='center', va='bottom', fontsize=6,
                            color=C_DRO)

            ax.set_xticks(ALPHAS)
            ax.set_xlabel(r'$\alpha$')
            ax.set_ylabel(ylabels[met])
            ax.set_xlim(-0.03, 0.43)

            # Row labels on far left edge (no overlap with ylabel)
            if col == 0:
                ax.annotate(DS_LABEL[ds], xy=(-0.50, 0.5),
                            xycoords='axes fraction', fontsize=9,
                            fontweight='bold', ha='center', va='center',
                            rotation=90)

            # Column headers on top row
            if row == 0:
                ax.set_title(ylabels[met], fontsize=9)

            if row == 0 and col == 2:
                ax.legend(loc='best', fontsize=6.5)

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
            nm, dm = np.mean(nv), np.mean(dv)
            r = (nm - dm) / max(nm, 1e-9) * 100
            raw[i, j]  = r
            clip[i, j] = np.clip(r, -100, 100)
            pmat[i, j] = _wilcox(nv, dv)

    fig, ax = plt.subplots(figsize=(FULL_W * 0.65, 2.0))

    # Diverging: brown-white-green (not traffic-light red/green)
    cmap = LinearSegmentedColormap.from_list(
        'bwg', ['#a8432e', '#f7f7f7', '#2a6e3f'], N=256)
    im = ax.imshow(clip, cmap=cmap, vmin=-100, vmax=100, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, pad=0.03, fraction=0.04, shrink=0.9)
    cbar.set_label(r'$\Delta_{\mathrm{DP}}$ reduction (%)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xticks(range(5))
    ax.set_xticklabels([fr'$\alpha={a}$' for a in ALPHAS])
    ax.set_yticks(range(3))
    ax.set_yticklabels([DS_LABEL[d] for d in DATASETS])
    ax.tick_params(length=0)
    ax.grid(False)

    for i in range(3):
        for j in range(5):
            v = raw[i, j]
            if np.isnan(v): continue
            p  = pmat[i, j]
            st = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
            tc = 'white' if abs(clip[i, j]) > 55 else 'black'
            txt = f'{v:+.0f}%' if abs(v) <= 999 else f'{v:+.0f}'
            ax.text(j, i, f'{txt}\n{st}', ha='center', va='center',
                    fontsize=6.5, color=tc)

    ax.set_frame_on(False)
    _save(fig, 'fig2_dp_reduction_heatmap')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Clean vs corrupted test robustness
# ─────────────────────────────────────────────────────────────────────────────
def fig3(results):
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.6))
    fig.subplots_adjust(hspace=0.52, wspace=0.36, top=0.90, bottom=0.10, left=0.10)

    methods = ['naive', 'dro']
    mlabel  = {'naive': 'Naive-Fair', 'dro': 'DRO-Fair'}

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
                        ms=3, lw=1.0, capsize=1.5, capthick=0.7, label='Clean')
            ax.errorbar(x, cr_m, yerr=cr_se, fmt='s--', color=C_CORR,
                        ms=3, lw=1.0, capsize=1.5, capthick=0.7, label='Corrupted')

            ax.set_xticks(ALPHAS)
            ax.set_xlabel(r'$\alpha$')
            if col == 0:
                ax.set_ylabel(r'$\Delta_{\mathrm{DP}}$')

            if row == 0:
                ax.set_title(DS_LABEL[ds], fontsize=9)
            if col == 0:
                ax.annotate(mlabel[method], xy=(-0.52, 0.5),
                            xycoords='axes fraction', fontsize=8,
                            ha='center', va='center', rotation=90)

            if row == 0 and col == 2:
                ax.legend(fontsize=6, loc='best')

    _save(fig, 'fig3_robustness_clean_vs_corrupted')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Significance matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig4(results):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.78, 2.2))
    fig.subplots_adjust(wspace=0.30, top=0.85, bottom=0.18)

    c_win  = np.array([0.165, 0.431, 0.247, 0.80])  # DRO wins — muted green
    c_lose = np.array([0.784, 0.263, 0.196, 0.80])   # Naive wins — muted red
    c_tie  = np.array([0.88,  0.88,  0.88,  0.55])   # tie — light grey

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
                nm, dm = (np.mean(nv) if nv else np.nan), (np.mean(dv) if dv else np.nan)
                if   p < 0.05 and dm < nm: wgrid[i, j] =  1
                elif p < 0.05 and dm > nm: wgrid[i, j] = -1

        cmat = np.zeros((3, 5, 4))
        for i in range(3):
            for j in range(5):
                cmat[i, j] = (c_win if wgrid[i, j] == 1
                              else c_lose if wgrid[i, j] == -1
                              else c_tie)

        ax.imshow(cmat, aspect='auto', interpolation='nearest')
        ax.grid(False)
        ax.set_xticks(range(5))
        ax.set_xticklabels([fr'$\alpha\!=\!{a}$' for a in ALPHAS])
        ax.set_yticks(range(3))
        ax.set_yticklabels([DS_LABEL[d] for d in DATASETS])
        ax.tick_params(length=0)

        title = (r'$\Delta_{\mathrm{DP}}$' if met == 'dp_violation'
                 else r'$\mathcal{L}_{\mathrm{IF}}$')
        ax.set_title(title, fontsize=9)

        for i in range(3):
            for j in range(5):
                p = pgrid[i, j]
                w = wgrid[i, j]
                st = ('***' if p < 0.001 else '**' if p < 0.01
                      else '*' if p < 0.05 else 'n.s.')
                winner = ('DRO' if w == 1 else 'Naive' if w == -1 else '')
                tc = 'white' if w != 0 else '#555'
                line1 = f'$p$={p:.3f}'
                line2 = f'{st}  {winner}' if winner else st
                ax.text(j, i, f'{line1}\n{line2}',
                        ha='center', va='center', fontsize=5.5, color=tc)

        ax.set_frame_on(False)

    # Shared legend below
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c_win,  label='DRO sig. better'),
               Patch(facecolor=c_lose, label='Naive sig. better'),
               Patch(facecolor=c_tie,  label='Not significant')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    _save(fig, 'fig4_significance_matrix')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Accuracy vs DP tradeoff (Pareto-style scatter)
# ─────────────────────────────────────────────────────────────────────────────
def fig5(results):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4))
    fig.subplots_adjust(wspace=0.34, top=0.85, bottom=0.18, right=0.86)

    markers = {0.0: 'o', 0.1: 's', 0.2: '^', 0.3: 'D', 0.4: 'v'}

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        for alpha in ALPHAS:
            na = np.mean(_get(results, ds, alpha, 'naive', 'accuracy'))
            nd = np.mean(_get(results, ds, alpha, 'naive', 'dp_violation'))
            da = np.mean(_get(results, ds, alpha, 'dro',   'accuracy'))
            dd = np.mean(_get(results, ds, alpha, 'dro',   'dp_violation'))

            ax.scatter(na, nd, color=C_NAIVE, s=28, marker=markers[alpha],
                       zorder=3, edgecolors='white', linewidths=0.4)
            ax.scatter(da, dd, color=C_DRO,   s=28, marker=markers[alpha],
                       zorder=3, edgecolors='white', linewidths=0.4)

            # Label only DRO points — alternate offset to avoid cluster
            dy_off = 3 if (alpha * 10) % 2 == 0 else -8
            ax.annotate(fr'${alpha}$', (da, dd), fontsize=5, color='#555',
                        textcoords='offset points', xytext=(5, dy_off))

        ax.set_xlabel('Accuracy')
        if col == 0:
            ax.set_ylabel(r'$\Delta_{\mathrm{DP}}$')
        ax.set_title(DS_LABEL[ds], fontsize=9)

    # Compact legend in margin
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_NAIVE,
               ms=5, label='Naive-Fair'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_DRO,
               ms=5, label='DRO-Fair'),
    ]
    fig.legend(handles=handles, loc='center right', fontsize=6.5,
               frameon=True, bbox_to_anchor=(0.99, 0.55))

    _save(fig, 'fig5_accuracy_fairness_tradeoff')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Per-seed stability (boxplots)
# ─────────────────────────────────────────────────────────────────────────────
def fig6(results):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4))
    fig.subplots_adjust(wspace=0.30, top=0.85, bottom=0.18)

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        data, labels = [], []
        for alpha in [0.1, 0.2, 0.3, 0.4]:
            dv = _get(results, ds, alpha, 'dro', 'accuracy')
            if dv:
                data.append(dv)
                labels.append(fr'${alpha}$')

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data, labels=labels, patch_artist=True, widths=0.45,
            medianprops=dict(color='white', lw=1.5),
            whiskerprops=dict(color=C_DRO, lw=0.8),
            capprops=dict(color=C_DRO, lw=0.8),
            flierprops=dict(marker='.', markerfacecolor=C_NAIVE,
                            markeredgecolor='none', ms=3),
            boxprops=dict(linewidth=0.8),
        )
        for patch in bp['boxes']:
            patch.set_facecolor(C_DRO)
            patch.set_alpha(0.55)

        # Auto-scale y-axis with some padding
        all_vals = [v for d in data for v in d]
        ylo = max(0, min(all_vals) - 0.08)
        yhi = min(1.02, max(all_vals) + 0.04)
        ax.set_ylim(ylo, yhi)
        if ylo < 0.75 < yhi:
            ax.axhline(0.75, color='#999', lw=0.7, ls=':', zorder=1)
        ax.set_xlabel(r'$\alpha$')
        if col == 0:
            ax.set_ylabel('Accuracy')
        ax.set_title(DS_LABEL[ds], fontsize=9)

    _save(fig, 'fig6_seed_stability')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Win-rate summary (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig7(results):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 1.6, 2.3))
    fig.subplots_adjust(wspace=0.35, top=0.82, bottom=0.20)

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
                nm, dm = np.mean(nv), np.mean(dv)
                if   p < 0.05 and dm < nm: wins[lbl]   += 1
                elif p < 0.05 and dm > nm: losses[lbl] += 1
                else:                      ties[lbl]   += 1

        lbls = list(wins.keys())
        w = [wins[l] for l in lbls]
        t = [ties[l] for l in lbls]
        lo = [losses[l] for l in lbls]
        x = np.arange(len(lbls))
        bw = 0.48

        ax.bar(x, w, bw, color=C_DRO,   alpha=0.75, label='DRO wins')
        ax.bar(x, t, bw, bottom=w, color='#ccc', alpha=0.7, label='No sig. diff.')
        ax.bar(x, lo, bw, bottom=[a+b for a, b in zip(w, t)],
               color=C_NAIVE, alpha=0.75, label='Naive wins')

        # Annotate DRO wins count
        for i, wi in enumerate(w):
            if wi > 0:
                ax.text(i, wi - 0.15, str(wi), ha='center', va='top',
                        fontsize=7, color='white', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(lbls)
        ax.set_ylabel('Comparisons')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(0, 3.5)
        title = (r'$\Delta_{\mathrm{DP}}$' if met == 'dp_violation'
                 else r'$\mathcal{L}_{\mathrm{IF}}$')
        ax.set_title(title, fontsize=9)

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02))

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
