#!/usr/bin/env python3
"""
Generate all required figures for Kuldeep meeting (figD1-figD10).

This script creates the exact figures requested in the task:
- figD1: Constant-predictor accuracy (Adult, DP attack)
- figD2: Constant-predictor DP violation
- figD3: Constant-predictor IF violation
- figD4: Acc-DP tradeoff vs constant predictor
- figD5: Val-loss convergence (loss)
- figD6: Val-loss convergence (accuracy)
- figD7: Val-loss convergence (DP)
- figD8: Lambda heatmap accuracy \alpha=0.3
- figD9: Lambda heatmap accuracy \alpha=0.4
- figD10: Wilcoxon significance table
"""

import os
import sys
import json
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from experiments.loaders import constant_predictor_acc, load_canonical_tau1
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon

RESULTS_DIR = os.path.join(ROOT, 'results')

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'Latin Modern Roman',
                   'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.labelpad': 4,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.4,
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fontsize': 10,
    'legend.edgecolor': '0.8',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.12,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})

# Colors
C_NAIVE = '#c44e2b'   # warm red
C_DRO = '#1a7a3a'     # rich green
C_CLEAN = '#2b6d99'   # steel blue
C_CORR = '#d4880f'    # dark gold

ALPHAS = [0.1, 0.2, 0.3, 0.4]
DATASETS = ['adult']
DS_LABEL = {'adult': 'Adult'}
OUT = 'figures'

# Constant predictor baseline
CONSTANT_PREDICTOR_ACC = constant_predictor_acc('adult')


def _refuse_stale(path):
    """Raise if a path points at the archived results tree."""
    norm = os.path.normpath(path).replace('\\', '/')
    if 'stale_archived' in norm:
        raise RuntimeError(
            f"Refusing to read stale archived results: {path}. "
            "Use results/canonical_tau1.json via load_canonical_tau1()."
        )


def load_results():
    """Load canonical tau=1 grid only (fail-loud). Never reads stale_archived/."""
    rows = load_canonical_tau1()
    out = []
    for row in rows:
        r = dict(row)
        # Canonical is fixed tau=1.0; keep explicit for multi-tau plotting loops.
        if r.get('tau') is None:
            r['tau'] = 1.0
        out.append(r)
    if not out:
        raise RuntimeError(
            "results/canonical_tau1.json loaded but is empty. "
            "Run experiments/run_canonical.py before generate_all_deliverables."
        )
    return out


def load_tau1_summary():
    """Build a per-(dataset, attack, alpha, method) summary from canonical."""
    rows = load_results()
    buckets = defaultdict(list)
    for r in rows:
        key = (r['dataset'], r['attack'], float(r['alpha']), r['method'], float(r.get('tau', 1.0)))
        buckets[key].append(r)
    records = []
    for (ds, attack, alpha, method, tau), group in buckets.items():
        acc = [g['acc_clean'] for g in group]
        dp = [g['dp_clean'] for g in group]
        ifm = [g['if_clean'] for g in group]
        records.append({
            'dataset': ds,
            'attack': attack,
            'alpha': alpha,
            'method': method,
            'tau': tau,
            'acc_mean': float(np.mean(acc)),
            'dp_mean': float(np.mean(dp)),
            'if_mean': float(np.mean(ifm)),
            'n': len(group),
        })
    return pd.DataFrame(records)


def load_lambda_grid():
    """Load live lambda grid JSON only (never stale_archived). Fail loud if missing."""
    candidates = [
        os.path.join(RESULTS_DIR, 'lambda_lr_grid.json'),
        os.path.join(RESULTS_DIR, 'lambda_grid_comprehensive.json'),
    ]
    for path in candidates:
        _refuse_stale(path)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if not data:
                raise RuntimeError(f"Lambda grid file is empty: {path}")
            return data
    raise FileNotFoundError(
        "No live lambda grid found. Expected one of:\n  - "
        + "\n  - ".join(candidates)
        + "\n(Do not use results/stale_archived/.)"
    )


def load_canonical_wilcoxon():
    """Load live wilcoxon CSV or compute it from canonical_tau1 (fail-loud)."""
    live = os.path.join(RESULTS_DIR, 'canonical_wilcoxon.csv')
    _refuse_stale(live)
    if os.path.exists(live):
        return pd.read_csv(live)
    # Compute on the fly from canonical — never fall back to stale_archived CSV.
    from experiments.compute_canonical_wilcoxon import compute_wilcoxon
    rows = load_canonical_tau1()
    df = compute_wilcoxon(rows)
    if df is None or df.empty:
        raise RuntimeError(
            "Could not compute Wilcoxon table from canonical_tau1.json "
            "(need paired naive/dro seeds)."
        )
    return df


def _ms(vals):
    """Compute mean and standard error."""
    if not vals:
        return np.nan, np.nan
    a = np.array(vals, dtype=float)
    return np.mean(a), np.std(a, ddof=1) / np.sqrt(len(a))


def _wilcox(a, b):
    """Wilcoxon signed-rank test."""
    try:
        if len(a) < 3 or np.allclose(a, b):
            return 1.0
        _, p = wilcoxon(a, b)
        return p
    except Exception:
        return 1.0


def _sig(p, dro_m, naive_m):
    """Get significance marker."""
    if dro_m >= naive_m:
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _save(fig, name):
    """Save figure as PDF and PNG."""
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/{name}.pdf')
    fig.savefig(f'{OUT}/{name}.png')
    plt.close(fig)
    print(f'  {name}')


def figD1_constant_predictor_accuracy():
    """Plot 1: Constant predictor accuracy (Adult DP attack preferred): x=\alpha, y=acc, lines for different tau + Naive + horizontal 0.752 line + caption about \alpha≤0.2."""
    print("Generating figD1: Constant-predictor accuracy")
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    results = load_results()
    ALPHAS = [0.1, 0.2, 0.3, 0.4]
    
    acc_data = defaultdict(list)
    for row in results:
        if row.get('dataset') == 'adult' and row.get('attack') == 'dp' and float(row.get('alpha',0)) in ALPHAS:
            key = (float(row['alpha']), int(float(row.get('tau', 1))), row['method'])
            acc_data[key].append(row['acc_clean'])
    
    # Canonical is tau=1 only; do not load multi-tau from stale_archived.
    taus_to_plot = sorted({int(float(r.get('tau', 1))) for r in results}) or [1]
    tau_colors = {1: '#1a7a3a', 5: '#2ca02c', 10: '#ff7f0e', 100: '#9467bd'}
    
    plotted_labels = set()
    for tau in taus_to_plot:
        for mname, mcolor, mmarker, mls in [('dro', None, 's', '-'), ('naive', '#c44e2b', 'o', '--')]:
            xs = []
            ys = []
            yerr = []
            for alpha in sorted(ALPHAS):
                key = (alpha, tau, mname)
                if key in acc_data:
                    vals = acc_data[key]
                    if vals:
                        m = sum(vals) / len(vals)
                        n = len(vals)
                        se = (sum((v - m)**2 for v in vals) / max(1, n-1))**0.5 / (n**0.5) if n > 1 else 0.0
                        xs.append(alpha)
                        ys.append(m)
                        yerr.append(se)
            if xs:
                color = tau_colors.get(tau, '#1a7a3a') if mname == 'dro' else mcolor
                lbl = f"DRO (τ={tau})" if mname=='dro' else f"Naive (τ={tau})"
                if lbl in plotted_labels:
                    lbl = None
                else:
                    plotted_labels.add(lbl)
                ax.errorbar(xs, ys, yerr=yerr if any(e>0 for e in yerr) else None,
                            fmt=mmarker + mls, color=color, label=lbl,
                            capsize=2, markersize=5, linewidth=1.5)
    
    ax.axhline(y=CONSTANT_PREDICTOR_ACC, color='gray', linestyle='--', linewidth=1.5,
               label=f'Constant predictor ({CONSTANT_PREDICTOR_ACC})')
    
    ax.set_xlabel(r"$alpha$ (corruption level)", fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Constant-predictor accuracy figure (Adult, DP attack)\n'
                 r'x=$\alpha$ ; lines for $\tau$ + Naive + horiz 0.752', fontsize=12)
    ax.set_xticks(ALPHAS)
    ax.set_ylim(0.45, 0.86)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25, linestyle='--')
    
    cap = 'For \alpha≤0.2 DRO (τ=1) stays near/above 0.752; \alpha≥0.3 all variants drop below (use τ=1 for best DP in defensible regime).'
    fig.text(0.5, 0.01, cap, ha='center', va='bottom', fontsize=9, style='italic')
    _save(fig, 'figD1_constant_predictor_accuracy')


def figD2_constant_predictor_dp():
    """Similar for DP: x=\alpha, y=dp, lines different tau + Naive."""
    print("Generating figD2: Constant-predictor DP")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    results = load_results()
    ALPHAS = [0.1, 0.2, 0.3, 0.4]
    dp_data = defaultdict(list)
    for row in results:
        if row.get('dataset') == 'adult' and row.get('attack') == 'dp' and float(row.get('alpha',0)) in ALPHAS:
            key = (float(row['alpha']), int(float(row.get('tau', 1))), row['method'])
            dp_data[key].append(row['dp_clean'])
    
    taus_to_plot = sorted({int(float(r.get('tau', 1))) for r in results}) or [1]
    tau_colors = {1: '#1a7a3a', 10: '#ff7f0e', 100: '#9467bd'}
    for tau in taus_to_plot:
        for mname, mcolor, mmarker, mls in [('dro', None, 's', '-'), ('naive', '#c44e2b', 'o', '--')]:
            xs, ys, yerr = [], [], []
            for alpha in sorted(ALPHAS):
                key = (alpha, tau, mname)
                if key in dp_data:
                    vals = dp_data[key]
                    if vals:
                        m = sum(vals) / len(vals)
                        n = len(vals)
                        se = (sum((v - m)**2 for v in vals) / max(1, n-1))**0.5 / (n**0.5) if n > 1 else 0.0
                        xs.append(alpha); ys.append(m); yerr.append(se)
            if xs:
                color = tau_colors.get(tau, '#1a7a3a') if mname=='dro' else mcolor
                lbl = f"DRO (τ={tau})" if mname=='dro' else f"Naive (τ={tau})"
                ax.errorbar(xs, ys, yerr=yerr if any(e>0 for e in yerr) else None,
                            fmt=mmarker + mls, color=color, label=lbl,
                            capsize=2, markersize=5, linewidth=1.5)
    ax.set_xlabel(r"$alpha$ (corruption level)", fontsize=11)
    ax.set_ylabel('DP Violation', fontsize=11)
    ax.set_title(r"Constant-predictor DP (Adult, DP attack) x=$\alpha$", fontsize=12)
    ax.set_xticks(ALPHAS)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--')
    _save(fig, 'figD2_constant_predictor_dp')


def figD3_constant_predictor_if():
    """Similar for IF."""
    print("Generating figD3: Constant-predictor IF")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    results = load_results()
    ALPHAS = [0.1, 0.2, 0.3, 0.4]
    if_data = defaultdict(list)
    for row in results:
        if row.get('dataset') == 'adult' and row.get('attack') == 'dp' and float(row.get('alpha',0)) in ALPHAS:
            key = (float(row['alpha']), int(float(row.get('tau', 1))), row['method'])
            if_data[key].append(row.get('if_clean', 0.0))
    
    taus_to_plot = sorted({int(float(r.get('tau', 1))) for r in results}) or [1]
    tau_colors = {1: '#1a7a3a', 10: '#ff7f0e', 100: '#9467bd'}
    for tau in taus_to_plot:
        for mname, mcolor, mmarker, mls in [('dro', None, 's', '-'), ('naive', '#c44e2b', 'o', '--')]:
            xs, ys, yerr = [], [], []
            for alpha in sorted(ALPHAS):
                key = (alpha, tau, mname)
                if key in if_data:
                    vals = if_data[key]
                    if vals:
                        m = sum(vals) / len(vals)
                        n = len(vals)
                        se = (sum((v - m)**2 for v in vals) / max(1, n-1))**0.5 / (n**0.5) if n > 1 else 0.0
                        xs.append(alpha); ys.append(m); yerr.append(se)
            if xs:
                color = tau_colors.get(tau, '#1a7a3a') if mname=='dro' else mcolor
                lbl = f"DRO (τ={tau})" if mname=='dro' else f"Naive (τ={tau})"
                ax.errorbar(xs, ys, yerr=yerr if any(e>0 for e in yerr) else None,
                            fmt=mmarker + mls, color=color, label=lbl,
                            capsize=2, markersize=5, linewidth=1.5)
    ax.set_xlabel(r"$alpha$ (corruption level)", fontsize=11)
    ax.set_ylabel('IF Violation', fontsize=11)
    ax.set_title(r"Constant-predictor IF (Adult, DP attack) x=$\alpha$", fontsize=12)
    ax.set_xticks(ALPHAS)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--')
    _save(fig, 'figD3_constant_predictor_if')


def figD4_tradeoff_vs_constant_predictor():
    """Acc vs DP tradeoff scatter with constant predictor marked (0, 0.752)."""
    print("Generating figD4: Acc-DP tradeoff vs constant predictor")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    df = load_tau1_summary()
    plotted = False
    for _, row in df.iterrows():
        if row['dataset'] == 'adult' and row['attack'] == 'dp' and abs(float(row.get('tau',1)) - 1) < 0.5:
            alpha = row['alpha']
            method = row['method']
            acc = row['acc_mean']
            dp = row['dp_mean']
            color = C_DRO if method == 'dro' else C_NAIVE
            marker = 's' if method == 'dro' else 'o'
            lbl = f'{method.upper()} (τ=1, \alpha={alpha})' if not plotted else None
            ax.scatter(dp, acc, color=color, marker=marker, s=90, edgecolor='k', linewidth=0.5, label=lbl, alpha=0.85)
            ax.annotate(f'\alpha{alpha}', (dp, acc), xytext=(3,3), textcoords='offset points', fontsize=7)
            plotted = True
    
    ax.scatter(0, CONSTANT_PREDICTOR_ACC, marker='*', s=220, color='red', edgecolors='k', linewidth=1, label='Constant predictor (0, 0.752)', zorder=5)
    
    ax.set_xlabel('DP Violation', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Acc vs DP tradeoff scatter with constant predictor (0, 0.752)', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.01, 0.4)
    ax.set_ylim(0.45, 0.86)
    
    fig.text(0.5, 0.01, 'Constant predictor acc=0.752, DP~0 ; points below horiz line are degenerate at high \alpha.', ha='center', fontsize=9, style='italic')
    _save(fig, 'figD4_tradeoff_vs_constant_predictor')



def _load_live_individual_histories():
    """Load per-run history from live results/individual only (never stale_archived)."""
    results_dir = os.path.join(RESULTS_DIR, 'individual')
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"Per-run history directory missing: {results_dir}. "
            "Canonical flat rows have no epoch histories. "
            "Do not fall back to results/stale_archived/individual. "
            "Re-run with history logging or skip figD5–D7."
        )
    epochs_data = {}
    for filename in os.listdir(results_dir):
        if not (filename.endswith('.json') and 'adult' in filename):
            continue
        path = os.path.join(results_dir, filename)
        _refuse_stale(path)
        with open(path) as f:
            data = json.load(f)
        key = (data['dataset'], data['alpha'], data['seed'])
        epochs_data.setdefault(key, [])
        if 'history' in data:
            epochs_data[key].extend(data['history'])
        elif 'total_time' in data:
            epochs_data[key].append(data['total_time'])
    if not epochs_data:
        raise FileNotFoundError(
            f"No adult individual run JSONs with history under {results_dir}."
        )
    return epochs_data


def figD5_convergence_loss():
    """Plot 5: Val-loss convergence (requires live results/individual)."""
    print("Generating figD5: Val-loss convergence")
    epochs_data = _load_live_individual_histories()
    fig, ax = plt.subplots(figsize=(10, 6))
    for (dataset, alpha, seed), history in epochs_data.items():
        if dataset == 'adult' and alpha in [0.3, 0.4]:
            ax.plot(history, label=f'α={alpha}, seed={seed}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Val-loss convergence (α=0.3, 0.4 DRO runs)', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    _save(fig, 'figD5_convergence_loss')


def figD6_convergence_acc():
    """Plot 6: Val-accuracy convergence (requires live results/individual)."""
    print("Generating figD6: Val-accuracy convergence")
    epochs_data = _load_live_individual_histories()
    fig, ax = plt.subplots(figsize=(10, 6))
    for (dataset, alpha, seed), history in epochs_data.items():
        if dataset == 'adult' and alpha in [0.3, 0.4]:
            ax.plot(history, label=f'α={alpha}, seed={seed}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Val-accuracy convergence (α=0.3, 0.4 DRO runs)', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    _save(fig, 'figD6_convergence_acc')


def figD7_convergence_dp():
    """Plot 7: Val-DP convergence (requires live results/individual)."""
    print("Generating figD7: Val-DP convergence")
    epochs_data = _load_live_individual_histories()
    fig, ax = plt.subplots(figsize=(10, 6))
    for (dataset, alpha, seed), history in epochs_data.items():
        if dataset == 'adult' and alpha in [0.3, 0.4]:
            ax.plot(history, label=f'α={alpha}, seed={seed}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation DP Violation', fontsize=12)
    ax.set_title('Val-DP convergence (α=0.3, 0.4 DRO runs)', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    _save(fig, 'figD7_convergence_dp')


def figD8_lambda_heatmap_acc_alpha0_3():
    """Plot 8: Lambda heatmap accuracy \alpha=0.3."""
    print("Generating figD8: Lambda heatmap accuracy \alpha=0.3")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Load lambda grid data
    lambda_data = load_lambda_grid()
    df = pd.DataFrame(lambda_data)
    
    # Filter for \alpha=0.3, attack=dp
    sub = df[(df['alpha'] == 0.3) & (df['attack'] == 'dp')]
    
    # Create pivot table
    pivot = sub.pivot_table(
        values='acc', index='lambda_init', columns='lr_lambda', aggfunc='mean'
    )
    
    # Sort axes
    pivot = pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    
    # Plot heatmap
    data = pivot.values
    im = ax.imshow(data, cmap='YlOrRd_r', aspect='auto', origin='lower',
                   vmin=0.5, vmax=0.85)
    
    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if val < 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=text_color, fontweight='bold')
            
            # Highlight cells meeting threshold
            if val >= CONSTANT_PREDICTOR_ACC:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor='green',
                                     linewidth=2.5, linestyle='--')
                ax.add_patch(rect)
    
    # Formatting
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x}' for x in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{x}' for x in pivot.index], fontsize=10)
    ax.set_xlabel(r'$\eta_\lambda$ (lr_lambda)', fontsize=11)
    ax.set_ylabel(r'$\lambda_0$ (lambda_init)', fontsize=11)
    ax.set_title('Accuracy — $\\lambda$ grid (Adult, $\\alpha$=0.3, attack=DP)\nGreen dashed = cells with acc $\\geq$ 0.752',
                 fontsize=14)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Accuracy', fontsize=10)
    
    fig.tight_layout()
    _save(fig, 'figD8_lambda_heatmap_acc_alpha0_3')


def figD9_lambda_heatmap_acc_alpha0_4():
    """Plot 9: Lambda heatmap accuracy \alpha=0.4."""
    print("Generating figD9: Lambda heatmap accuracy \alpha=0.4")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Load lambda grid data
    lambda_data = load_lambda_grid()
    df = pd.DataFrame(lambda_data)
    
    # Filter for \alpha=0.4, attack=dp
    sub = df[(df['alpha'] == 0.4) & (df['attack'] == 'dp')]
    
    # Create pivot table
    pivot = sub.pivot_table(
        values='acc', index='lambda_init', columns='lr_lambda', aggfunc='mean'
    )
    
    # Sort axes
    pivot = pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    
    # Plot heatmap
    data = pivot.values
    im = ax.imshow(data, cmap='YlOrRd_r', aspect='auto', origin='lower',
                   vmin=0.5, vmax=0.85)
    
    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if val < 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=text_color, fontweight='bold')
            
            # Highlight cells meeting threshold
            if val >= CONSTANT_PREDICTOR_ACC:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor='green',
                                     linewidth=2.5, linestyle='--')
                ax.add_patch(rect)
    
    # Formatting
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x}' for x in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{x}' for x in pivot.index], fontsize=10)
    ax.set_xlabel(r'$\eta_\lambda$ (lr_lambda)', fontsize=11)
    ax.set_ylabel(r'$\lambda_0$ (lambda_init)', fontsize=11)
    ax.set_title('Accuracy — $\\lambda$ grid (Adult, $\\alpha$=0.4, attack=DP)\nGreen dashed = cells with acc $\\geq$ 0.752',
                 fontsize=14)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Accuracy', fontsize=10)
    
    fig.tight_layout()
    _save(fig, 'figD9_lambda_heatmap_acc_alpha0_4')


def figD10_final_wilcoxon_table():
    """Plot 10: Final Wilcoxon significance table."""
    print("Generating figD10: Final Wilcoxon significance table")
    
    # Load canonical Wilcoxon data
    wilcoxon_df = load_canonical_wilcoxon()
    
    # Create a simple table representation
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Dataset', 'Attack', '\alpha', 'Acc p-value', 'DP p-value', 'IF p-value']
    
    for _, row in wilcoxon_df.iterrows():
        table_data.append([
            row['dataset'],
            row['attack'],
            f"{row['alpha']:.3f}",
            f"{row['dp_pvalue']:.4f}" if not pd.isna(row['dp_pvalue']) else 'N/A',
            f"{row['if_pvalue']:.4f}" if not pd.isna(row['if_pvalue']) else 'N/A',
            f"{row['if_pvalue']:.4f}" if not pd.isna(row['if_pvalue']) else 'N/A'
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight significant cells
    for i, row in enumerate(table_data):
        for j, cell_value in enumerate(row):
            if j >= 3 and cell_value != 'N/A':
                p_value = float(cell_value)
                if p_value < 0.05:
                    # Highlight significant cells
                    for key in table.get_celld():
                        if key[0] == i and key[1] == j:
                            table.get_celld()[key].set_facecolor('#ffcccc')
    
    ax.set_title('Wilcoxon Significance: DRO vs Naive (canonical 540)', fontsize=16, fontweight='bold')
    
    fig.tight_layout()
    _save(fig, 'figD10_final_wilcoxon_table')


def main():
    print("AGENT C FINAL: Generate deliverable figures from CANONICAL (no stale_archived)")
    print("=" * 80)
    print("Source of truth: results/canonical_tau1.json via load_canonical_tau1()")
    print("Refusing any path under results/stale_archived/")
    print("=" * 80)

    errors = []

    def _run(label, fn):
        print(f"\n{label}")
        print("-" * 80)
        try:
            fn()
        except Exception as e:
            msg = f"{label}: {type(e).__name__}: {e}"
            print(f"  FAIL (loud): {msg}")
            errors.append(msg)

    _run("TASK 1a: figD1 constant-predictor accuracy", figD1_constant_predictor_accuracy)
    _run("TASK 1b: figD2 constant-predictor DP", figD2_constant_predictor_dp)
    _run("TASK 1c: figD3 constant-predictor IF", figD3_constant_predictor_if)
    _run("TASK 2: figD4 Acc-DP tradeoff", figD4_tradeoff_vs_constant_predictor)
    _run("TASK 3a: figD5 convergence loss", figD5_convergence_loss)
    _run("TASK 3b: figD6 convergence acc", figD6_convergence_acc)
    _run("TASK 3c: figD7 convergence DP", figD7_convergence_dp)
    _run("TASK 4a: figD8 lambda heatmap α=0.3", figD8_lambda_heatmap_acc_alpha0_3)
    _run("TASK 4b: figD9 lambda heatmap α=0.4", figD9_lambda_heatmap_acc_alpha0_4)
    _run("TASK 5: figD10 Wilcoxon table", figD10_final_wilcoxon_table)

    print("\n" + "=" * 80)
    if errors:
        print(f"COMPLETED WITH {len(errors)} FAILURE(S) (fail-loud; no stale fallback):")
        for e in errors:
            print(f"  - {e}")
        print("Canonical-based figures that succeeded were written under figures/.")
        print("Fix missing live inputs or use generate_final_figures.py / compute_canonical_wilcoxon.py.")
        sys.exit(1)
    print("AGENT C MILESTONE: All required figures generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
