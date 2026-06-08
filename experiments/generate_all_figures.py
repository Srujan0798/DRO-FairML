#!/usr/bin/env python3
"""
Generate ALL publication figures from completed experimental results.

Reads every results/*.json file and produces:
  - fig11_lambda_diagnostic.pdf/png  (already from plot_lambda_diagnostic.py)
  - fig12_utkface_h3_lmax_cap.pdf/png
  - fig13_utkface_alpha_sweep.pdf/png
  - fig14_utkface_fairness_pgd.pdf/png
  - fig15_utkface_pixel_vs_feature.pdf/png
  - fig16_utkface_randinit.pdf/png
  - fig17_summary_dp_vs_alpha.pdf/png  (master comparison)

Usage:
    venv/bin/python3 experiments/generate_all_figures.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def group_by(runs, key):
    out = {}
    for r in (runs or []):
        out.setdefault(r.get(key), []).append(r)
    return out


def extract_dp(runs, method='dro'):
    """Extract DP violations for a method across runs."""
    vals = []
    for r in runs:
        m = r.get(method, {})
        if isinstance(m, dict):
            if 'dp_violation' in m:
                vals.append(m['dp_violation'])
            elif 'clean' in m and 'corrupted' in m:
                vals.append(m['corrupted']['dp_violation'])
    return np.array(vals)


def extract_acc(runs, method='dro'):
    """Extract accuracies for a method across runs."""
    vals = []
    for r in runs:
        m = r.get(method, {})
        if isinstance(m, dict):
            if 'accuracy' in m:
                vals.append(m['accuracy'])
            elif 'clean' in m and 'corrupted' in m:
                vals.append(m['corrupted']['accuracy'])
    return np.array(vals)


def plot_h3_lambda_cap(runs, out_stem):
    """Fig 12: H3 test — compare lambda_max=1.5 vs 0.5 on UTKFace."""
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in [(axes[0], 'dp', 'DP violation'),
                                (axes[1], 'acc', 'Accuracy')]:
        for lmax, color in [(1.5, '#d62728'), (0.5, '#2ca02c')]:
            lmax_runs = [r for r in runs if r.get('lambda_max') == lmax]
            alphas = sorted(set(r['alpha'] for r in lmax_runs if 'alpha' in r))
            means = []
            stds = []
            for a in alphas:
                subset = [r for r in lmax_runs if r['alpha'] == a]
                if metric == 'dp':
                    vals = extract_dp(subset, 'dro')
                else:
                    vals = extract_acc(subset, 'dro')
                means.append(np.mean(vals) if len(vals) else np.nan)
                stds.append(np.std(vals) if len(vals) else 0)
            ax.errorbar(alphas, means, yerr=stds, marker='o', capsize=4,
                        label=f'λ_max={lmax}', color=color, linewidth=2)
        ax.set_xlabel('α (corruption budget)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_title('DRO DP violation vs α (UTKFace, H3)')
    axes[1].set_title('DRO accuracy vs α (UTKFace, H3)')
    plt.suptitle('H3: Does capping λ_max restore DRO performance?', fontsize=12)
    plt.tight_layout()
    save(fig, out_stem)


def plot_alpha_sweep(runs, out_stem, title='UTKFace alpha sweep'):
    """Fig 13: Alpha sweep {0.3, 0.4} comparing Naive vs DRO."""
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in [(axes[0], 'dp', 'DP violation'),
                                (axes[1], 'acc', 'Accuracy')]:
        for method, color, label in [('naive', '#1f77b4', 'Naive'),
                                      ('dro', '#ff7f0e', 'DRO')]:
            alphas = sorted(set(r['alpha'] for r in runs if 'alpha' in r))
            means = []
            stds = []
            for a in alphas:
                subset = [r for r in runs if r['alpha'] == a]
                if metric == 'dp':
                    vals = extract_dp(subset, method)
                else:
                    vals = extract_acc(subset, method)
                means.append(np.mean(vals) if len(vals) else np.nan)
                stds.append(np.std(vals) if len(vals) else 0)
            ax.errorbar(alphas, means, yerr=stds, marker='s', capsize=4,
                        label=label, color=color, linewidth=2)
        ax.set_xlabel('α (corruption budget)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_title('DP violation vs α')
    axes[1].set_title('Accuracy vs α')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    save(fig, out_stem)


def plot_fairness_pgd(runs, out_stem):
    """Fig 14: FairnessTargetedPGD on UTKFace — 3 attack modes."""
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, attack in enumerate(['dp', 'if', 'combined']):
        ax = axes[idx]
        attack_runs = [r for r in runs if r.get('attack') == attack]
        if not attack_runs:
            ax.set_title(f'{attack}: no data')
            continue

        for method, color in [('naive', '#1f77b4'), ('dro', '#ff7f0e')]:
            alphas = sorted(set(r['alpha'] for r in attack_runs if 'alpha' in r))
            means = []
            stds = []
            for a in alphas:
                subset = [r for r in attack_runs if r['alpha'] == a]
                vals = extract_dp(subset, method)
                means.append(np.mean(vals) if len(vals) else np.nan)
                stds.append(np.std(vals) if len(vals) else 0)
            ax.errorbar(alphas, means, yerr=stds, marker='o', capsize=4,
                        label=method.capitalize(), color=color, linewidth=2)
        ax.set_xlabel('α')
        ax.set_ylabel('DP violation')
        ax.set_title(f'Attack: {attack}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('FairnessTargetedPGD on UTKFace', fontsize=12)
    plt.tight_layout()
    save(fig, out_stem)


def plot_pixel_vs_feature(feature_runs, pixel_runs, out_stem):
    """Fig 15: Feature-space vs pixel-space attacks (H2)."""
    if not feature_runs or not pixel_runs:
        return
    fig, ax = plt.subplots(figsize=(8, 6))

    # Feature-space (baseline)
    alphas_f = sorted(set(r['alpha'] for r in feature_runs if 'alpha' in r))
    means_f = []
    for a in alphas_f:
        subset = [r for r in feature_runs if r['alpha'] == a]
        vals = extract_dp(subset, 'dro')
        means_f.append(np.mean(vals) if len(vals) else np.nan)
    ax.plot(alphas_f, means_f, marker='o', label='Feature-space attack',
            color='#1f77b4', linewidth=2)

    # Pixel-space
    alphas_p = sorted(set(r['alpha'] for r in pixel_runs if 'alpha' in r))
    means_p = []
    for a in alphas_p:
        subset = [r for r in pixel_runs if r['alpha'] == a]
        vals = extract_dp(subset, 'dro')
        means_p.append(np.mean(vals) if len(vals) else np.nan)
    ax.plot(alphas_p, means_p, marker='s', label='Pixel-space attack',
            color='#d62728', linewidth=2)

    ax.set_xlabel('α (corruption budget)')
    ax.set_ylabel('DRO DP violation')
    ax.set_title('H2: Feature-space vs pixel-space PGD')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, out_stem)


def plot_randinit(randinit_runs, baseline_runs, out_stem):
    """Fig 16: Random-init ResNet18 (H1) vs pretrained baseline."""
    if not randinit_runs or not baseline_runs:
        return
    fig, ax = plt.subplots(figsize=(8, 6))

    # Baseline (pretrained)
    alphas_b = sorted(set(r['alpha'] for r in baseline_runs if 'alpha' in r))
    means_b = []
    for a in alphas_b:
        subset = [r for r in baseline_runs if r['alpha'] == a]
        vals = extract_dp(subset, 'dro')
        means_b.append(np.mean(vals) if len(vals) else np.nan)
    ax.plot(alphas_b, means_b, marker='o', label='Pretrained ResNet18',
            color='#1f77b4', linewidth=2)

    # Random init
    alphas_r = sorted(set(r['alpha'] for r in randinit_runs if 'alpha' in r))
    means_r = []
    for a in alphas_r:
        subset = [r for r in randinit_runs if r['alpha'] == a]
        vals = extract_dp(subset, 'dro')
        means_r.append(np.mean(vals) if len(vals) else np.nan)
    ax.plot(alphas_r, means_r, marker='s', label='Random-init ResNet18',
            color='#2ca02c', linewidth=2)

    ax.set_xlabel('α (corruption budget)')
    ax.set_ylabel('DRO DP violation')
    ax.set_title('H1: Pretrained vs random-init backbone')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, out_stem)


def plot_master_comparison(all_data, out_stem):
    """Fig 17: Master figure — DRO DP vs alpha across all settings."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tabular datasets
    tabular_colors = {'adult': '#1f77b4', 'credit': '#ff7f0e', 'lsac': '#2ca02c'}
    for ds in ['adult', 'credit', 'lsac']:
        ds_runs = [r for r in (all_data.get('fairness_pgd') or [])
                   if r.get('dataset') == ds and r.get('method') == 'dro']
        alphas = sorted(set(r['alpha'] for r in ds_runs if 'alpha' in r))
        means = []
        for a in alphas:
            subset = [r for r in ds_runs if r['alpha'] == a]
            vals = [r.get('dp_clean', np.nan) for r in subset]
            means.append(np.mean(vals) if vals else np.nan)
        if means:
            ax.plot(alphas, means, marker='o', label=f'{ds.capitalize()} (tabular)',
                    color=tabular_colors.get(ds), linewidth=2, linestyle='--')

    # UTKFace baseline
    utk_runs = all_data.get('utkface_baseline') or []
    alphas = sorted(set(r['alpha'] for r in utk_runs if 'alpha' in r))
    means = []
    for a in alphas:
        subset = [r for r in utk_runs if r['alpha'] == a]
        vals = extract_dp(subset, 'dro')
        means.append(np.mean(vals) if len(vals) else np.nan)
    if means:
        ax.plot(alphas, means, marker='s', label='UTKFace (images)',
                color='#d62728', linewidth=2)

    ax.set_xlabel('α (corruption budget)')
    ax.set_ylabel('DRO DP violation')
    ax.set_title('DRO fairness violation across domains')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, out_stem)


def save(fig, stem):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf = os.path.join(FIGURES_DIR, f'{stem}.pdf')
    png = os.path.join(FIGURES_DIR, f'{stem}.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {pdf}\nSaved {png}")


def main():
    print("Loading results...")
    data = {
        'fairness_pgd': load(os.path.join(RESULTS_DIR, 'fairness_pgd_results.json')),
        'utkface_baseline': load(os.path.join(RESULTS_DIR, 'utkface_results.json')),
        'utkface_lambda_max_cap': load(os.path.join(RESULTS_DIR, 'utkface_lambda_max_cap.json')),
        'utkface_alpha_sweep': load(os.path.join(RESULTS_DIR, 'utkface_alpha_sweep.json')),
        'utkface_fairness_pgd': load(os.path.join(RESULTS_DIR, 'utkface_fairness_pgd.json')),
        'utkface_pixel_pgd': load(os.path.join(RESULTS_DIR, 'utkface_pixel_pgd.json')),
        'utkface_randinit': load(os.path.join(RESULTS_DIR, 'utkface_randinit.json')),
        'lambda_diagnostic': load(os.path.join(RESULTS_DIR, 'lambda_diagnostic.json')),
    }

    print("\nGenerating figures...")
    plot_h3_lambda_cap(data['utkface_lambda_max_cap'], 'fig12_utkface_h3_lmax_cap')
    plot_alpha_sweep(data['utkface_alpha_sweep'], 'fig13_utkface_alpha_sweep',
                     title='UTKFace alpha sweep {0.3, 0.4}')
    plot_fairness_pgd(data['utkface_fairness_pgd'], 'fig14_utkface_fairness_pgd')
    plot_pixel_vs_feature(data['utkface_baseline'], data['utkface_pixel_pgd'],
                          'fig15_utkface_pixel_vs_feature')
    plot_randinit(data['utkface_randinit'], data['utkface_baseline'],
                  'fig16_utkface_randinit')
    plot_master_comparison(data, 'fig17_summary_dp_vs_alpha')

    print("\nDone. Check figures/ for new plots.")


if __name__ == '__main__':
    main()
