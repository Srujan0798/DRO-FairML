#!/usr/bin/env python3
"""
Analyze Fairness-Targeted PGD experiment results.
Generates summary table, figures, and Wilcoxon tests.

Usage:
    python3 experiments/analyze_fairness_pgd.py
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(path='results/fairness_pgd_results.json'):
    with open(path) as f:
        return json.load(f)


def summarize(results):
    """Compute mean ± SE for each (dataset, attack, method, alpha) group."""
    df = pd.DataFrame(results)
    grouped = df.groupby(['dataset', 'attack', 'method', 'alpha'])

    summary = []
    for (dataset, attack, method, alpha), group in grouped:
        summary.append({
            'dataset': dataset,
            'attack': attack,
            'method': method,
            'alpha': alpha,
            'n_seeds': len(group),
            'acc_mean': group['acc_clean'].mean(),
            'acc_se': group['acc_clean'].std(ddof=1) / np.sqrt(len(group)),
            'dp_mean': group['dp_clean'].mean(),
            'dp_se': group['dp_clean'].std(ddof=1) / np.sqrt(len(group)),
            'if_mean': group['if_clean'].mean(),
            'if_se': group['if_clean'].std(ddof=1) / np.sqrt(len(group)),
        })

    return pd.DataFrame(summary)


def wilcoxon_tests(df):
    """Run Wilcoxon signed-rank: Naive DP > DRO DP? (one-sided)."""
    records = []

    for (dataset, attack, alpha), group in df.groupby(['dataset', 'attack', 'alpha']):
        naive = group[group['method'] == 'naive']
        dro = group[group['method'] == 'dro']

        if len(naive) == 0 or len(dro) == 0:
            continue

        # Merge by seed
        merged = pd.merge(naive[['seed', 'dp_clean', 'if_clean', 'acc_clean']],
                          dro[['seed', 'dp_clean', 'if_clean', 'acc_clean']],
                          on='seed', suffixes=('_naive', '_dro'))

        if len(merged) < 3:
            continue

        diff_dp = merged['dp_clean_naive'] - merged['dp_clean_dro']
        diff_if = merged['if_clean_naive'] - merged['if_clean_dro']

        try:
            _, p_dp = wilcoxon(diff_dp, alternative='greater')
        except Exception:
            p_dp = 1.0

        try:
            _, p_if = wilcoxon(diff_if, alternative='greater')
        except Exception:
            p_if = 1.0

        records.append({
            'dataset': dataset,
            'attack': attack,
            'alpha': alpha,
            'n': len(merged),
            'dp_naive_mean': merged['dp_clean_naive'].mean(),
            'dp_dro_mean': merged['dp_clean_dro'].mean(),
            'dp_reduction_pct': 100 * (merged['dp_clean_naive'].mean() - merged['dp_clean_dro'].mean()) / (merged['dp_clean_naive'].mean() + 1e-8),
            'dp_pvalue': p_dp,
            'dp_wins': (diff_dp > 0).sum(),
            'if_naive_mean': merged['if_clean_naive'].mean(),
            'if_dro_mean': merged['if_clean_dro'].mean(),
            'if_reduction_pct': 100 * (merged['if_clean_naive'].mean() - merged['if_clean_dro'].mean()) / (merged['if_clean_naive'].mean() + 1e-8),
            'if_pvalue': p_if,
            'if_wins': (diff_if > 0).sum(),
        })

    return pd.DataFrame(records)


def plot_attack_comparison(summary_df, outpath='figures/fig8_fairness_pgd_comparison.pdf'):
    """Bar plot: Naive vs DRO DP under each attack mode, per dataset."""
    datasets = summary_df['dataset'].unique()
    attacks = ['dp', 'if', 'combined']
    alphas = sorted(summary_df['alpha'].unique())

    fig, axes = plt.subplots(len(datasets), len(alphas), figsize=(4*len(alphas), 3.5*len(datasets)), squeeze=False)

    for i, dataset in enumerate(datasets):
        for j, alpha in enumerate(alphas):
            ax = axes[i][j]
            sub = summary_df[(summary_df['dataset'] == dataset) & (summary_df['alpha'] == alpha)]

            x = np.arange(len(attacks))
            width = 0.35

            naive_dp = [sub[(sub['attack'] == a) & (sub['method'] == 'naive')]['dp_mean'].values[0]
                        if len(sub[(sub['attack'] == a) & (sub['method'] == 'naive')]) > 0 else 0
                        for a in attacks]
            dro_dp = [sub[(sub['attack'] == a) & (sub['method'] == 'dro')]['dp_mean'].values[0]
                      if len(sub[(sub['attack'] == a) & (sub['method'] == 'dro')]) > 0 else 0
                      for a in attacks]

            ax.bar(x - width/2, naive_dp, width, label='Naive-FAIR', color='#e74c3c', alpha=0.8)
            ax.bar(x + width/2, dro_dp, width, label='DRO-FAIR', color='#2ecc71', alpha=0.8)

            ax.set_ylabel('DP Violation')
            ax.set_title(f'{dataset.upper()}  α={alpha}')
            ax.set_xticks(x)
            ax.set_xticklabels(['DP-Attack', 'IF-Attack', 'Joint'])
            ax.legend()
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.savefig(outpath.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outpath}")


def plot_alpha_curves(summary_df, outpath='figures/fig9_fairness_pgd_curves.pdf'):
    """Line plot: DP vs alpha for each attack and method."""
    datasets = summary_df['dataset'].unique()
    attacks = ['dp', 'if', 'combined']

    fig, axes = plt.subplots(len(datasets), len(attacks), figsize=(4*len(attacks), 3.5*len(datasets)), squeeze=False)

    for i, dataset in enumerate(datasets):
        for j, attack in enumerate(attacks):
            ax = axes[i][j]
            sub = summary_df[(summary_df['dataset'] == dataset) & (summary_df['attack'] == attack)]

            for method, color in [('naive', '#e74c3c'), ('dro', '#2ecc71')]:
                msub = sub[sub['method'] == method].sort_values('alpha')
                if len(msub) > 0:
                    ax.errorbar(msub['alpha'], msub['dp_mean'], yerr=msub['dp_se'],
                                marker='o', label=f'{method.upper()}-FAIR', color=color, capsize=3)

            ax.set_xlabel('Corruption α')
            ax.set_ylabel('DP Violation')
            ax.set_title(f'{dataset.upper()} — {attack.upper()}-Attack')
            ax.legend()
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.savefig(outpath.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {outpath}")


def main():
    results = load_results()
    print(f"Loaded {len(results)} experiment results")

    summary = summarize(results)
    summary.to_csv('results/fairness_pgd_summary.csv', index=False)
    print("Saved summary to results/fairness_pgd_summary.csv")

    tests = wilcoxon_tests(pd.DataFrame(results))
    tests.to_csv('results/fairness_pgd_wilcoxon.csv', index=False)
    print("Saved Wilcoxon tests to results/fairness_pgd_wilcoxon.csv")

    print("\n=== WILCOXON TESTS ===")
    for _, row in tests.iterrows():
        sig = "***" if row['dp_pvalue'] < 0.05 else ""
        print(f"{row['dataset']:8s} {row['attack']:8s} α={row['alpha']:.1f}: "
              f"DP Naive={row['dp_naive_mean']:.4f} DRO={row['dp_dro_mean']:.4f} "
              f"reduction={row['dp_reduction_pct']:+.1f}% p={row['dp_pvalue']:.3f} {sig}")

    plot_attack_comparison(summary)
    plot_alpha_curves(summary)

    print("\nDone. Figures saved to figures/fig8_* and figures/fig9_*")


if __name__ == '__main__':
    main()
