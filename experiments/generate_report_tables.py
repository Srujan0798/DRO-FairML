#!/usr/bin/env python3
"""
Auto-generate LaTeX table fragments for report.tex.
Reads results/fairness_pgd_results.json and produces report/sections/*.tex files.

Usage:
    venv/bin/python3 experiments/generate_report_tables.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_PATH = 'results/fairness_pgd_results.json'
OUT_DIR = 'report/sections'


def load_results(path=RESULTS_PATH):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def summarize(df):
    grouped = df.groupby(['dataset', 'attack', 'alpha', 'method'])
    rows = []
    for (ds, atk, alpha, method), group in grouped:
        rows.append({
            'dataset': ds, 'attack': atk, 'alpha': alpha, 'method': method, 'n': len(group),
            'acc_mean': group['acc_clean'].mean(),
            'acc_se': group['acc_clean'].std(ddof=1) / np.sqrt(len(group)),
            'dp_mean': group['dp_clean'].mean(),
            'dp_se': group['dp_clean'].std(ddof=1) / np.sqrt(len(group)),
            'if_mean': group['if_clean'].mean(),
            'if_se': group['if_clean'].std(ddof=1) / np.sqrt(len(group)),
        })
    return pd.DataFrame(rows)


def wilcoxon_table(df):
    records = []
    for (ds, atk, alpha), group in df.groupby(['dataset', 'attack', 'alpha']):
        naive = group[group['method'] == 'naive']
        dro = group[group['method'] == 'dro']
        if len(naive) == 0 or len(dro) == 0:
            continue
        merged = pd.merge(
            naive[['seed', 'dp_clean', 'if_clean', 'acc_clean']],
            dro[['seed', 'dp_clean', 'if_clean', 'acc_clean']],
            on='seed', suffixes=('_naive', '_dro')
        )
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
            'dataset': ds, 'attack': atk, 'alpha': alpha, 'n': len(merged),
            'dp_naive_mean': merged['dp_clean_naive'].mean(),
            'dp_dro_mean': merged['dp_clean_dro'].mean(),
            'dp_reduction_pct': 100 * diff_dp.mean() / (merged['dp_clean_naive'].mean() + 1e-8),
            'dp_pvalue': p_dp,
            'if_naive_mean': merged['if_clean_naive'].mean(),
            'if_dro_mean': merged['if_clean_dro'].mean(),
            'if_reduction_pct': 100 * diff_if.mean() / (merged['if_clean_naive'].mean() + 1e-8),
            'if_pvalue': p_if,
        })
    return pd.DataFrame(records)


def generate_main_results_tex(summary_df, outpath):
    """Report-style main results table with Acc/DP/IF per dataset and alpha."""
    datasets = ['adult', 'credit', 'lsac']
    lines = ["% AUTO-GENERATED: do not edit manually", "\\begin{tabular}{llcccccc}", "\\toprule"]
    lines.append("Dataset & $\\alpha$ & Acc (Naive) & Acc (DRO) & DP (Naive) & DP (DRO) & IF (Naive) & IF (DRO) \\\\")
    lines.append("\\midrule")

    for ds in datasets:
        for alpha in sorted(summary_df['alpha'].unique()):
            # Aggregate across attacks for a high-level view, or pick dp attack
            sub = summary_df[(summary_df['dataset'] == ds) & (summary_df['alpha'] == alpha) & (summary_df['attack'] == 'dp')]
            naive = sub[sub['method'] == 'naive']
            dro = sub[sub['method'] == 'dro']
            if len(naive) == 0 or len(dro) == 0:
                continue
            acc_n = f"{naive['acc_mean'].values[0]:.3f} $\\pm$ {naive['acc_se'].values[0]:.3f}"
            acc_d = f"{dro['acc_mean'].values[0]:.3f} $\\pm$ {dro['acc_se'].values[0]:.3f}"
            dp_n = f"{naive['dp_mean'].values[0]:.4f} $\\pm$ {naive['dp_se'].values[0]:.4f}"
            dp_d = f"\\textbf{{{dro['dp_mean'].values[0]:.4f} $\\pm$ {dro['dp_se'].values[0]:.4f}}}"
            if_n = f"{naive['if_mean'].values[0]:.4f} $\\pm$ {naive['if_se'].values[0]:.4f}"
            if_d = f"\\textbf{{{dro['if_mean'].values[0]:.4f} $\\pm$ {dro['if_se'].values[0]:.4f}}}"
            lines.append(f"{ds.capitalize()} & {alpha:.1f} & {acc_n} & {acc_d} & {dp_n} & {dp_d} & {if_n} & {if_d} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def generate_wilcoxon_tex(wdf, outpath):
    """Wilcoxon table for report."""
    lines = ["% AUTO-GENERATED: do not edit manually", "\\begin{tabular}{llrrrr}", "\\toprule"]
    lines.append("Dataset & $\\alpha$ & $\\Delta$DP\\% & $p$-value & $\\Delta$IF\\% & $p$-value \\\\")
    lines.append("\\midrule")

    for _, row in wdf.iterrows():
        if row['attack'] != 'dp':
            continue
        dp_sig = "***" if row['dp_pvalue'] < 0.05 else ""
        if_sig = "***" if row['if_pvalue'] < 0.05 else ""
        lines.append(
            f"{row['dataset'].capitalize()} & {row['alpha']:.1f} & "
            f"{row['dp_reduction_pct']:.1f}\\% & {row['dp_pvalue']:.3f}{dp_sig} & "
            f"{row['if_reduction_pct']:.1f}\\% & {row['if_pvalue']:.3f}{if_sig} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def generate_pgd_results_tex(summary_df, outpath):
    """Fairness-Targeted PGD attack results table."""
    datasets = ['adult', 'credit', 'lsac']
    attacks = ['dp', 'if', 'combined']
    lines = ["% AUTO-GENERATED: do not edit manually", "\\begin{tabular}{lllcccc}", "\\toprule"]
    lines.append("Dataset & Attack & $\\alpha$ & DP Naive & DP DRO & Reduction & $p$-value \\\\")
    lines.append("\\midrule")

    for ds in datasets:
        for atk in attacks:
            for alpha in sorted(summary_df['alpha'].unique()):
                sub = summary_df[(summary_df['dataset'] == ds) & (summary_df['attack'] == atk) & (summary_df['alpha'] == alpha)]
                naive = sub[sub['method'] == 'naive']
                dro = sub[sub['method'] == 'dro']
                if len(naive) == 0 or len(dro) == 0:
                    continue
                dp_n = naive['dp_mean'].values[0]
                dp_d = dro['dp_mean'].values[0]
                red = 100 * (dp_n - dp_d) / (dp_n + 1e-8)
                lines.append(f"{ds.capitalize()} & {atk.upper()} & {alpha:.1f} & {dp_n:.4f} & {dp_d:.4f} & {red:.1f}\\% & TBD \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def main():
    results = load_results()
    print(f"Loaded {len(results)} results")

    if len(results) == 0:
        print("No results yet — generating placeholder files")
        os.makedirs(OUT_DIR, exist_ok=True)
        for name in ['auto_generated_main_results.tex', 'auto_generated_wilcoxon.tex', 'auto_generated_pgd.tex']:
            path = os.path.join(OUT_DIR, name)
            with open(path, 'w') as f:
                f.write(f"% PLACEHOLDER: run experiments to generate {name}\\n")
            print(f"Saved placeholder {path}")
        return

    df = pd.DataFrame(results)
    summary = summarize(df)
    wdf = wilcoxon_table(df)

    generate_main_results_tex(summary, os.path.join(OUT_DIR, 'auto_generated_main_results.tex'))
    generate_wilcoxon_tex(wdf, os.path.join(OUT_DIR, 'auto_generated_wilcoxon.tex'))
    generate_pgd_results_tex(summary, os.path.join(OUT_DIR, 'auto_generated_pgd.tex'))

    print("\nDone. Report tables generated in report/sections/")


if __name__ == '__main__':
    main()
