#!/usr/bin/env python3
"""
Auto-generate LaTeX table fragments for report.tex.
Reads results/tau1_summary.csv (tau=1 canonical data) and produces report/sections/*.tex files.

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

TAU1_SUMMARY_PATH = 'results/tau1_summary.csv'
WILCOXON_PATH = 'results/tau1_wilcoxon.csv'
OUT_DIR = 'report/sections'
PAPER_OUT_DIR = 'paper/auto_generated'


def load_tau1_summary(path=TAU1_SUMMARY_PATH):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Filter to tau=1 only (canonical data)
    df = df[df['tau'] == 1.0].copy()
    print(f"Loaded {len(df)} tau=1 rows from {path}")
    return df


def load_wilcoxon(path=WILCOXON_PATH):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} wilcoxon rows from {path}")
    return df


def summarize(df):
    """tau1_summary.csv already has precomputed stats; just return it grouped."""
    return df


def wilcoxon_table_from_csv(path=WILCOXON_PATH):
    """Load precomputed Wilcoxon results from tau1_wilcoxon.csv."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def generate_main_results_tex(summary_df, outpath):
    """Report-style main results table with Acc/DP/IF per dataset and alpha.
    
    Uses tau=1 data from tau1_summary.csv (precomputed mean±se).
    """
    datasets = ['adult', 'credit', 'lsac']
    lines = ["% AUTO-GENERATED from tau1_summary.csv (tau=1 canonical): do not edit manually",
             "\\begin{tabular}{llcccccc}", "\\toprule"]
    lines.append("Dataset & $\\alpha$ & Acc (Naive) & Acc (DRO) & DP (Naive) & DP (DRO) & IF (Naive) & IF (DRO) \\\\")
    lines.append("\\midrule")

    for ds in datasets:
        for alpha in sorted(summary_df['alpha'].unique()):
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
    """Wilcoxon table for report from precomputed tau1_wilcoxon.csv."""
    lines = ["% AUTO-GENERATED from tau1_wilcoxon.csv (tau=1 canonical): do not edit manually",
             "\\begin{tabular}{llrrrr}", "\\toprule"]
    lines.append("Dataset & $\\alpha$ & $\\Delta$DP\\% & $p$-value & $\\Delta$IF\\% & $p$-value \\\\")
    lines.append("\\midrule")

    for _, row in wdf.iterrows():
        if row['attack'] != 'dp':
            continue
        dp_red = 100 * row['dp_diff_mean'] / (row['dp_naive_mean'] + 1e-8)
        if_red = 100 * row['if_diff_mean'] / (row['if_naive_mean'] + 1e-8) if row['if_naive_mean'] > 1e-6 else 0.0
        dp_sig = "***" if row['dp_pvalue'] < 0.05 else ""
        if_sig = "***" if row['if_pvalue'] < 0.05 else ""
        lines.append(
            f"{row['dataset'].capitalize()} & {row['alpha']:.1f} & "
            f"{dp_red:.1f}\\% & {row['dp_pvalue']:.3f}{dp_sig} & "
            f"{if_red:.1f}\\% & {row['if_pvalue']:.3f}{if_sig} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def generate_pgd_results_tex(summary_df, wdf, outpath):
    """Fairness-Targeted PGD attack results table."""
    datasets = ['adult', 'credit', 'lsac']
    attacks = ['dp', 'if', 'combined']
    lines = [
        "% AUTO-GENERATED from tau1_summary.csv (tau=1 canonical): do not edit manually",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
    ]
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
                pv = wdf[(wdf['dataset'] == ds) & (wdf['attack'] == atk) & (wdf['alpha'] == alpha)]['dp_pvalue']
                pv_str = f"{pv.values[0]:.3f}" if len(pv) > 0 else "TBD"
                lines.append(f"{ds.capitalize()} & {atk.upper()} & {alpha:.1f} & {dp_n:.4f} & {dp_d:.4f} & {red:.1f}\\% & {pv_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def generate_paper_tabular_results_tex(summary_df, outpath):
    """Paper-style main results table (Acc/DP/IF per dataset x attack x alpha) from tau=1.

    This mirrors the structure previously hand-written in
    paper/auto_generated/tabular_results.tex — but now driven from the
    tau=1 canonical CSV (results/tau1_summary.csv) so it cannot drift to
    tau=100 numbers again.
    """
    datasets = ['adult', 'credit', 'lsac']
    attacks = ['dp', 'if', 'combined']
    lines = [
        "% AUTO-GENERATED by experiments/generate_report_tables.py (tau=1 canonical, was stale hand-edited).",
        "\\begin{tabular}{lllcccccc}",
        "\\toprule",
    ]
    lines.append(
        "Dataset & Attack & $\\alpha$ & Acc (Naive) & Acc (DRO) & DP (Naive) & DP (DRO) & IF (Naive) & IF (DRO) \\\\"
    )
    lines.append("\\midrule")

    for ds in datasets:
        for atk in attacks:
            for alpha in sorted(summary_df['alpha'].unique()):
                sub = summary_df[
                    (summary_df['dataset'] == ds)
                    & (summary_df['attack'] == atk)
                    & (summary_df['alpha'] == alpha)
                ]
                naive = sub[sub['method'] == 'naive']
                dro = sub[sub['method'] == 'dro']
                if len(naive) == 0 or len(dro) == 0:
                    continue
                a_n = naive['acc_mean'].values[0]
                a_d = dro['acc_mean'].values[0]
                d_n = naive['dp_mean'].values[0]
                d_d = dro['dp_mean'].values[0]
                i_n = naive['if_mean'].values[0]
                i_d = dro['if_mean'].values[0]
                lines.append(
                    f"{ds.capitalize()} & {atk.upper()} & {alpha:.1f} & "
                    f"${a_n:.3f}$ & \\textbf{{${a_d:.3f}$}} & "
                    f"${d_n:.3f}$ & \\textbf{{${d_d:.3f}$}} & "
                    f"${i_n:.3f}$ & \\textbf{{${i_d:.3f}$}} \\\\"
                )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def generate_paper_wilcoxon_tex(wdf, outpath):
    """Paper-style Wilcoxon table from tau=1 data, all attacks."""
    lines = [
        "% AUTO-GENERATED by experiments/generate_report_tables.py (tau=1 canonical, was stale hand-edited).",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
    ]
    lines.append("Dataset & Attack & $\\alpha$ & $\\Delta$DP\\% & $p$-value & $\\Delta$IF\\% & $p$-value \\\\")
    lines.append("\\midrule")

    for _, row in wdf.iterrows():
        dp_red = 100 * row['dp_diff_mean'] / (row['dp_naive_mean'] + 1e-8)
        if_red = (
            100 * row['if_diff_mean'] / (row['if_naive_mean'] + 1e-8)
            if row['if_naive_mean'] > 1e-6 else 0.0
        )
        dp_sig = "***" if row['dp_pvalue'] < 0.05 else ""
        if_sig = "***" if row['if_pvalue'] < 0.05 else ""
        lines.append(
            f"{row['dataset'].capitalize()} & {row['attack'].upper()} & {row['alpha']:.1f} & "
            f"{dp_red:.1f}\\% & {row['dp_pvalue']:.3f}{dp_sig} & "
            f"{if_red:.1f}\\% & {row['if_pvalue']:.3f}{if_sig} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def main():
    df = load_tau1_summary()
    print(f"Loaded {len(df)} tau=1 summary rows")

    if len(df) == 0:
        print("No tau=1 results yet — generating placeholder files")
        os.makedirs(OUT_DIR, exist_ok=True)
        for name in ['auto_generated_main_results.tex', 'auto_generated_wilcoxon.tex', 'auto_generated_pgd.tex']:
            path = os.path.join(OUT_DIR, name)
            with open(path, 'w') as f:
                f.write(f"% PLACEHOLDER: run experiments to generate {name}\n")
            print(f"Saved placeholder {path}")
        return

    summary = summarize(df)
    wdf = load_wilcoxon()

    generate_main_results_tex(summary, os.path.join(OUT_DIR, 'auto_generated_main_results.tex'))
    generate_wilcoxon_tex(wdf, os.path.join(OUT_DIR, 'auto_generated_wilcoxon.tex'))
    generate_pgd_results_tex(summary, wdf, os.path.join(OUT_DIR, 'auto_generated_pgd.tex'))

    generate_paper_tabular_results_tex(summary, os.path.join(PAPER_OUT_DIR, 'tabular_results.tex'))
    generate_paper_wilcoxon_tex(wdf, os.path.join(PAPER_OUT_DIR, 'wilcoxon.tex'))

    print("\nDone. Report tables generated in report/sections/ + paper/auto_generated/ (from tau=1 canonical data)")


if __name__ == '__main__':
    main()
