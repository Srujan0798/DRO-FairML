#!/usr/bin/env python3
"""
Auto-generate LaTeX table fragments for paper and report.
Reads results/fairness_pgd_results.json and produces:
  - paper/auto_generated/tabular_results.tex
  - paper/auto_generated/wilcoxon.tex
  - paper/auto_generated/key_findings.tex
  - report/sections/auto_generated_main_results.tex

Usage:
    venv/bin/python3 experiments/generate_paper_tables.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_PATH = 'results/fairness_pgd_results.json'
OUT_DIR_PAPER = 'paper/auto_generated'
OUT_DIR_REPORT = 'report/sections'


def load_results(path=RESULTS_PATH):
    from experiments.loaders import load_fairness_pgd_results
    return load_fairness_pgd_results()


def summarize(df):
    grouped = df.groupby(['dataset', 'attack', 'alpha', 'method'])
    rows = []
    for (ds, atk, alpha, method), group in grouped:
        rows.append({
            'dataset': ds,
            'attack': atk,
            'alpha': alpha,
            'method': method,
            'n': len(group),
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
            'dataset': ds,
            'attack': atk,
            'alpha': alpha,
            'n': len(merged),
            'dp_naive_mean': merged['dp_clean_naive'].mean(),
            'dp_dro_mean': merged['dp_clean_dro'].mean(),
            'dp_reduction_pct': 100 * (merged['dp_clean_naive'].mean() - merged['dp_clean_dro'].mean()) / (merged['dp_clean_naive'].mean() + 1e-8),
            'dp_pvalue': p_dp,
            'if_naive_mean': merged['if_clean_naive'].mean(),
            'if_dro_mean': merged['if_clean_dro'].mean(),
            'if_reduction_pct': 100 * (merged['if_clean_naive'].mean() - merged['if_clean_dro'].mean()) / (merged['if_clean_naive'].mean() + 1e-8),
            'if_pvalue': p_if,
        })
    return pd.DataFrame(records)


def fmt(mean, se=None, bold=False):
    if se is not None:
        s = f"${mean:.3f} \\pm {se:.3f}$"
    else:
        s = f"{mean:.3f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def generate_tabular_results_tex(summary_df, outpath):
    """Main results table: Naive vs DRO per dataset/attack/alpha."""
    datasets = ['adult', 'credit', 'lsac']
    attacks = ['dp', 'if', 'combined']
    alphas = sorted(summary_df['alpha'].unique())

    lines = ["% AUTO-GENERATED: do not edit manually", "\\begin{tabular}{lllcccccc}", "\\toprule"]
    lines.append("Dataset & Attack & $\\alpha$ & Acc (Naive) & Acc (DRO) & DP (Naive) & DP (DRO) & IF (Naive) & IF (DRO) \\\\")
    lines.append("\\midrule")

    for ds in datasets:
        for atk in attacks:
            for alpha in alphas:
                sub = summary_df[(summary_df['dataset'] == ds) & (summary_df['attack'] == atk) & (summary_df['alpha'] == alpha)]
                naive = sub[sub['method'] == 'naive']
                dro = sub[sub['method'] == 'dro']
                if len(naive) == 0 or len(dro) == 0:
                    continue
                acc_n = fmt(naive['acc_mean'].values[0], naive['acc_se'].values[0])
                acc_d = fmt(dro['acc_mean'].values[0], dro['acc_se'].values[0])
                dp_n = fmt(naive['dp_mean'].values[0], naive['dp_se'].values[0])
                dp_d = fmt(dro['dp_mean'].values[0], dro['dp_se'].values[0], bold=True)
                if_n = fmt(naive['if_mean'].values[0], naive['if_se'].values[0])
                if_d = fmt(dro['if_mean'].values[0], dro['if_se'].values[0], bold=True)
                lines.append(f"{ds.capitalize()} & {atk.upper()} & {alpha:.1f} & {acc_n} & {acc_d} & {dp_n} & {dp_d} & {if_n} & {if_d} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def generate_wilcoxon_tex(wdf, outpath):
    """Wilcoxon signed-rank test table."""
    lines = ["% AUTO-GENERATED: do not edit manually", "\\begin{tabular}{lllrrrr}", "\\toprule"]
    lines.append("Dataset & Attack & $\\alpha$ & $\\Delta$DP\\% & $p$-value & $\\Delta$IF\\% & $p$-value \\\\")
    lines.append("\\midrule")

    for _, row in wdf.iterrows():
        dp_sig = "***" if row['dp_pvalue'] < 0.05 else ""
        if_sig = "***" if row['if_pvalue'] < 0.05 else ""
        lines.append(
            f"{row['dataset'].capitalize()} & {row['attack'].upper()} & {row['alpha']:.1f} & "
            f"{row['dp_reduction_pct']:.1f}\\% & {row['dp_pvalue']:.3f}{dp_sig} & "
            f"{row['if_reduction_pct']:.1f}\\% & {row['if_pvalue']:.3f}{if_sig} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def generate_key_findings_tex(summary_df, wdf, outpath):
    """LaTeX macros for key findings (abstract, conclusion, etc.)."""
    if len(wdf) == 0:
        lines = ["% AUTO-GENERATED: no data available yet"]
    else:
        dp_reds = wdf['dp_reduction_pct'].values
        dp_reds_valid = dp_reds[~np.isnan(dp_reds)]
        min_red = dp_reds_valid.min() if len(dp_reds_valid) > 0 else 0
        max_red = dp_reds_valid.max() if len(dp_reds_valid) > 0 else 0
        n_sig = (wdf['dp_pvalue'] < 0.05).sum()
        n_total = len(wdf)

        lines = [
            "% AUTO-GENERATED: do not edit manually",
            f"\\newcommand{{\\tabularReductionRange}}{{{min_red:.0f}--{max_red:.0f}\\%}}",
            f"\\newcommand{{\\tabularSigTests}}{{{n_sig}/{n_total}}}",
            f"\\newcommand{{\\tabularNExperiments}}{{{len(summary_df) * 2 if len(summary_df) > 0 else 0}}}",
        ]

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def main():
    results = load_results()
    print(f"Loaded {len(results)} results")

    if len(results) == 0:
        print("No results yet — generating placeholder files")
        os.makedirs(OUT_DIR_PAPER, exist_ok=True)
        os.makedirs(OUT_DIR_REPORT, exist_ok=True)
        for p in [
            os.path.join(OUT_DIR_PAPER, 'tabular_results.tex'),
            os.path.join(OUT_DIR_PAPER, 'wilcoxon.tex'),
            os.path.join(OUT_DIR_PAPER, 'key_findings.tex'),
        ]:
            with open(p, 'w') as f:
                f.write(f"% PLACEHOLDER: run experiments to generate {os.path.basename(p)}\\n")
            print(f"Saved placeholder {p}")
        return

    df = pd.DataFrame(results)
    summary = summarize(df)
    wdf = wilcoxon_table(df)

    generate_tabular_results_tex(summary, os.path.join(OUT_DIR_PAPER, 'tabular_results.tex'))
    generate_wilcoxon_tex(wdf, os.path.join(OUT_DIR_PAPER, 'wilcoxon.tex'))
    generate_key_findings_tex(summary, wdf, os.path.join(OUT_DIR_PAPER, 'key_findings.tex'))

    print("\nDone. Paper tables generated in paper/auto_generated/")


if __name__ == '__main__':
    main()
