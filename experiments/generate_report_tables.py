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
from collections import defaultdict
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CANONICAL_PATH = 'results/canonical_tau1.json'
OUT_DIR = 'report/sections'
PAPER_OUT_DIR = 'paper/auto_generated'


def load_canonical_rows():
    """Load the canonical 540-row grid (results/canonical_tau1.json)."""
    from experiments.loaders import load_canonical_tau1
    return load_canonical_tau1()


IF_DEGENERATE_TOL = 1e-6


def _format_if_cell(row):
    """Render the two IF columns (effect %, p-value), or mark them degenerate.

    Under the DP and COMBINED attacks the canonical rows carry an IF violation
    of ~1e-11 -- floating-point noise, not a measured quantity. It is an
    artifact of the pre-fix Euclidean k-NN training graph (corrected to cosine
    in 04d00a6, after the canonical grid was run). Running a Wilcoxon test on
    noise produced p-values that printed as significant (*** at p=0.016 for
    adult/combined alpha=0.1 and 0.3) beside a 0.0%% effect size -- an
    indefensible pairing. Such cells are now printed as "--" rather than
    given a fake number and a significance star.
    """
    if row['if_naive_mean'] <= IF_DEGENERATE_TOL:
        return "-- & --"
    if_red = 100 * row['if_diff_mean'] / (row['if_naive_mean'] + 1e-8)
    if_sig = "***" if row['if_pvalue'] < 0.05 else ""
    return f"{if_red:.1f}\\% & {row['if_pvalue']:.3f}{if_sig}"


def build_summary(rows):
    """Per (dataset, alpha, attack, method) mean/sem of acc/dp/if."""
    recs = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['alpha'], r['attack'], r['method'])].append(r)
    for (ds, alpha, atk, method), rs in groups.items():
        acc = np.array([x['acc_clean'] for x in rs])
        dp = np.array([x['dp_clean'] for x in rs])
        ifv = np.array([x['if_clean'] for x in rs])
        n = len(acc)
        se = lambda a: a.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        recs.append({
            'dataset': ds, 'alpha': alpha, 'attack': atk, 'method': method,
            'acc_mean': acc.mean(), 'acc_se': se(acc),
            'dp_mean': dp.mean(), 'dp_se': se(dp),
            'if_mean': ifv.mean(), 'if_se': se(ifv),
        })
    return pd.DataFrame(recs)


def build_wilcoxon(rows):
    """One-sided Wilcoxon (DRO better = naive DP - dro DP > 0) per (dataset,alpha,attack).

    Critical: pair by *seed* before subtracting. JSON row order is not guaranteed
    method-interleaved; unpaired lists yield wrong p-values (e.g. Adult/IF α=0.2
    mis-reported 0.344 instead of 0.0156).
    """
    recs = []
    groups = defaultdict(lambda: {'naive': {}, 'dro': {}})
    for r in rows:
        groups[(r['dataset'], r['alpha'], r['attack'])][r['method']][r['seed']] = r
    for (ds, alpha, atk), md in groups.items():
        seeds = sorted(set(md['naive']) & set(md['dro']))
        nv = np.array([md['naive'][s]['dp_clean'] for s in seeds])
        dv = np.array([md['dro'][s]['dp_clean'] for s in seeds])
        diff = nv - dv
        p = 1.0
        if len(diff) >= 2 and np.std(diff) > 0:
            _, p = wilcoxon(diff, alternative='greater')
        if_nv = np.array([md['naive'][s]['if_clean'] for s in seeds])
        if_dv = np.array([md['dro'][s]['if_clean'] for s in seeds])
        if_diff = if_nv - if_dv
        if_p = 1.0
        if len(if_diff) >= 2 and np.std(if_diff) > 0:
            _, if_p = wilcoxon(if_diff, alternative='greater')
        recs.append({
            'dataset': ds, 'alpha': alpha, 'attack': atk,
            'n_pairs': len(seeds),
            'dp_naive_mean': float(nv.mean()) if len(nv) else float('nan'),
            'dp_diff_mean': float(diff.mean()) if len(diff) else float('nan'),
            'dp_pvalue': float(p),
            'if_naive_mean': float(if_nv.mean()) if len(if_nv) else float('nan'),
            'if_diff_mean': float(if_diff.mean()) if len(if_diff) else float('nan'),
            'if_pvalue': float(if_p),
        })
    return pd.DataFrame(recs)


def generate_main_results_tex(summary_df, outpath):
    """Report-style main results table with Acc/DP/IF per dataset and alpha (DP attack).

    Driven from results/canonical_tau1.json means (not a stale CSV).
    Bold only the better method (higher acc / lower fairness violation).
    """
    datasets = ['adult', 'credit', 'lsac']
    lines = ["% AUTO-GENERATED from results/canonical_tau1.json (tau=1): do not edit manually",
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
            an, ad = naive['acc_mean'].values[0], dro['acc_mean'].values[0]
            dn, dd = naive['dp_mean'].values[0], dro['dp_mean'].values[0]
            inn, idd = naive['if_mean'].values[0], dro['if_mean'].values[0]
            acc_n = f"{an:.3f} $\\pm$ {naive['acc_se'].values[0]:.3f}"
            acc_d = f"{ad:.3f} $\\pm$ {dro['acc_se'].values[0]:.3f}"
            if ad > an:
                acc_d = f"\\textbf{{{acc_d}}}"
            elif an > ad:
                acc_n = f"\\textbf{{{acc_n}}}"
            dp_n = f"{dn:.4f} $\\pm$ {naive['dp_se'].values[0]:.4f}"
            dp_d = f"{dd:.4f} $\\pm$ {dro['dp_se'].values[0]:.4f}"
            if dd < dn:
                dp_d = f"\\textbf{{{dp_d}}}"
            elif dn < dd:
                dp_n = f"\\textbf{{{dp_n}}}"
            if_n = f"{inn:.4f} $\\pm$ {naive['if_se'].values[0]:.4f}"
            if_d = f"{idd:.4f} $\\pm$ {dro['if_se'].values[0]:.4f}"
            if idd < inn:
                if_d = f"\\textbf{{{if_d}}}"
            elif inn < idd:
                if_n = f"\\textbf{{{if_n}}}"
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
        dp_sig = "***" if row['dp_pvalue'] < 0.05 else ""
        if_cell = _format_if_cell(row)
        lines.append(
            f"{row['dataset'].capitalize()} & {row['alpha']:.1f} & "
            f"{dp_red:.1f}\\% & {row['dp_pvalue']:.3f}{dp_sig} & "
            f"{if_cell} \\\\"
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

                def _fmt(val, bold):
                    s = f"${val:.3f}$"
                    return f"\\textbf{{{s}}}" if bold else s

                # Bold the better method: higher acc, lower DP/IF
                lines.append(
                    f"{ds.capitalize()} & {atk.upper()} & {alpha:.1f} & "
                    f"{_fmt(a_n, a_n >= a_d)} & {_fmt(a_d, a_d > a_n)} & "
                    f"{_fmt(d_n, d_n <= d_d)} & {_fmt(d_d, d_d < d_n)} & "
                    f"{_fmt(i_n, i_n <= i_d)} & {_fmt(i_d, i_d < i_n)} \\\\"
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
        dp_sig = "***" if row['dp_pvalue'] < 0.05 else ""
        if_cell = _format_if_cell(row)
        lines.append(
            f"{row['dataset'].capitalize()} & {row['attack'].upper()} & {row['alpha']:.1f} & "
            f"{dp_red:.1f}\\% & {row['dp_pvalue']:.3f}{dp_sig} & "
            f"{if_cell} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved {outpath}")


def main():
    rows = load_canonical_rows()
    print(f"Loaded {len(rows)} canonical rows from {CANONICAL_PATH}")

    if len(rows) == 0:
        print("No canonical results yet — generating placeholder files")
        os.makedirs(OUT_DIR, exist_ok=True)
        for name in ['auto_generated_main_results.tex', 'auto_generated_wilcoxon.tex', 'auto_generated_pgd.tex']:
            path = os.path.join(OUT_DIR, name)
            with open(path, 'w') as f:
                f.write(f"% PLACEHOLDER: run experiments to generate {name}\n")
            print(f"Saved placeholder {path}")
        return

    summary = build_summary(rows)
    wdf = build_wilcoxon(rows)

    generate_main_results_tex(summary, os.path.join(OUT_DIR, 'auto_generated_main_results.tex'))
    generate_wilcoxon_tex(wdf, os.path.join(OUT_DIR, 'auto_generated_wilcoxon.tex'))
    generate_pgd_results_tex(summary, wdf, os.path.join(OUT_DIR, 'auto_generated_pgd.tex'))

    generate_paper_tabular_results_tex(summary, os.path.join(PAPER_OUT_DIR, 'tabular_results.tex'))
    generate_paper_wilcoxon_tex(wdf, os.path.join(PAPER_OUT_DIR, 'wilcoxon.tex'))

    print("\nDone. Report tables generated in report/sections/ + paper/auto_generated/ (from results/canonical_tau1.json)")


if __name__ == '__main__':
    main()
