#!/usr/bin/env python3
"""
Auto-generate all Week 2 deliverables from experiment results.
Run this after experiments finish.

Usage:
    python3 scripts/auto_generate_deliverables.py
"""
import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_PATH = 'results/fairness_pgd_results.json'
REPORT_PATH = 'docs/ADVERSARIAL_FAIRNESS_REPORT.md'


def count_results():
    with open(RESULTS_PATH) as f:
        return len(json.load(f))


def run_analysis():
    print("=" * 60)
    print("STEP 1: Running analysis + generating figures")
    print("=" * 60)
    subprocess.run([sys.executable, 'experiments/analyze_fairness_pgd.py'], check=True)
    print("✓ Analysis complete")


def update_report():
    print("\n" + "=" * 60)
    print("STEP 2: Updating report with real numbers")
    print("=" * 60)
    
    import pandas as pd
    from scipy.stats import wilcoxon
    import numpy as np

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Wilcoxon tests
    records = []
    for (dataset, attack, alpha), group in df.groupby(['dataset', 'attack', 'alpha']):
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
        try:
            _, p_dp = wilcoxon(diff_dp, alternative='greater')
        except:
            p_dp = 1.0
        records.append({
            'dataset': dataset,
            'attack': attack,
            'alpha': alpha,
            'n': len(merged),
            'dp_naive': merged['dp_clean_naive'].mean(),
            'dp_dro': merged['dp_clean_dro'].mean(),
            'dp_reduction': 100 * diff_dp.mean() / (merged['dp_clean_naive'].mean() + 1e-8),
            'dp_p': p_dp,
            'dp_wins': (diff_dp > 0).sum(),
        })

    wilcoxon_df = pd.DataFrame(records)
    wilcoxon_df.to_csv('results/fairness_pgd_wilcoxon.csv', index=False)
    print("✓ Wilcoxon table saved")

    # Generate markdown table
    lines = ["## Wilcoxon Test Results (Auto-Generated)", ""]
    lines.append("| Dataset | Attack | α | n | Naive DP | DRO DP | Reduction | p-value | Wins |")
    lines.append("|---------|--------|---|---|----------|--------|-----------|---------|------|")
    for _, row in wilcoxon_df.iterrows():
        sig = "✓" if row['dp_p'] < 0.05 else ""
        lines.append(f"| {row['dataset']} | {row['attack']} | {row['alpha']:.1f} | {int(row['n'])} | "
                     f"{row['dp_naive']:.4f} | {row['dp_dro']:.4f} | "
                     f"{row['dp_reduction']:+.1f}% | {row['dp_p']:.3f} | {int(row['dp_wins'])}/{int(row['n'])} {sig} |")
    lines.append("")

    # Write to report
    with open(REPORT_PATH, 'r') as f:
        content = f.read()

    marker = "## 5. What to Show Madam"
    if marker in content:
        parts = content.split(marker)
        new_content = parts[0] + "\n".join(lines) + "\n\n" + marker + parts[1]
        with open(REPORT_PATH, 'w') as f:
            f.write(new_content)
        print("✓ Report updated")
    else:
        print("⚠ Could not find marker in report, skipping report update")


def git_commit():
    print("\n" + "=" * 60)
    print("STEP 3: Git commit")
    print("=" * 60)
    subprocess.run(['git', 'add', '-A'], check=False)
    result = subprocess.run(['git', 'commit', '-m', 'Week 2: Full Fairness-PGD results + figures + analysis'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Committed:", result.stdout.strip())
    else:
        print("⚠ Git commit issue:", result.stderr.strip())


def main():
    n = count_results()
    print(f"Found {n} experiment results")

    if n < 50:
        print(f"WARNING: Only {n} results — need at least 50 for meaningful analysis")
        print("Run this script again after more experiments complete.")
        return

    run_analysis()
    update_report()
    git_commit()

    print("\n" + "=" * 60)
    print("ALL DELIVERABLES GENERATED")
    print("=" * 60)
    print("Files ready for Madam:")
    print("  - figures/fig8_fairness_pgd_comparison.png")
    print("  - figures/fig9_fairness_pgd_curves.png")
    print("  - results/fairness_pgd_wilcoxon.csv")
    print("  - docs/ADVERSARIAL_FAIRNESS_REPORT.md")
    print("  - results/fairness_pgd_results.json")


if __name__ == '__main__':
    main()
