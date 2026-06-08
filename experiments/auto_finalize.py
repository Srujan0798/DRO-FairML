#!/usr/bin/env python3
"""
Auto-finalize: run analysis + generate figures + summary report
after all experiments complete. Commits and pushes results.

Usage:
    venv/bin/python3 experiments/auto_finalize.py
"""
import os
import sys
import json
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

EXPECTED = {
    'fairness_pgd_results.json': 270,
    'lambda_diagnostic_full.json': 12,
}


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD:  {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"WARNING: Command failed with code {result.returncode}")
    return result.returncode == 0


def check_results():
    print("Checking result files...")
    ready = {}
    for fname, expected_count in EXPECTED.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {fname}")
            ready[fname] = False
            continue
        with open(path) as f:
            data = json.load(f)
        count = len(data)
        status = "✅" if count >= expected_count else "⏳"
        print(f"  {status} {fname}: {count}/{expected_count}")
        ready[fname] = count >= expected_count
    return all(ready.values())


def generate_summary():
    """Generate a markdown summary of key findings."""
    lines = ["# Experiment Summary\n", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]

    # Load tabular results
    tabular_path = os.path.join(RESULTS_DIR, 'fairness_pgd_results.json')
    if os.path.exists(tabular_path):
        with open(tabular_path) as f:
            tabular = json.load(f)
        lines.append(f"\n## Tabular Results ({len(tabular)} runs)\n")
        import pandas as pd
        df = pd.DataFrame(tabular)
        summary = df.groupby(['dataset', 'attack', 'alpha', 'method']).agg(
            acc_mean=('acc_clean', 'mean'),
            dp_mean=('dp_clean', 'mean'),
            if_mean=('if_clean', 'mean'),
            n=('seed', 'count')
        ).reset_index()
        lines.append(summary.to_markdown(index=False))

    # Load lambda diagnostic
    lambda_path = os.path.join(RESULTS_DIR, 'lambda_diagnostic_full.json')
    if os.path.exists(lambda_path):
        with open(lambda_path) as f:
            lam = json.load(f)
        lines.append(f"\n## Lambda Diagnostic ({len(lam)} runs)\n")
        for r in lam:
            lines.append(f"- {r.get('dataset')} λ_max={r.get('lambda_max')} seed={r.get('seed')}: "
                        f"final λ_DP={r.get('lambda_dp_final', 'N/A'):.4f}, "
                        f"final λ_IF={r.get('lambda_if_final', 'N/A'):.4f}")

    summary_path = 'RESULTS_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved summary to {summary_path}")


def git_commit_push():
    """Commit all new results and push."""
    print("\nGit commit + push...")
    cmds = [
        "git add -A",
        "git diff --cached --quiet || git commit -m 'auto: finalize experimental results + figures'",
        "git push origin main"
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True)
    print("Pushed to origin/main")


def main():
    print("="*60)
    print("AUTO-FINALIZE PIPELINE")
    print("="*60)

    if not check_results():
        print("\n❌ Not all results are ready. Exiting.")
        print("Run this again after experiments complete.")
        return

    # Analysis
    run_cmd("venv/bin/python3 experiments/analyze_fairness_pgd.py",
            "Analyze tabular Fairness-PGD results")

    # Figures
    run_cmd("venv/bin/python3 experiments/generate_all_figures.py",
            "Generate all publication figures")

    # Paper tables
    run_cmd("venv/bin/python3 experiments/generate_paper_tables.py",
            "Generate paper LaTeX tables")

    # Report tables
    run_cmd("venv/bin/python3 experiments/generate_report_tables.py",
            "Generate report LaTeX tables")

    # Summary
    generate_summary()

    # Git
    git_commit_push()

    print("\n" + "="*60)
    print("✅ AUTO-FINALIZE COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
