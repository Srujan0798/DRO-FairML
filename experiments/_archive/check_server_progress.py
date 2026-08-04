#!/usr/bin/env python3
"""
Check which server experiments have completed and which are pending.
Run this on the server after starting the batch to monitor progress.

Usage:
    venv/bin/python3 experiments/check_server_progress.py
"""
import os
import json
from pathlib import Path

RESULTS = {
    '01_lambda_diag': 'results/lambda_diagnostic.json',
    '02_utkface_lmax': 'results/utkface_lambda_max_cap.json',
    '05_utkface_alpha': 'results/utkface_alpha_sweep.json',
    '06_utkface_fpgd': 'results/utkface_fairness_pgd.json',
    '03_utkface_pixel': 'results/utkface_pixel_pgd.json',
    '04_utkface_randinit': 'results/utkface_randinit.json',
}

LOGS = {
    '01_lambda_diag': 'logs/server_batch_01_lambda_diag.log',
    '02_utkface_lmax': 'logs/server_batch_02_utkface_lmax.log',
    '05_utkface_alpha': 'logs/server_batch_05_utkface_alpha.log',
    '06_utkface_fpgd': 'logs/server_batch_06_utkface_fpgd.log',
    '03_utkface_pixel': 'logs/server_batch_03_pixel_pgd.log',
    '04_utkface_randinit': 'logs/server_batch_04_randinit.log',
}


def count_runs(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path) as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def tail_log(path, n=3):
    if not os.path.exists(path):
        return "[log not found]"
    try:
        with open(path) as f:
            lines = f.readlines()
        return ''.join(lines[-n:]).strip()
    except Exception as e:
        return f"[error reading log: {e}]"


def main():
    print("=" * 60)
    print("SERVER EXPERIMENT PROGRESS")
    print("=" * 60)

    for key, res_path in RESULTS.items():
        runs = count_runs(res_path)
        status = "✅ DONE" if runs > 0 else "⏳ PENDING"
        print(f"\n{key}: {status} ({runs} runs saved)")
        print(f"  Result: {res_path}")
        if runs == 0:
            log = tail_log(LOGS.get(key, ''), n=2)
            print(f"  Log tail: {log[:200]}")

    print("\n" + "=" * 60)
    print("AGGREGATE REPORT")
    print("=" * 60)
    if os.path.exists('results/utkface_all_results.json'):
        print("✅ Aggregate JSON exists")
    else:
        print("⏳ Not yet generated — run: venv/bin/python3 experiments/aggregate_all_results.py")

    if os.path.exists('results/ALL_RESULTS_SUMMARY.md'):
        print("✅ Summary report exists")
    else:
        print("⏳ Not yet generated")

    print("\nFIGURES")
    print("-" * 60)
    for fig in ['fig12_utkface_h3_lmax_cap', 'fig13_utkface_alpha_sweep',
                'fig14_utkface_fairness_pgd', 'fig15_utkface_pixel_vs_feature',
                'fig16_utkface_randinit', 'fig17_summary_dp_vs_alpha']:
        pdf = f'figures/{fig}.pdf'
        png = f'figures/{fig}.png'
        if os.path.exists(pdf) or os.path.exists(png):
            print(f"  ✅ {fig}")
        else:
            print(f"  ⏳ {fig}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
