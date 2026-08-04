#!/usr/bin/env python3
"""
Generate extra LaTeX tables (runtime, ablation) for the report.
Run after experiments + ablations complete.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def generate_runtime_table(output_path='results/runtime_table.tex'):
    """Generate LaTeX runtime table from runtimes.json or all_results.json."""
    if os.path.exists('results/runtimes.json'):
        data = json.load(open('results/runtimes.json'))
        naive_mean = data['naive_mean']
        naive_std = data['naive_std']
        dro_mean = data['dro_mean']
        dro_std = data['dro_std']
        overhead = data['overhead']
    elif os.path.exists('results/all_results.json'):
        results = json.load(open('results/all_results.json'))
        naive_times = [r['naive']['time'] for r in results if 'time' in r.get('naive', {})]
        dro_times = [r['dro']['time'] for r in results if 'time' in r.get('dro', {})]
        naive_mean = np.mean(naive_times) if naive_times else 0
        naive_std = np.std(naive_times) if naive_times else 0
        dro_mean = np.mean(dro_times) if dro_times else 0
        dro_std = np.std(dro_times) if dro_times else 0
        overhead = dro_mean / naive_mean if naive_mean > 0 else 0
    else:
        print("No runtime data found.")
        return

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Runtime comparison (seconds per experiment, CPU).}")
    latex.append("\\label{tab:runtime}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("Method & Time (s) & Overhead \\\\")
    latex.append("\\midrule")
    latex.append(f"Naive-FAIR & ${naive_mean:.1f} \\pm {naive_std:.1f}$ & $1.0\\times$ \\\\")
    latex.append(f"DRO-FAIR & ${dro_mean:.1f} \\pm {dro_std:.1f}$ & ${overhead:.1f}\\times$ \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write("\n".join(latex))
    print(f"Saved {output_path}")


def generate_ablation_table(output_path='results/ablation_table.tex'):
    """Generate LaTeX ablation table from ablation results."""
    ablation_path = 'results/ablation_full.json'
    if not os.path.exists(ablation_path):
        print(f"No ablation data found at {ablation_path}.")
        return

    data = json.load(open(ablation_path))

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation study: effect of radius calibration and constraint type.}")
    latex.append("\\label{tab:ablation}")
    latex.append("\\begin{tabular}{ll|ccc}")
    latex.append("\\toprule")
    latex.append("Variant & Dataset & Acc & DP & IF \\\\")
    latex.append("\\midrule")

    for entry in data:
        name = entry.get('method', entry.get('variant', ''))
        dataset = entry.get('dataset', '')
        acc = entry.get('accuracy', entry.get('acc_mean', 0))
        dp = entry.get('dp_violation', entry.get('dp_mean', 0))
        if_v = entry.get('if_violation', entry.get('if_mean', 0))
        latex.append(f"{name} & {dataset} & {acc:.3f} & {dp:.3f} & {if_v:.3f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write("\n".join(latex))
    print(f"Saved {output_path}")


if __name__ == '__main__':
    generate_runtime_table()
    generate_ablation_table()
