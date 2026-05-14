"""
Generate final tables, figures, and summary statistics from experiment results.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import shutil
import numpy as np
import matplotlib.pyplot as plt


def load_results():
    """Load all experiment results."""
    results_path = 'results/all_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return []


def generate_table1(results):
    """Generate Table 1: Main results for all datasets and alphas."""
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]

    table_data = []

    for dataset in datasets:
        for alpha in alphas:
            # Filter results for this dataset/alpha
            subset = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]

            if not subset:
                continue

            for method in ['naive', 'dro']:
                for eval_type in ['clean', 'corrupted']:
                    accs = [r[method][eval_type]['accuracy'] for r in subset]
                    dps = [r[method][eval_type]['dp_violation'] for r in subset]
                    ifs = [r[method][eval_type]['if_violation'] for r in subset]

                    n = len(accs)
                    if n == 0:
                        continue

                    table_data.append({
                        'dataset': dataset,
                        'alpha': alpha,
                        'method': method,
                        'eval': eval_type,
                        'acc_mean': np.mean(accs),
                        'acc_std': np.std(accs) / np.sqrt(n),
                        'dp_mean': np.mean(dps),
                        'dp_std': np.std(dps) / np.sqrt(n),
                        'if_mean': np.mean(ifs),
                        'if_std': np.std(ifs) / np.sqrt(n),
                    })

    return table_data


def compute_reductions(results):
    """Compute DRO-FAIR reductions over Naive-FAIR."""
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]

    reductions = []

    for dataset in datasets:
        for alpha in alphas:
            subset = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]

            if not subset:
                continue

            for eval_type in ['clean', 'corrupted']:
                # Naive DP and IF
                naive_dps = [r['naive'][eval_type]['dp_violation'] for r in subset]
                naive_ifs = [r['naive'][eval_type]['if_violation'] for r in subset]

                # DRO DP and IF
                dro_dps = [r['dro'][eval_type]['dp_violation'] for r in subset]
                dro_ifs = [r['dro'][eval_type]['if_violation'] for r in subset]

                naive_dp_mean = np.mean(naive_dps)
                dro_dp_mean = np.mean(dro_dps)
                dp_reduction = (naive_dp_mean - dro_dp_mean) / naive_dp_mean if naive_dp_mean > 0 else 0

                naive_if_mean = np.mean(naive_ifs)
                dro_if_mean = np.mean(dro_ifs)
                if_reduction = (naive_if_mean - dro_if_mean) / naive_if_mean if naive_if_mean > 0 else 0

                reductions.append({
                    'dataset': dataset,
                    'alpha': alpha,
                    'eval': eval_type,
                    'dp_reduction': dp_reduction * 100,
                    'if_reduction': if_reduction * 100,
                })

    return reductions


def generate_summary_stats(results):
    """Generate summary statistics CSV."""
    lines = ["dataset,alpha,method,eval_type,accuracy_mean,accuracy_se,dp_mean,dp_se,if_mean,if_se"]

    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]

    for dataset in datasets:
        for alpha in alphas:
            subset = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
            if not subset:
                continue

            for method in ['naive', 'dro']:
                for eval_type in ['clean', 'corrupted']:
                    accs = [r[method][eval_type]['accuracy'] for r in subset]
                    dps = [r[method][eval_type]['dp_violation'] for r in subset]
                    ifs = [r[method][eval_type]['if_violation'] for r in subset]

                    n = len(accs)
                    lines.append(f"{dataset},{alpha},{method},{eval_type},"
                                f"{np.mean(accs):.4f},{np.std(accs)/np.sqrt(n):.4f},"
                                f"{np.mean(dps):.4f},{np.std(dps)/np.sqrt(n):.4f},"
                                f"{np.mean(ifs):.4f},{np.std(ifs)/np.sqrt(n):.4f}")

    return "\n".join(lines)


def generate_latex_table(results):
    """Generate LaTeX table for main results."""
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Main Results: DRO-FAIR vs Naive-FAIR under Adversarial Corruption}")
    latex.append("\\begin{tabular}{ll|ccc|ccc}")
    latex.append("\\hline")
    latex.append("\\hline")
    latex.append("Dataset & $\\alpha$ & \\multicolumn{3}{c|}{Naive-FAIR} & \\multicolumn{3}{c}{DRO-FAIR} \\\\")
    latex.append("& & Acc & DP & IF & Acc & DP & IF \\\\")
    latex.append("\\hline")

    for dataset in datasets:
        for alpha in alphas:
            subset = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
            if not subset:
                continue

            # Get clean evaluation results
            naive = subset[0]['naive']['clean']
            dro = subset[0]['dro']['clean']

            latex.append(f"{dataset.upper()} & {alpha} & "
                        f"{np.mean([r['naive']['clean']['accuracy'] for r in subset]):.3f} & "
                        f"{np.mean([r['naive']['clean']['dp_violation'] for r in subset]):.3f} & "
                        f"{np.mean([r['naive']['clean']['if_violation'] for r in subset]):.3f} & "
                        f"{np.mean([r['dro']['clean']['accuracy'] for r in subset]):.3f} & "
                        f"{np.mean([r['dro']['clean']['dp_violation'] for r in subset]):.3f} & "
                        f"{np.mean([r['dro']['clean']['if_violation'] for r in subset]):.3f} \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def plot_main_results(results, output_dir='figures'):
    """Generate main results plot."""
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for row, dataset in enumerate(datasets):
        for col, metric in enumerate(['accuracy', 'dp_violation', 'if_violation']):
            ax = axes[row, col]

            naive_means = []
            dro_means = []
            naive_stds = []
            dro_stds = []

            for alpha in alphas:
                subset = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
                if not subset:
                    naive_means.append(0)
                    dro_means.append(0)
                    naive_stds.append(0)
                    dro_stds.append(0)
                else:
                    naive_vals = [r['naive']['clean'][metric] for r in subset]
                    dro_vals = [r['dro']['clean'][metric] for r in subset]

                    naive_means.append(np.mean(naive_vals))
                    dro_means.append(np.mean(dro_vals))
                    naive_stds.append(np.std(naive_vals) / np.sqrt(len(naive_vals)))
                    dro_stds.append(np.std(dro_vals) / np.sqrt(len(dro_vals)))

            x = np.arange(len(alphas))
            width = 0.35

            ax.bar(x - width/2, naive_means, width, yerr=naive_stds, label='Naive-FAIR', alpha=0.7)
            ax.bar(x + width/2, dro_means, width, yerr=dro_stds, label='DRO-FAIR', alpha=0.7)

            ax.set_xlabel('α')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels([str(a) for a in alphas])
            ax.legend()
            ax.set_title(f'{dataset.upper()} - {metric.replace("_", " ").title()}')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'main_results.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'main_results.pdf'))
    print(f"Saved figures/main_results.png")


def plot_test_time_eval(results, output_dir='figures'):
    """Plot clean vs corrupted test evaluation."""
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for row, dataset in enumerate(datasets):
        for col, method in enumerate(['naive', 'dro']):
            ax = axes[row, col]

            clean_dps = []
            corrupt_dps = []

            for alpha in alphas:
                subset = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
                if not subset:
                    continue

                clean_vals = [r[method]['clean']['dp_violation'] for r in subset]
                corrupt_vals = [r[method]['corrupted']['dp_violation'] for r in subset]

                clean_dps.append(np.mean(clean_vals))
                corrupt_dps.append(np.mean(corrupt_vals))

            x = np.arange(len(alphas))
            width = 0.35

            ax.bar(x - width/2, clean_dps, width, label='Clean Test', alpha=0.7)
            ax.bar(x + width/2, corrupt_dps, width, label='Corrupted Test', alpha=0.7)

            ax.set_xlabel('α')
            ax.set_ylabel('DP Violation')
            ax.set_xticks(x)
            ax.set_xticklabels([str(a) for a in alphas])
            ax.legend()
            ax.set_title(f'{dataset.upper()} - {method.upper()}')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'test_time_eval.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'test_time_eval.pdf'))
    print(f"Saved figures/test_time_eval.png")


def main():
    print("Generating results from experiments...")

    results = load_results()

    if not results:
        print("No results found. Please run experiments first.")
        return

    print(f"Loaded {len(results)} experiment results")

    # Check if we have new format (with clean/corrupted evaluation)
    has_test_eval = all('clean' in r.get('naive', {}) for r in results)

    if has_test_eval:
        print("Detected new format with test-time evaluation")
    else:
        print("Warning: Results are in old format (without test-time evaluation)")
        print("Please run run_experiments.py to regenerate results")

    # Generate tables
    table_data = generate_table1(results)
    print(f"Generated Table 1 with {len(table_data)} rows")

    # Save summary stats
    summary = generate_summary_stats(results)
    with open('results/summary_stats.csv', 'w') as f:
        f.write(summary)
    print("Saved results/summary_stats.csv")

    # Save LaTeX table
    latex = generate_latex_table(results)
    with open('results/table1_latex.tex', 'w') as f:
        f.write(latex)
    with open('results/table1.tex', 'w') as f:
        f.write(latex)
    print("Saved results/table1_latex.tex")
    print("Saved results/table1.tex")

    # Save table1_results.csv
    lines = ["dataset,alpha,method,eval_type,acc_mean,acc_std,dp_mean,dp_std,if_mean,if_std"]
    for row in table_data:
        lines.append(f"{row['dataset']},{row['alpha']},{row['method']},{row['eval']},{row['acc_mean']:.4f},"
                    f"{row['acc_std']:.4f},{row['dp_mean']:.4f},{row['dp_std']:.4f},"
                    f"{row['if_mean']:.4f},{row['if_std']:.4f}")

    with open('results/table1_results.csv', 'w') as f:
        f.write("\n".join(lines))
    with open('results/table1.csv', 'w') as f:
        f.write("\n".join(lines))
    print("Saved results/table1_results.csv")
    print("Saved results/table1.csv")

    # Generate figures
    try:
        plot_main_results(results)
        plot_test_time_eval(results)
        os.makedirs('results/figures', exist_ok=True)
        for name in ['main_results.png', 'main_results.pdf',
                     'test_time_eval.png', 'test_time_eval.pdf']:
            src = os.path.join('figures', name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join('results/figures', name))
        print("Saved results/figures/")
    except Exception as e:
        print(f"Warning: Could not generate figures: {e}")

    # Compute reductions
    reductions = compute_reductions(results)
    print("\nDRO-FAIR Reductions (Clean Test Evaluation):")
    for r in reductions[:15]:  # Show first 15
        print(f"  {r['dataset']} α={r['alpha']} ({r['eval']}): "
              f"DP {r['dp_reduction']:.1f}%, IF {r['if_reduction']:.1f}%")

    with open('results/reductions.json', 'w') as f:
        json.dump(reductions, f, indent=2)
    print("Saved results/reductions.json")

    print("\nAll results generated successfully!")


if __name__ == '__main__':
    main()
