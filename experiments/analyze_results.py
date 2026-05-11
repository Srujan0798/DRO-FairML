"""
Advanced analysis script for DRO-FAIR results.
Generates additional insights beyond the main results table.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def load_results(results_dir='results'):
    with open(os.path.join(results_dir, 'all_results.json'), 'r') as f:
        return json.load(f)


def analyze_dp_if_tradeoff(results, output_dir='figures'):
    """Analyze the DP-IF tradeoff across methods and corruption levels."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        for method, color, marker in [('naive', 'C0', 'o'), ('dro', 'C1', 's')]:
            dp_vals = []
            if_vals = []
            for alpha in alphas:
                filtered = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
                if filtered:
                    dps = [r[method]['dp_violation'] for r in filtered]
                    ifs = [r[method]['if_violation'] for r in filtered]
                    dp_vals.append(np.mean(dps))
                    if_vals.append(np.mean(ifs))
            
            ax.plot(dp_vals, if_vals, marker=marker, color=color, 
                   label=method.upper(), linewidth=2, markersize=8)
            
            # Annotate alpha values
            for i, alpha in enumerate(alphas[:len(dp_vals)]):
                ax.annotate(f'α={alpha}', (dp_vals[i], if_vals[i]), 
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        ax.set_xlabel('DP Violation')
        ax.set_ylabel('IF Violation')
        ax.set_title(f'{dataset.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dp_if_tradeoff.png'), dpi=300, bbox_inches='tight')
    print(f"DP-IF tradeoff plot saved to {output_dir}/dp_if_tradeoff.png")
    plt.close()


def analyze_robustness_heatmap(results, output_dir='figures'):
    """Create heatmap showing DRO-FAIR improvement over Naive-FAIR."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]
    metrics = ['accuracy', 'dp_violation', 'if_violation']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        heatmap_data = np.zeros((len(datasets), len(alphas)))
        
        for d_idx, dataset in enumerate(datasets):
            for a_idx, alpha in enumerate(alphas):
                filtered = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
                if filtered:
                    naive_vals = [r['naive'][metric] for r in filtered]
                    dro_vals = [r['dro'][metric] for r in filtered]
                    
                    if metric == 'accuracy':
                        # For accuracy, positive = DRO is better
                        improvement = (np.mean(dro_vals) - np.mean(naive_vals)) * 100
                    else:
                        # For violations, negative = DRO is better (reduction)
                        improvement = (np.mean(naive_vals) - np.mean(dro_vals)) / (np.mean(naive_vals) + 1e-8) * 100
                    
                    heatmap_data[d_idx, a_idx] = improvement
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   xticklabels=[f'α={a}' for a in alphas],
                   yticklabels=[d.upper() for d in datasets],
                   ax=ax, cbar_kws={'label': 'Improvement (%)'})
        
        metric_name = 'Accuracy Gain' if metric == 'accuracy' else f'{metric.upper()} Reduction'
        ax.set_title(f'DRO-FAIR {metric_name} vs Naive-FAIR')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"Robustness heatmap saved to {output_dir}/robustness_heatmap.png")
    plt.close()


def analyze_seed_stability(results, output_dir='figures'):
    """Analyze stability across random seeds."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.1, 0.2, 0.3]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for d_idx, dataset in enumerate(datasets):
        for a_idx, alpha in enumerate(alphas):
            ax = axes[d_idx, a_idx]
            
            filtered = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
            if not filtered:
                continue
            
            seeds = [r['seed'] for r in filtered]
            naive_acc = [r['naive']['accuracy'] for r in filtered]
            dro_acc = [r['dro']['accuracy'] for r in filtered]
            naive_dp = [r['naive']['dp_violation'] for r in filtered]
            dro_dp = [r['dro']['dp_violation'] for r in filtered]
            
            ax.plot(seeds, naive_acc, 'o-', label='Naive Acc', color='C0')
            ax.plot(seeds, dro_acc, 's-', label='DRO Acc', color='C1')
            ax_twin = ax.twinx()
            ax_twin.plot(seeds, naive_dp, 'o--', label='Naive DP', color='C2')
            ax_twin.plot(seeds, dro_dp, 's--', label='DRO DP', color='C3')
            
            ax.set_xlabel('Seed')
            ax.set_ylabel('Accuracy', color='C0')
            ax_twin.set_ylabel('DP Violation', color='C2')
            ax.set_title(f'{dataset.upper()} α={alpha}')
            ax.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = axes[0, 0].get_legend_handles_labels()
    lines2, labels2 = axes[0, 0].twinx().get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
              bbox_to_anchor=(0.5, 0.98), ncol=4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'seed_stability.png'), dpi=300, bbox_inches='tight')
    print(f"Seed stability plot saved to {output_dir}/seed_stability.png")
    plt.close()


def generate_latex_table(results, output_dir='results'):
    """Generate LaTeX table for paper submission."""
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Quantitative Results: Mean $\\pm$ SE over 10 seeds}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\begin{tabular}{ll|ccc|ccc|ccc}")
    latex.append("\\toprule")
    latex.append("& & \\multicolumn{3}{c|}{Adult} & \\multicolumn{3}{c|}{Credit} & \\multicolumn{3}{c}{LSAC} \\\\")
    latex.append("$\\alpha$ & Method & Acc$\\uparrow$ & DP$\\downarrow$ & IF$\\downarrow$ & Acc$\\uparrow$ & DP$\\downarrow$ & IF$\\downarrow$ & Acc$\\uparrow$ & DP$\\downarrow$ & IF$\\downarrow$ \\\\")
    latex.append("\\midrule")
    
    for alpha in alphas:
        for method_name, method_key in [('Naive', 'naive'), ('DRO-FAIR', 'dro')]:
            row = [f"{alpha:.1f}" if method_name == 'Naive' else "", method_name]
            
            for dataset in datasets:
                filtered = [r for r in results if r['dataset'] == dataset and r['alpha'] == alpha]
                if filtered:
                    accs = [r[method_key]['accuracy'] for r in filtered]
                    dps = [r[method_key]['dp_violation'] for r in filtered]
                    ifs = [r[method_key]['if_violation'] for r in filtered]
                    
                    acc_str = f"{np.mean(accs):.3f} $\\pm$ {np.std(accs)/np.sqrt(len(accs)):.3f}"
                    dp_str = f"{np.mean(dps):.3f} $\\pm$ {np.std(dps)/np.sqrt(len(dps)):.3f}"
                    if_str = f"{np.mean(ifs):.3f} $\\pm$ {np.std(ifs)/np.sqrt(len(ifs)):.3f}"
                    row.extend([acc_str, dp_str, if_str])
                else:
                    row.extend(['-', '-', '-'])
            
            latex.append(" & ".join(row) + " \\\\")
        
        if alpha != alphas[-1]:
            latex.append("\\midrule")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    output_path = os.path.join(output_dir, 'table1_latex.tex')
    with open(output_path, 'w') as f:
        f.write("\n".join(latex))
    
    print(f"LaTeX table saved to {output_path}")


if __name__ == '__main__':
    results = load_results('results')
    
    analyze_dp_if_tradeoff(results, 'figures')
    analyze_robustness_heatmap(results, 'figures')
    analyze_seed_stability(results, 'figures')
    generate_latex_table(results, 'results')
