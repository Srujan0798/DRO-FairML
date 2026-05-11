"""
Generate result tables and plots from experiment outputs.
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
    """Load experiment results."""
    with open(os.path.join(results_dir, 'all_results.json'), 'r') as f:
        results = json.load(f)
    return results


def generate_table1(results, output_dir='results'):
    """Generate Table 1: Quantitative Results (Mean ± SE over seeds)."""
    
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]
    methods = ['naive', 'dro']
    
    rows = []
    for dataset in datasets:
        for alpha in alphas:
            # Filter results
            filtered = [r for r in results 
                       if r['dataset'] == dataset and r['alpha'] == alpha]
            
            if not filtered:
                continue
            
            for method in methods:
                accs = [r[method]['accuracy'] for r in filtered]
                dps = [r[method]['dp_violation'] for r in filtered]
                ifs = [r[method]['if_violation'] for r in filtered]
                
                rows.append({
                    'Dataset': dataset.upper(),
                    'Alpha': alpha,
                    'Method': 'Naive' if method == 'naive' else 'DRO-FAIR',
                    'Acc': f"{np.mean(accs):.3f}±{np.std(accs)/np.sqrt(len(accs)):.3f}",
                    'DP': f"{np.mean(dps):.3f}±{np.std(dps)/np.sqrt(len(dps)):.3f}",
                    'IF': f"{np.mean(ifs):.3f}±{np.std(ifs)/np.sqrt(len(ifs)):.3f}",
                    'Acc_mean': np.mean(accs),
                    'DP_mean': np.mean(dps),
                    'IF_mean': np.mean(ifs),
                })
    
    df = pd.DataFrame(rows)
    
    # Print formatted table
    print("\n" + "="*80)
    print("TABLE 1: Quantitative Results (Mean ± SE over 10 seeds)")
    print("="*80)
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}")
        print("-"*80)
        print(f"{'α':>4} {'Method':>12} {'Acc↑':>14} {'DP↓':>14} {'IF↓':>14}")
        print("-"*80)
        
        for alpha in alphas:
            for method in ['Naive', 'DRO-FAIR']:
                row = df[(df['Dataset'] == dataset.upper()) & 
                        (df['Alpha'] == alpha) & 
                        (df['Method'] == method)]
                if not row.empty:
                    r = row.iloc[0]
                    print(f"{alpha:>4.1f} {method:>12} {r['Acc']:>14} {r['DP']:>14} {r['IF']:>14}")
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'table1_results.csv'), index=False)
    print(f"\nTable saved to {output_dir}/table1_results.csv")
    
    return df


def generate_plots(results, output_dir='figures'):
    """Generate plots for accuracy, DP, and IF across corruption levels."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['adult', 'credit', 'lsac']
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, dataset in enumerate(datasets):
        # Aggregate data
        acc_data = {'naive': [], 'dro': []}
        dp_data = {'naive': [], 'dro': []}
        if_data = {'naive': [], 'dro': []}
        
        for alpha in alphas:
            filtered = [r for r in results 
                       if r['dataset'] == dataset and r['alpha'] == alpha]
            
            for method in ['naive', 'dro']:
                accs = [r[method]['accuracy'] for r in filtered]
                dps = [r[method]['dp_violation'] for r in filtered]
                ifs = [r[method]['if_violation'] for r in filtered]
                
                acc_data[method].append((np.mean(accs), np.std(accs)/np.sqrt(len(accs))))
                dp_data[method].append((np.mean(dps), np.std(dps)/np.sqrt(len(dps))))
                if_data[method].append((np.mean(ifs), np.std(ifs)/np.sqrt(len(ifs))))
        
        # Accuracy
        ax = axes[i, 0]
        for method, color in [('naive', 'C0'), ('dro', 'C1')]:
            means = [x[0] for x in acc_data[method]]
            ses = [x[1] for x in acc_data[method]]
            ax.errorbar(alphas, means, yerr=ses, label=method.upper(), marker='o', color=color)
        ax.set_xlabel('Corruption Rate (α)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset.upper()}: Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # DP Violation
        ax = axes[i, 1]
        for method, color in [('naive', 'C0'), ('dro', 'C1')]:
            means = [x[0] for x in dp_data[method]]
            ses = [x[1] for x in dp_data[method]]
            ax.errorbar(alphas, means, yerr=ses, label=method.upper(), marker='s', color=color)
        ax.set_xlabel('Corruption Rate (α)')
        ax.set_ylabel('DP Violation')
        ax.set_title(f'{dataset.upper()}: DP Violation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # IF Violation
        ax = axes[i, 2]
        for method, color in [('naive', 'C0'), ('dro', 'C1')]:
            means = [x[0] for x in if_data[method]]
            ses = [x[1] for x in if_data[method]]
            ax.errorbar(alphas, means, yerr=ses, label=method.upper(), marker='^', color=color)
        ax.set_xlabel('Corruption Rate (α)')
        ax.set_ylabel('IF Violation')
        ax.set_title(f'{dataset.upper()}: IF Violation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'main_results.png'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_dir}/main_results.png")
    plt.close()


def generate_summary_stats(results, output_dir='results'):
    """Generate summary statistics comparing DRO-FAIR vs Naive-FAIR."""
    
    summary = []
    
    for dataset in ['adult', 'credit', 'lsac']:
        for alpha in [0.1, 0.2, 0.3]:
            filtered = [r for r in results 
                       if r['dataset'] == dataset and r['alpha'] == alpha]
            
            if not filtered:
                continue
            
            naive_dps = [r['naive']['dp_violation'] for r in filtered]
            dro_dps = [r['dro']['dp_violation'] for r in filtered]
            naive_ifs = [r['naive']['if_violation'] for r in filtered]
            dro_ifs = [r['dro']['if_violation'] for r in filtered]
            naive_accs = [r['naive']['accuracy'] for r in filtered]
            dro_accs = [r['dro']['accuracy'] for r in filtered]
            
            dp_reduction = (np.mean(naive_dps) - np.mean(dro_dps)) / (np.mean(naive_dps) + 1e-8) * 100
            if_reduction = (np.mean(naive_ifs) - np.mean(dro_ifs)) / (np.mean(naive_ifs) + 1e-8) * 100
            acc_diff = (np.mean(dro_accs) - np.mean(naive_accs)) * 100
            
            summary.append({
                'Dataset': dataset.upper(),
                'Alpha': alpha,
                'DP_Reduction_%': dp_reduction,
                'IF_Reduction_%': if_reduction,
                'Accuracy_Diff_%': acc_diff,
                'Naive_DP': np.mean(naive_dps),
                'DRO_DP': np.mean(dro_dps),
                'Naive_IF': np.mean(naive_ifs),
                'DRO_IF': np.mean(dro_ifs),
            })
    
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(output_dir, 'summary_stats.csv'), index=False)
    
    print("\n" + "="*80)
    print("SUMMARY: DRO-FAIR Improvements over Naive-FAIR")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


if __name__ == '__main__':
    results = load_results('results')
    
    generate_table1(results, 'results')
    generate_plots(results, 'figures')
    generate_summary_stats(results, 'results')
