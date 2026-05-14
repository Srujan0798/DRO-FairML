#!/usr/bin/env python3
"""
DRO-FAIR: Distributionally Robust Optimization for Fairness
Main entry point for running experiments and generating results.

Usage:
    python main.py --run-experiments
    python main.py --generate-results
    python main.py --full-pipeline
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='DRO-FAIR Experiment Runner')
    parser.add_argument('--run-experiments', action='store_true',
                        help='Run all experiments (Naive-FAIR and DRO-FAIR)')
    parser.add_argument('--generate-results', action='store_true',
                        help='Generate result tables and plots')
    parser.add_argument('--full-pipeline', action='store_true',
                        help='Run experiments then generate results')
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'],
                        help='Datasets to evaluate')
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4],
                        help='Corruption rates to evaluate')
    parser.add_argument('--n-seeds', type=int, default=10,
                        help='Number of random seeds')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for training (cpu or cuda)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    if args.full_pipeline:
        args.run_experiments = True
        args.generate_results = True
    
    if args.run_experiments:
        print("="*80)
        print("Running DRO-FAIR Experiments")
        print("="*80)
        from experiments.run_experiments import run_all_experiments
        run_all_experiments(
            datasets=args.datasets,
            alphas=args.alphas,
            n_seeds=args.n_seeds,
            device=args.device,
            output_dir=args.output_dir
        )
    
    if args.generate_results:
        print("="*80)
        print("Generating Results and Plots")
        print("="*80)
        from experiments.generate_results import (
            load_results, generate_table1, plot_main_results,
            plot_test_time_eval, generate_summary_stats, compute_reductions,
            generate_latex_table
        )
        results = load_results()
        
        if not results:
            print("No results found. Please run experiments first.")
            return
        
        print(f"Loaded {len(results)} experiment results")
        
        # Generate tables
        table_data = generate_table1(results)
        print(f"Generated Table 1 with {len(table_data)} rows")
        
        # Save summary stats
        summary = generate_summary_stats(results)
        with open(os.path.join(args.output_dir, 'summary_stats.csv'), 'w') as f:
            f.write(summary)
        print(f"Saved {args.output_dir}/summary_stats.csv")
        
        # Save LaTeX table
        latex = generate_latex_table(results)
        with open(os.path.join(args.output_dir, 'table1_latex.tex'), 'w') as f:
            f.write(latex)
        print(f"Saved {args.output_dir}/table1_latex.tex")
        
        # Save table1_results.csv
        lines = ["dataset,alpha,method,acc_mean,acc_std,dp_mean,dp_std,if_mean,if_std"]
        for row in table_data:
            lines.append(f"{row['dataset']},{row['alpha']},{row['method']},{row['acc_mean']:.4f},"
                        f"{row['acc_std']:.4f},{row['dp_mean']:.4f},{row['dp_std']:.4f},"
                        f"{row['if_mean']:.4f},{row['if_std']:.4f}")
        
        with open(os.path.join(args.output_dir, 'table1_results.csv'), 'w') as f:
            f.write("\n".join(lines))
        print(f"Saved {args.output_dir}/table1_results.csv")
        
        # Generate figures
        try:
            plot_main_results(results, output_dir=args.figures_dir)
            plot_test_time_eval(results, output_dir=args.figures_dir)
        except Exception as e:
            print(f"Warning: Could not generate figures: {e}")
        
        # Compute and save reductions
        reductions = compute_reductions(results)
        print("\nDRO-FAIR Reductions (Clean Test Evaluation):")
        for r in reductions[:15]:
            print(f"  {r['dataset']} α={r['alpha']} ({r['eval']}): "
                  f"DP {r['dp_reduction']:.1f}%, IF {r['if_reduction']:.1f}%")
        
        with open(os.path.join(args.output_dir, 'reductions.json'), 'w') as f:
            json.dump(reductions, f, indent=2)
        print(f"Saved {args.output_dir}/reductions.json")
        
        print("\nAll results generated successfully!")
    
    if not args.run_experiments and not args.generate_results:
        parser.print_help()


if __name__ == '__main__':
    main()
