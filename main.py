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
            load_results, generate_table1, generate_plots, generate_summary_stats
        )
        results = load_results(args.output_dir)
        generate_table1(results, args.output_dir)
        generate_plots(results, args.figures_dir)
        generate_summary_stats(results, args.output_dir)
    
    if not args.run_experiments and not args.generate_results:
        parser.print_help()


if __name__ == '__main__':
    main()
