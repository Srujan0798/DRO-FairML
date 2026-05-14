#!/usr/bin/env python3
"""
Generate All Deliverables
=========================

One-shot script to generate ALL outputs after experiments finish.
Run this AFTER `python3 experiments/run_experiments.py --n_seeds 10` completes.

Usage:
    python3 experiments/generate_all_deliverables.py

Creates:
    - results/table1_results.csv
    - results/table1_latex.tex
    - results/summary_stats.csv
    - results/reductions.json
    - figures/main_results.png
    - figures/test_time_eval.png
    - results/ablation_full.json

Then runs:
    - Theory verification
    - Professor review simulator
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"WARNING: {description} returned code {result.returncode}")
    return result.returncode == 0


def main():
    print("="*60)
    print("GENERATING ALL DELIVERABLES")
    print("="*60)

    # Merge individual results if needed
    import json
    if os.path.exists('results/individual') and os.listdir('results/individual'):
        print("\nMerging individual result files...")
        from experiments.run_robust import merge_all
        merge_all()

    # Check if results exist
    if not os.path.exists('results/all_results.json'):
        print("\n ERROR: results/all_results.json not found!")
        print("   Run experiments first: python3 experiments/run_robust.py")
        return 1

    with open('results/all_results.json') as f:
        results = json.load(f)
    print(f"\nFound {len(results)} experiment results")

    if len(results) < 150:
        print(f"   WARNING: Expected 150, got {len(results)}. Some experiments may be missing.")
        print("   Continuing with available results...")

    # 1. Generate tables and plots
    success = True
    success &= run_command(
        "python3 experiments/generate_results.py",
        "Step 1/6: Generating tables and plots"
    )

    # 2. Run ablations
    success &= run_command(
        "python3 experiments/run_ablations.py",
        "Step 2/6: Running ablation studies"
    )

    # 3. Verify theory
    success &= run_command(
        "python3 experiments/verify_theory.py",
        "Step 3/6: Verifying theoretical guarantees"
    )

    # 4. Random vs adversarial comparison
    success &= run_command(
        "python3 experiments/run_random_vs_adversarial.py",
        "Step 4/6: Running random vs adversarial comparison"
    )

    # 5. Generate extra LaTeX tables (runtime, ablation)
    success &= run_command(
        "python3 experiments/generate_latex_extras.py",
        "Step 5/6: Generating LaTeX tables for report"
    )

    # 6. Validate results
    success &= run_command(
        "python3 experiments/validate_results.py",
        "Step 6/6: Validating results (DRO wins >= 6/9)"
    )

    # Final check
    print(f"\n{'='*60}")
    print("Final deliverables check")
    print(f"{'='*60}")

    deliverables = {
        'results/all_results.json': 'Experiment results',
        'results/table1_results.csv': 'Table 1 (CSV)',
        'results/table1.tex': 'Table 1 (LaTeX)',
        'results/summary_stats.csv': 'Summary statistics',
        'results/reductions.json': 'Reduction metrics',
        'results/runtime_table.tex': 'Runtime table (LaTeX)',
        'figures/main_results.png': 'Main results plot',
        'figures/test_time_eval.png': 'Test-time eval plot',
    }

    all_present = True
    for path, desc in deliverables.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {desc}: {path} ({size:,} bytes)")
        else:
            print(f"  ❌ {desc}: {path} MISSING")
            all_present = False

    print(f"\n{'='*60}")
    if all_present and success:
        print("🎉 ALL DELIVERABLES GENERATED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now:")
        print("  1. Check results/ for tables and data")
        print("  2. Check figures/ for plots")
        print("  3. Compile report: cd report && pdflatex report.tex")
        print("  4. Run diagnostics: python3 experiments/diagnostics.py --compare")
        return 0
    else:
        print("⚠️  SOME DELIVERABLES MISSING OR FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
