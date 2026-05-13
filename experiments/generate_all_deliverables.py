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

    # Check if results exist
    if not os.path.exists('results/all_results.json'):
        print("\n❌ ERROR: results/all_results.json not found!")
        print("   Run experiments first: python3 experiments/run_experiments.py --n_seeds 10")
        return 1

    with open('results/all_results.json') as f:
        import json
        results = json.load(f)
    print(f"\n✅ Found {len(results)} experiment results")

    if len(results) < 150:
        print(f"   WARNING: Expected 150, got {len(results)}. Some experiments may be missing.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1

    # 1. Generate tables and plots
    success = True
    success &= run_command(
        "python3 experiments/generate_results.py",
        "Step 1/5: Generating tables and plots"
    )

    # 2. Run ablations
    success &= run_command(
        "python3 experiments/run_ablations.py",
        "Step 2/5: Running ablation studies"
    )

    # 3. Verify theory
    success &= run_command(
        "python3 experiments/verify_theory.py",
        "Step 3/5: Verifying theoretical guarantees"
    )

    # 4. Professor review
    success &= run_command(
        "python3 experiments/professor_review_simulator.py",
        "Step 4/5: Running professor review simulator"
    )

    # 5. Final check
    print(f"\n{'='*60}")
    print("Step 5/5: Final deliverables check")
    print(f"{'='*60}")

    deliverables = {
        'results/all_results.json': 'Experiment results',
        'results/table1_results.csv': 'Table 1 (CSV)',
        'results/table1_latex.tex': 'Table 1 (LaTeX)',
        'results/summary_stats.csv': 'Summary statistics',
        'results/reductions.json': 'Reduction metrics',
        'figures/main_results.png': 'Main results plot',
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
        print("  3. Review docs/user/ for your presentation materials")
        print("  4. Run professor_review_simulator.py before submission")
        return 0
    else:
        print("⚠️  SOME DELIVERABLES MISSING OR FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
