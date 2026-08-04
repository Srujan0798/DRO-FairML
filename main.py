#!/usr/bin/env python3
"""
DRO-FAIR — thin CLI for artifact generation from committed results.

Preferred path (no retrain):
    make install && make data && make test && make validate
    make tables && make results && make paper && make report

Canonical retrain (optional; results already committed at 540 rows):
    python3 experiments/run_canonical.py
    python3 experiments/run_canonical.py --smoke

Legacy full-suite train (NOT the canonical grid) requires FORCE_LEGACY=1.
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="DRO-FAIR: generate tables/plots from committed canonical results",
        epilog="Canonical grid: results/canonical_tau1.json (540 rows). "
        "See README.md / make help.",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="DEPRECATED legacy train (n_seeds default 10). Requires FORCE_LEGACY=1.",
    )
    parser.add_argument(
        "--generate-results",
        action="store_true",
        help="Generate tables/plots from results/canonical_tau1.json (make results)",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="DEPRECATED: would train then plot. Refuses unless FORCE_LEGACY=1; "
        "prefer make full (artifacts only).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "credit", "lsac"],
        help="(legacy) datasets for --run-experiments",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4],
        help="(legacy) alphas for --run-experiments",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="(legacy) seeds for --run-experiments — not canonical n=6",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="(legacy) device for --run-experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for tables/CSVs",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    if args.full_pipeline:
        args.run_experiments = True
        args.generate_results = True

    if args.run_experiments:
        if os.environ.get("FORCE_LEGACY") != "1":
            print(
                "ERROR: --run-experiments is DEPRECATED and is NOT the canonical "
                "540-row grid (τ=1, K_inner=10, n=6).\n"
                "  Claims use: results/canonical_tau1.json (already committed).\n"
                "  Optional retrain: python3 experiments/run_canonical.py\n"
                "  Artifacts only:  make results / make full\n"
                "  Override only if you mean legacy n_seeds=10: FORCE_LEGACY=1",
                file=sys.stderr,
            )
            sys.exit(1)
        print("=" * 80)
        print("WARNING: legacy run_experiments (not canonical 540-grid)")
        print("=" * 80)
        from experiments.run_experiments import run_all_experiments

        run_all_experiments(
            datasets=args.datasets,
            alphas=args.alphas,
            n_seeds=args.n_seeds,
            device=args.device,
            output_dir=args.output_dir,
        )

    if args.generate_results:
        print("=" * 80)
        print("Generating Results and Plots (from canonical_tau1.json)")
        print("=" * 80)
        from experiments.generate_results import (
            load_results,
            generate_table1,
            plot_main_results,
            plot_test_time_eval,
            generate_summary_stats,
            compute_reductions,
            generate_latex_table,
        )

        results = load_results()
        if not results:
            raise SystemExit(
                "No convertible nested results from canonical_tau1.json. "
                "Refusing empty success."
            )

        print(f"Loaded {len(results)} experiment results")

        table_data = generate_table1(results)
        print(f"Generated Table 1 with {len(table_data)} rows")

        summary = generate_summary_stats(results)
        with open(os.path.join(args.output_dir, "summary_stats.csv"), "w") as f:
            f.write(summary)
        print(f"Saved {args.output_dir}/summary_stats.csv")

        latex = generate_latex_table(results)
        with open(os.path.join(args.output_dir, "table1_latex.tex"), "w") as f:
            f.write(latex)
        print(f"Saved {args.output_dir}/table1_latex.tex")

        lines = [
            "dataset,alpha,method,acc_mean,acc_std,dp_mean,dp_std,if_mean,if_std"
        ]
        for row in table_data:
            lines.append(
                f"{row['dataset']},{row['alpha']},{row['method']},"
                f"{row['acc_mean']:.4f},{row['acc_std']:.4f},"
                f"{row['dp_mean']:.4f},{row['dp_std']:.4f},"
                f"{row['if_mean']:.4f},{row['if_std']:.4f}"
            )
        with open(os.path.join(args.output_dir, "table1_results.csv"), "w") as f:
            f.write("\n".join(lines))
        print(f"Saved {args.output_dir}/table1_results.csv")

        try:
            plot_main_results(results, output_dir=args.figures_dir)
            plot_test_time_eval(results, output_dir=args.figures_dir)
        except Exception as e:
            print(f"Warning: Could not generate figures: {e}")

        reductions = compute_reductions(results)
        print("\nDRO-FAIR Reductions (Clean Test Evaluation):")
        for r in reductions[:15]:
            print(
                f"  {r['dataset']} α={r['alpha']} ({r['eval']}): "
                f"DP {r['dp_reduction']:.1f}%, IF {r['if_reduction']:.1f}%"
            )

        with open(os.path.join(args.output_dir, "reductions.json"), "w") as f:
            json.dump(reductions, f, indent=2)
        print(f"Saved {args.output_dir}/reductions.json")
        print("\nAll results generated successfully!")

    if not args.run_experiments and not args.generate_results:
        parser.print_help()
        print(
            "\nHint: make help · make results · make validate · "
            "python3 experiments/run_canonical.py --smoke"
        )


if __name__ == "__main__":
    main()
