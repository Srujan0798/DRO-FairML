#!/usr/bin/env python3
"""
Agent C — Lambda grid heatmap (Kuldeep Q1).

Reads results/lambda_lr_grid.json and produces a 1x2 (or 1x1) heatmap:
one panel per alpha, x = lr_lambda, y = lambda_init, color = DP violation
(or accuracy in a twin panel). The default config (lambda_init=0,
lr_lambda=0.001) is annotated for context.

The grid is in progress (run by Agent A); this script works on whatever
is present and is safe to re-run when more cells land.

Usage
-----
    python3 experiments/analyze_lambda_grid.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the Agent C publication-quality style
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman",
                   "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "errorbar.capsize": 3,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
    "legend.fancybox": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")


def load():
    path = os.path.join(RESULTS_DIR, "lambda_lr_grid.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def build_heatmap(grid: pd.DataFrame, metric: str, vmin=None, vmax=None,
                   cmap: str = "viridis"):
    """Pivot the grid into a (lambda_init, lr_lambda) matrix averaged over seeds."""
    if grid.empty:
        return None, None, None
    pivot = (grid.groupby(["lambda_init", "lr_lambda"])[metric]
                  .mean()
                  .unstack("lr_lambda")
                  .sort_index())
    return pivot, vmin, vmax


def plot_heatmaps(grid: pd.DataFrame, out_stem: str):
    alphas = sorted(grid["alpha"].unique())
    n_panels = len(alphas)
    if n_panels == 0:
        print("  no data; skipping")
        return

    fig, axes = plt.subplots(2, n_panels, figsize=(4.4 * n_panels, 6.0),
                              squeeze=False)

    # Compute shared color scale for DP across panels
    dp_panels = []
    for a in alphas:
        sub = grid[grid["alpha"] == a]
        piv, _, _ = build_heatmap(sub, "dp")
        if piv is not None:
            dp_panels.append(piv.values)
    if dp_panels:
        all_vals = np.concatenate([p[~np.isnan(p)] for p in dp_panels])
        vmin_dp = float(all_vals.min())
        vmax_dp = float(all_vals.max())
    else:
        vmin_dp = vmax_dp = None

    # Same for acc
    acc_panels = []
    for a in alphas:
        sub = grid[grid["alpha"] == a]
        piv, _, _ = build_heatmap(sub, "acc")
        if piv is not None:
            acc_panels.append(piv.values)
    if acc_panels:
        all_vals = np.concatenate([p[~np.isnan(p)] for p in acc_panels])
        vmin_acc = float(all_vals.min())
        vmax_acc = float(all_vals.max())
    else:
        vmin_acc = vmax_acc = None

    default_li  = 0.0
    default_lr  = 0.001

    for col, alpha in enumerate(alphas):
        sub = grid[grid["alpha"] == a] if False else grid[grid["alpha"] == alpha]

        # ---- Top row: DP ----
        ax = axes[0, col]
        piv_dp, _, _ = build_heatmap(sub, "dp")
        if piv_dp is None or piv_dp.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"$\\alpha = {alpha}$ — DP", fontsize=10)
        else:
            im = ax.imshow(piv_dp.values, aspect="auto",
                            cmap="viridis", vmin=vmin_dp, vmax=vmax_dp)
            ax.set_xticks(range(len(piv_dp.columns)))
            ax.set_xticklabels([f"{c:g}" for c in piv_dp.columns], fontsize=8)
            ax.set_yticks(range(len(piv_dp.index)))
            ax.set_yticklabels([f"{i:g}" for i in piv_dp.index], fontsize=8)
            ax.set_xlabel("lr_lambda")
            ax.set_ylabel("lambda_init" if col == 0 else "")
            ax.set_title(f"$\\alpha = {alpha}$ — DP violation", fontsize=10)
            # Annotate each cell with the mean value
            for i in range(piv_dp.shape[0]):
                for j in range(piv_dp.shape[1]):
                    val = piv_dp.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                fontsize=7, color="white" if val > (vmin_dp + vmax_dp) / 2 else "black")
            # Mark default config
            try:
                i_default = list(piv_dp.index).index(default_li)
                j_default = list(piv_dp.columns).index(default_lr)
                ax.add_patch(plt.Rectangle((j_default - 0.5, i_default - 0.5), 1, 1,
                                             fill=False, edgecolor="red", linewidth=2.0))
            except ValueError:
                pass
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ---- Bottom row: accuracy ----
        ax = axes[1, col]
        piv_acc, _, _ = build_heatmap(sub, "acc")
        if piv_acc is None or piv_acc.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"$\\alpha = {alpha}$ — Acc", fontsize=10)
        else:
            im = ax.imshow(piv_acc.values, aspect="auto",
                            cmap="cividis", vmin=vmin_acc, vmax=vmax_acc)
            ax.set_xticks(range(len(piv_acc.columns)))
            ax.set_xticklabels([f"{c:g}" for c in piv_acc.columns], fontsize=8)
            ax.set_yticks(range(len(piv_acc.index)))
            ax.set_yticklabels([f"{i:g}" for i in piv_acc.index], fontsize=8)
            ax.set_xlabel("lr_lambda")
            ax.set_ylabel("lambda_init" if col == 0 else "")
            ax.set_title(f"$\\alpha = {alpha}$ — Accuracy", fontsize=10)
            for i in range(piv_acc.shape[0]):
                for j in range(piv_acc.shape[1]):
                    val = piv_acc.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                fontsize=7, color="white" if val < (vmin_acc + vmax_acc) / 2 else "black")
            try:
                i_default = list(piv_acc.index).index(default_li)
                j_default = list(piv_acc.columns).index(default_lr)
                ax.add_patch(plt.Rectangle((j_default - 0.5, i_default - 0.5), 1, 1,
                                             fill=False, edgecolor="red", linewidth=2.0))
            except ValueError:
                pass
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Lambda init × lr_lambda grid — DP (top) and accuracy (bottom)  "
                 "(red box = paper-spec default)", fontsize=11, y=1.005)
    # Coverage note
    expected = (grid["alpha"].nunique()
                * grid["seed"].nunique() * 4 * 3)
    if expected:
        coverage = f"{len(grid)}/{expected} cells done"
        fig.text(0.99, 0.005, coverage, ha="right", va="bottom",
                 fontsize=8, color="0.4", style="italic")
    fig.tight_layout()
    pdf = os.path.join(FIGURES_DIR, f"{out_stem}.pdf")
    png = os.path.join(FIGURES_DIR, f"{out_stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  saved {pdf}")
    print(f"  saved {png}")


def write_summary_csv(grid: pd.DataFrame, csv_path: str):
    if grid.empty:
        return
    summary = (grid.groupby(["alpha", "lambda_init", "lr_lambda"])
                  .agg(dp_mean=("dp", "mean"),
                       dp_se=("dp", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
                       acc_mean=("acc", "mean"),
                       acc_se=("acc", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0),
                       n_seeds=("seed", "count"))
                  .reset_index()
                  .sort_values(["alpha", "lambda_init", "lr_lambda"]))
    summary.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  saved {csv_path}")


def main():
    print("Loading lambda grid ...")
    rows = load()
    grid = pd.DataFrame(rows)
    print(f"  {len(grid)} cells complete "
          f"(out of 2 alphas × 3 seeds × 4 lambda_inits × 3 lr_lambdas = 72)")

    if grid.empty:
        print("  no data; nothing to plot")
        return

    print("\nWriting summary CSV ...")
    write_summary_csv(grid, os.path.join(RESULTS_DIR, "lambda_grid_summary.csv"))

    print("\nGenerating heatmaps ...")
    plot_heatmaps(grid, "figC5_lambda_grid_heatmap")

    print("\nDone.")


if __name__ == "__main__":
    main()
