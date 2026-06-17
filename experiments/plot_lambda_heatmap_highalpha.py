#!/usr/bin/env python3
"""
Agent C — Lambda/learning-rate heatmap for high-alpha regime.

Loads: results/lambda_lr_grid.json
Creates heatmaps: x=lr_lambda, y=lambda_init, color=accuracy (and DP).
Highlights cells where acc >= 0.78 (constant predictor baseline).

Saves: figures/fig_lambda_heatmap.{pdf,png}

Run:
    cd /Users/srujansai/Desktop/DRO-FairML && python experiments/plot_lambda_heatmap_highalpha.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths -----------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LAMBDA_GRID_PATH = os.path.join(RESULTS_DIR, "lambda_lr_grid.json")
CONSTANT_PREDICTOR_ACC = 0.78

# ---- Style -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman",
                   "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "figure.dpi": 150,
})


def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3, width=0.7)


def savefig(fig, stem: str):
    pdf = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    png = os.path.join(FIGURES_DIR, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  saved {pdf}")
    print(f"  saved {png}")


def make_heatmap(pivot: pd.DataFrame, metric: str, ylabel_str: str,
                 title: str, stem: str, vmin=None, vmax=None,
                 highlight_threshold=None):
    """Create a heatmap from a pivot table (rows=lambda_init, cols=lr_lambda)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    data = pivot.values
    row_labels = pivot.index.tolist()
    col_labels = pivot.columns.tolist()

    im = ax.imshow(data, cmap="YlOrRd_r" if metric == "acc" else "YlOrRd",
                   aspect="auto", origin="lower", vmin=vmin, vmax=vmax)

    # Annotate each cell
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if (metric == "acc" and val < 0.65) else "black"
            if metric == "dp" and val > 0.4:
                text_color = "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

            # Highlight cells meeting threshold
            if highlight_threshold is not None:
                if metric == "acc" and val >= highlight_threshold:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         fill=False, edgecolor="green",
                                         linewidth=2.5, linestyle="--")
                    ax.add_patch(rect)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"{x}" for x in col_labels], fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"{x}" for x in row_labels], fontsize=10)
    ax.set_xlabel(r"$\eta_\lambda$ (lr_lambda)", fontsize=11)
    ax.set_ylabel(r"$\lambda_0$ (lambda_init)", fontsize=11)
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(ylabel_str, fontsize=10)

    fig.tight_layout()
    savefig(fig, stem)


def main():
    print("AGENT C: lambda/learning-rate heatmaps for high-alpha regime")
    print("=" * 72)

    rows = load_json(LAMBDA_GRID_PATH)
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} entries from {os.path.basename(LAMBDA_GRID_PATH)}")
    print(f"Alphas: {sorted(df['alpha'].unique())}")
    print(f"lr_lambda: {sorted(df['lr_lambda'].unique())}")
    print(f"lambda_init: {sorted(df['lambda_init'].unique())}")

    # Filter for attack=dp (the grid data uses 'acc' not 'acc_clean')
    df_dp = df[df["attack"] == "dp"].copy()
    print(f"After filtering attack=dp: {len(df_dp)} rows")

    for alpha_val in sorted(df_dp["alpha"].unique()):
        sub = df_dp[df_dp["alpha"] == alpha_val]
        print(f"\n  α={alpha_val}: {len(sub)} rows")

        # Average over seeds for accuracy
        acc_pivot = sub.pivot_table(
            values="acc", index="lambda_init", columns="lr_lambda", aggfunc="mean"
        )
        dp_pivot = sub.pivot_table(
            values="dp", index="lambda_init", columns="lr_lambda", aggfunc="mean"
        )

        # Sort axes
        acc_pivot = acc_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)
        dp_pivot = dp_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)

        print(f"  Accuracy pivot:\n{acc_pivot.to_string()}")
        print(f"  DP pivot:\n{dp_pivot.to_string()}")

        # Accuracy heatmap
        make_heatmap(
            acc_pivot, "acc", "Accuracy",
            f"Accuracy — $\\lambda$ grid (Adult, $\\alpha$={alpha_val}, attack=DP)",
            f"fig_lambda_heatmap_acc_alpha{alpha_val}",
            vmin=0.5, vmax=0.85,
            highlight_threshold=CONSTANT_PREDICTOR_ACC,
        )

        # DP heatmap
        make_heatmap(
            dp_pivot, "dp", "DP Violation",
            f"DP Violation — $\\lambda$ grid (Adult, $\\alpha$={alpha_val}, attack=DP)",
            f"fig_lambda_heatmap_dp_alpha{alpha_val}",
            vmin=0.0, vmax=0.55,
        )

    # Combined figure (both alphas side by side for accuracy)
    alphas = sorted(df_dp["alpha"].unique())
    if len(alphas) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
        for idx, alpha_val in enumerate(alphas[:2]):
            ax = axes[idx]
            sub = df_dp[df_dp["alpha"] == alpha_val]
            acc_pivot = sub.pivot_table(
                values="acc", index="lambda_init", columns="lr_lambda", aggfunc="mean"
            )
            acc_pivot = acc_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)

            data = acc_pivot.values
            im = ax.imshow(data, cmap="YlOrRd_r", aspect="auto", origin="lower",
                           vmin=0.5, vmax=0.85)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if np.isnan(val):
                        continue
                    text_color = "white" if val < 0.65 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")
                    if val >= CONSTANT_PREDICTOR_ACC:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                             fill=False, edgecolor="green",
                                             linewidth=2.5, linestyle="--")
                        ax.add_patch(rect)

            ax.set_xticks(range(len(acc_pivot.columns)))
            ax.set_xticklabels([f"{x}" for x in acc_pivot.columns], fontsize=10)
            ax.set_yticks(range(len(acc_pivot.index)))
            ax.set_yticklabels([f"{x}" for x in acc_pivot.index], fontsize=10)
            ax.set_xlabel(r"$\eta_\lambda$ (lr_lambda)", fontsize=11)
            ax.set_ylabel(r"$\lambda_0$ (lambda_init)", fontsize=11)
            ax.set_title(f"$\\alpha$={alpha_val}", fontsize=12)
            cbar = fig.colorbar(im, ax=ax, shrink=0.85)
            cbar.set_label("Accuracy", fontsize=10)

        fig.suptitle("Accuracy — $\\lambda$ grid (Adult, attack=DP)\n"
                     "Green dashed = cells with acc $\\geq$ 0.78",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        savefig(fig, "fig_lambda_heatmap")

    print(f"\nAGENT C MILESTONE: lambda heatmap figures complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
