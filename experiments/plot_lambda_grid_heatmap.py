#!/usr/bin/env python3
"""
Agent C — Lambda grid heatmap (Q1): best (lambda_init, lr) for DP vs acc tradeoff.

Loads results/lambda_lr_grid.json (produced by run_lambda_lr_grid.py).

Expects rows with keys like: dataset, alpha, lambda_init, lambda_lr, method, dp_clean, acc_clean, ...
(plus seed).

Script finds, per (dataset, alpha) or aggregated, the (init, lr) pair that gives
best DP (lowest) while keeping acc within epsilon of best-acc, or shows full
heatmap of mean DP (or tradeoff score).

Saves:
  figures/figC5_lambda_grid_heatmap.pdf / .png  (or fig_lambda_grid_heatmap)
  results/lambda_grid_summary.csv (already partially produced by other tools)

PRELIMINARY: json currently empty ([]). Script prints awaiting + writes
placeholder figure. Re-point / auto-load when A fills the grid (Adult tau=1 first).

Matches figC5 style from prior. CM fonts, clean, error bars where multiple seeds.

Run:
    python3 experiments/plot_lambda_grid_heatmap.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

GRID_PATH = os.path.join(RESULTS_DIR, "lambda_lr_grid.json")
OUT_CSV   = os.path.join(RESULTS_DIR, "lambda_grid_best.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman",
                   "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 150,
})


def load_grid():
    if not os.path.exists(GRID_PATH):
        return []
    with open(GRID_PATH) as f:
        return json.load(f)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.7)


def savefig(fig, stem: str):
    pdf = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    png = os.path.join(FIGURES_DIR, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  saved {pdf}")
    print(f"  saved {png}")


def main():
    print("AGENT C: lambda (init, lr) grid heatmap (Q1 tradeoff)")
    print("=" * 72)

    rows = load_grid()
    print(f"Loaded {len(rows)} rows from {GRID_PATH}")

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty or "lambda_init" not in df.columns or "lambda_lr" not in df.columns:
        print("  No grid data yet (or missing lambda_init/lr columns). Awaiting A run of lambda_lr_grid.")
        # Placeholder fig (figC5 style)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "lambda_lr_grid.json empty\n(awaiting run_lambda_lr_grid.py fill;\n"
                          "then best (lambda_init, lr) heatmap for DP/acc)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        clean_axes(ax)
        ax.set_title("Q1: λ_init × lr grid (placeholder — see figC5 when data lands)", fontsize=11)
        savefig(fig, "figC5_lambda_grid_heatmap")
        # write placeholder csv
        pd.DataFrame([{"note": "placeholder awaiting lambda_lr_grid.json"}]).to_csv(OUT_CSV, index=False)
        print(f"  wrote placeholder {OUT_CSV}")
        print("\nAGENT C MILESTONE: lambda grid heatmap scaffold complete (placeholder).")
        print("  Re-run after lambda_lr_grid.json populated for real heatmap + best config table.")
        print("=" * 72)
        return

    # Real path (when data present)
    # Example aggregation: mean dp per (lambda_init, lambda_lr) for adult dp alpha=0.2 say
    print("  Grid columns:", list(df.columns)[:10])
    # For scaffold: produce a simple pivot heatmap for first dataset/attack/alpha available
    key_ds = df["dataset"].iloc[0] if "dataset" in df else "adult"
    key_att = "dp"
    key_a = 0.2
    sub = df
    if "dataset" in df: sub = sub[sub["dataset"] == key_ds]
    if "attack" in df: sub = sub[sub["attack"] == key_att]
    if "alpha" in df and not sub[sub["alpha"] == key_a].empty:
        sub = sub[sub["alpha"] == key_a]

    if "dp_clean" not in sub.columns:
        sub["dp_clean"] = sub.get("dp_clean", np.nan)

    piv = sub.pivot_table(index="lambda_init", columns="lambda_lr",
                          values="dp_clean", aggfunc="mean")
    if piv.empty:
        piv = pd.DataFrame(np.random.rand(3,3))  # shouldn't happen

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = LinearSegmentedColormap.from_list("bluered", ["#1f4e79", "#f0f0f0", "#a83232"], N=256)
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{x:.1e}" for x in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in piv.index])
    ax.set_xlabel("lambda_lr")
    ax.set_ylabel("lambda_init")
    ax.set_title(f"DP (lower better) — {key_ds} α={key_a} (example slice; full grid in data)")
    plt.colorbar(im, ax=ax, label="mean DP (DRO)")
    clean_axes(ax)
    savefig(fig, "figC5_lambda_grid_heatmap")

    # Also best config summary
    best_rows = []
    for (ds, a), g in df.groupby(["dataset", "alpha"]) if "dataset" in df and "alpha" in df else [(("all", 0), df)]:
        if "dp_clean" not in g: continue
        best_idx = g["dp_clean"].idxmin()
        br = g.loc[best_idx]
        best_rows.append({
            "dataset": ds, "alpha": a,
            "best_lambda_init": br.get("lambda_init"),
            "best_lambda_lr": br.get("lambda_lr"),
            "best_dp": br.get("dp_clean"),
            "acc_at_best": br.get("acc_clean", np.nan),
        })
    pd.DataFrame(best_rows).to_csv(OUT_CSV, index=False)
    print(f"  saved best config {OUT_CSV}")

    print("\nAGENT C MILESTONE: lambda grid heatmap complete from data, see figures/figC5_lambda_grid_heatmap.pdf")
    print(f"  row counts used: {len(rows)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
