#!/usr/bin/env python3
"""
Agent C — High-alpha tau ablation figures.

For Kuldeep: x=α, y=metric (accuracy / DP / IF), horizontal line at 0.78
(constant-label predictor baseline), curves for different τ values + Naive.

Loads: results/tau_ablation_tau{1,5,10,100}.json
Saves: figures/fig_high_alpha_tau.{pdf,png}  (3 subplots: acc, DP, IF)

Run:
    cd /Users/srujansai/Desktop/DRO-FairML && python experiments/plot_high_alpha_tau.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.loaders import constant_predictor_acc

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

TAU_FILES = {
    1:   os.path.join(RESULTS_DIR, "tau_ablation_tau1.json"),
    5:   os.path.join(RESULTS_DIR, "tau_ablation_tau5.json"),
    10:  os.path.join(RESULTS_DIR, "tau_ablation_tau10.json"),
    100: os.path.join(RESULTS_DIR, "tau_ablation_tau100.json"),
}

CONSTANT_PREDICTOR_ACC = constant_predictor_acc('adult')

# ---- Style -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman",
                   "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,
    "errorbar.capsize": 3,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
    "legend.fancybox": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "figure.dpi": 150,
})

TAU_COLORS = {
    "naive": "#a83232",
    1:      "#1f4e79",
    5:      "#2ca02c",
    10:     "#ff7f0e",
    100:    "#9467bd",
}
TAU_MARKERS = {
    "naive": "o",
    1:       "s",
    5:       "D",
    10:      "^",
    100:     "v",
}
TAU_LABELS = {
    "naive": "Naive",
    1:       r"DRO ($\tau{=}1$)",
    5:       r"DRO ($\tau{=}5$)",
    10:      r"DRO ($\tau{=}10$)",
    100:     r"DRO ($\tau{=}100$)",
}


def load_json(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return []
    with open(path) as f:
        return json.load(f)


def build_summary(rows: list[dict], tau_val: float) -> list[dict]:
    """For each (alpha, method) with attack=dp, compute mean±std across seeds."""
    df = pd.DataFrame(rows)
    sub = df[(df["dataset"] == "adult") & (df["attack"] == "dp")].copy()
    if sub.empty:
        return []
    recs = []
    for alpha in sorted(sub["alpha"].unique()):
        g = sub[sub["alpha"] == alpha]
        for meth in ("naive", "dro"):
            mg = g[g["method"] == meth]
            if mg.empty:
                continue
            recs.append({
                "tau": tau_val,
                "alpha": float(alpha),
                "method": meth,
                "acc_mean": mg["acc_clean"].mean(),
                "acc_std":  mg["acc_clean"].std(ddof=1) if len(mg) > 1 else 0.0,
                "dp_mean":  mg["dp_clean"].mean(),
                "dp_std":   mg["dp_clean"].std(ddof=1) if len(mg) > 1 else 0.0,
                "if_mean":  mg["if_clean"].mean(),
                "if_std":   mg["if_clean"].std(ddof=1) if len(mg) > 1 else 0.0,
                "n_seeds":  len(mg),
            })
    return recs


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3, width=0.7)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)


def savefig(fig, stem: str):
    pdf = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    png = os.path.join(FIGURES_DIR, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  saved {pdf}")
    print(f"  saved {png}")


def main():
    print("AGENT C: high-alpha tau ablation figures")
    print("=" * 72)

    # Load all tau ablation data
    all_rows = []
    for tau_val, path in sorted(TAU_FILES.items()):
        rows = load_json(path)
        print(f"  tau={tau_val}: {len(rows)} rows from {os.path.basename(path)}")
        all_rows.extend(rows)

    # Build summary
    summary = build_summary(all_rows, tau_val=None)
    # We need to process each tau file separately to tag tau correctly
    summary = []
    for tau_val, path in sorted(TAU_FILES.items()):
        rows = load_json(path)
        summary.extend(build_summary(rows, tau_val))

    sdf = pd.DataFrame(summary)
    print(f"\nSummary: {len(sdf)} (tau, alpha, method) cells")
    print(f"Alphas: {sorted(sdf['alpha'].unique())}")
    print(f"Taus:   {sorted(sdf['tau'].unique())}")

    # ---- Figure: 3 subplots (Accuracy, DP, IF) -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # Plot order: Naive first (background), then DRO curves by tau
    plot_order = ["naive", 1, 5, 10, 100]

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("acc", "Accuracy", "Accuracy vs $\\alpha$"),
        ("dp",  "DP Violation", "DP Violation vs $\\alpha$"),
        ("if",  "IF Violation", "IF Violation vs $\\alpha$"),
    ]):
        ax = axes[ax_idx]
        for key in plot_order:
            if key == "naive":
                sub = sdf[sdf["method"] == "naive"].sort_values("alpha")
                label = TAU_LABELS["naive"]
                color = TAU_COLORS["naive"]
                marker = TAU_MARKERS["naive"]
                ls = "--"
                lw = 1.2
                alpha_err = 0.6
            else:
                sub = sdf[(sdf["method"] == "dro") & (sdf["tau"] == key)].sort_values("alpha")
                label = TAU_LABELS[key]
                color = TAU_COLORS[key]
                marker = TAU_MARKERS[key]
                ls = "-"
                lw = 1.6
                alpha_err = 1.0

            if sub.empty:
                continue

            ax.errorbar(
                sub["alpha"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"],
                marker=marker, color=color, label=label,
                linewidth=lw, capsize=3, markersize=5.5, linestyle=ls,
                alpha=alpha_err,
            )

        # Horizontal baseline for accuracy
        if metric == "acc":
            ax.axhline(y=CONSTANT_PREDICTOR_ACC, color="gray", linestyle=":",
                       linewidth=1.2, label=f"Constant predictor ({CONSTANT_PREDICTOR_ACC})")

        clean_axes(ax)
        ax.set_xlabel(r"corruption $\alpha$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(sorted(sdf["alpha"].unique()))
        if metric == "acc":
            ax.set_ylim(bottom=0.45, top=0.86)
            ax.legend(loc="lower left", fontsize=7.5, ncol=1)
        else:
            ax.legend(loc="best", fontsize=7.5, ncol=1)

    fig.suptitle("Adult — Effect of $\\tau$ at high $\\alpha$ (attack=DP)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_high_alpha_tau")

    # ---- Also save individual subplots -------------------------------------
    for metric, ylabel, title, stem in [
        ("acc", "Accuracy",      "Accuracy vs $\\alpha$ — Adult (attack=DP)",       "fig_high_alpha_tau_acc"),
        ("dp",  "DP Violation",  "DP Violation vs $\\alpha$ — Adult (attack=DP)",   "fig_high_alpha_tau_dp"),
        ("if",  "IF Violation",  "IF Violation vs $\\alpha$ — Adult (attack=DP)",   "fig_high_alpha_tau_if"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
        for key in plot_order:
            if key == "naive":
                sub = sdf[sdf["method"] == "naive"].sort_values("alpha")
                label = TAU_LABELS["naive"]
                color = TAU_COLORS["naive"]
                marker = TAU_MARKERS["naive"]
                ls = "--"
                lw = 1.2
                alpha_err = 0.6
            else:
                sub = sdf[(sdf["method"] == "dro") & (sdf["tau"] == key)].sort_values("alpha")
                label = TAU_LABELS[key]
                color = TAU_COLORS[key]
                marker = TAU_MARKERS[key]
                ls = "-"
                lw = 1.6
                alpha_err = 1.0
            if sub.empty:
                continue
            ax2.errorbar(
                sub["alpha"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"],
                marker=marker, color=color, label=label,
                linewidth=lw, capsize=3, markersize=5.5, linestyle=ls,
                alpha=alpha_err,
            )
        if metric == "acc":
            ax2.axhline(y=CONSTANT_PREDICTOR_ACC, color="gray", linestyle=":",
                       linewidth=1.2, label=f"Constant predictor ({CONSTANT_PREDICTOR_ACC})")
            ax2.set_ylim(bottom=0.45, top=0.86)
            ax2.legend(loc="lower left", fontsize=8)
        else:
            ax2.legend(loc="best", fontsize=8)
        clean_axes(ax2)
        ax2.set_xlabel(r"corruption $\alpha$", fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title(title, fontsize=12)
        ax2.set_xticks(sorted(sdf["alpha"].unique()))
        fig2.tight_layout()
        savefig(fig2, stem)

    print(f"\nAGENT C MILESTONE: high-alpha tau ablation figures complete.")
    print(f"  Key finding: at α≥0.3, ALL tau values give acc < {CONSTANT_PREDICTOR_ACC}")
    print(f"  (constant predictor baseline beats DRO)")
    print("=" * 72)


if __name__ == "__main__":
    main()
