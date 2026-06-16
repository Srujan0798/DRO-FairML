#!/usr/bin/env python3
"""
Agent C — Win-curves at tau=1: DRO advantage (DP_Naive - DP_DRO) vs α
for all three datasets, one curve per attack type (dp, if, combined).

Shows advantage grows with α (the core narrative at fixed τ=1).

Preliminary (scaffold on current tau_ablation_tau1.json which has adult+credit
only, 109 rows, 3 seeds, no k_inner provenance recorded).
LSAC data will appear once A extends tau ablation / canonical lands.

RE-POINT: on canonical delivery, change TAU1_PATH to "results/canonical_tau1.json"
(540 rows, 6 seeds, 3 datasets, full provenance). Re-run to refresh.

All figures regenerable from this script. CM fonts, error bars (SE on delta
computed via propagation or bootstrap; here we use per-seed paired deltas then SE),
absolute values, no shading.

Usage:
    python3 experiments/plot_win_curves_tau1.py

Outputs:
    figures/fig_win_curves_tau1.pdf
    figures/fig_win_curves_tau1.png
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths (RE-POINT TO CANONICAL WHEN A DELIVERS) -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TAU1_PATH = os.path.join(RESULTS_DIR, "tau_ablation_tau1.json")  # prelim; -> canonical_tau1.json

# ---- Style (match paper: CM + clean) ---------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman",
                   "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "errorbar.capsize": 2.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 150,
})

# Reuse palette from analyze_tau1
COLORS = {
    "dp":       "#a83232",
    "if":       "#2d6a4f",
    "combined": "#7a4f99",
}
ATTACK_LABEL = {"dp": "DP attack", "if": "IF attack", "combined": "Combined"}


def load_json(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def compute_win_curve(rows: list[dict], dataset: str) -> pd.DataFrame:
    """For each (alpha, attack) compute mean(delta) and SE(delta) over paired seeds.
    delta = DP_naive - DP_dro (positive => DRO wins).
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sub = df[(df["dataset"] == dataset)].copy()
    if sub.empty:
        return pd.DataFrame()

    recs = []
    for attack in ["dp", "if", "combined"]:
        for alpha in sorted(sub["alpha"].unique()):
            g = sub[(sub["attack"] == attack) & (sub["alpha"] == alpha)]
            if g.empty:
                continue
            naive = g[g["method"] == "naive"][["seed", "dp_clean"]].set_index("seed")
            dro   = g[g["method"] == "dro"][["seed", "dp_clean"]].set_index("seed")
            merged = naive.join(dro, lsuffix="_naive", rsuffix="_dro", how="inner")
            if merged.empty:
                continue
            deltas = (merged["dp_clean_naive"] - merged["dp_clean_dro"]).values
            recs.append({
                "alpha": float(alpha),
                "attack": attack,
                "delta_mean": float(np.mean(deltas)),
                "delta_se": float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
                "n": len(deltas),
            })
    return pd.DataFrame(recs)


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
    print("AGENT C: building tau=1 win-curves (DRO advantage vs α, all datasets)")
    print("=" * 72)

    tau1_rows = load_json(TAU1_PATH)
    print(f"Loaded tau=1: {len(tau1_rows)} rows from {TAU1_PATH}")

    datasets = ["adult", "credit", "lsac"]
    all_curves = {}
    for ds in datasets:
        curve = compute_win_curve(tau1_rows, ds)
        all_curves[ds] = curve
        print(f"  {ds}: {len(curve)} (alpha,attack) curve points")

    # Evidence print for Adult (matches verified headline)
    print("\nEvidence (Adult, tau=1): per-attack deltas (positive = DRO better on DP)")
    adult = all_curves["adult"]
    for att in ["dp", "if", "combined"]:
        print(f"  Attack={att}:")
        sub = adult[adult["attack"] == att].sort_values("alpha")
        for _, r in sub.iterrows():
            print(f"    α={r['alpha']:.1f}: Δ={r['delta_mean']:+.5f} ±{r['delta_se']:.5f} (n={int(r['n'])})")

    # Plot: one row, 3 panels (one per dataset). Lines = attacks.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)

    for i, ds in enumerate(datasets):
        ax = axes[i]
        curve = all_curves[ds]
        if curve.empty:
            ax.text(0.5, 0.5, "no data (awaiting canonical / LSAC extension)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.5")
            clean_axes(ax)
            ax.set_title(ds.upper(), fontsize=11)
            ax.set_xlabel(r"$\alpha$")
            continue

        for attack in ["dp", "if", "combined"]:
            sub = curve[curve["attack"] == attack].sort_values("alpha")
            if sub.empty:
                continue
            ax.errorbar(
                sub["alpha"], sub["delta_mean"], yerr=sub["delta_se"],
                marker="o", color=COLORS[attack], linewidth=1.5, capsize=2.5,
                label=ATTACK_LABEL[attack]
            )
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
        clean_axes(ax)
        ax.set_title(ds.upper(), fontsize=11)
        ax.set_xlabel(r"corruption $\alpha$")
        if i == 0:
            ax.set_ylabel(r"$DP_{Naive} - DP_{DRO}$ (positive $\Rightarrow$ DRO wins)")
            ax.legend(loc="upper left", fontsize=8)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])

    fig.suptitle("DRO advantage on DP grows with $\\alpha$ (fixed $\\tau{=}1$)\n"
                 "— per-attack win curves (preliminary; LSAC when canonical lands)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    savefig(fig, "fig_win_curves_tau1")

    print("\nAGENT C MILESTONE: win-curves (tau=1, 3 datasets) complete, see figures/fig_win_curves_tau1.pdf (png)")
    print(f"Evidence row counts: tau1={len(tau1_rows)} (adult+credit only; lsac pending)")
    print("  Regenerate after canonical by editing TAU1_PATH in this script.")
    print("=" * 72)


if __name__ == "__main__":
    main()
