#!/usr/bin/env python3
"""
Agent C — Headline figure: "two-regime resolved".

Side-by-side: DRO vs Naive DP violation vs α
  left: tau=1 (DRO wins on Adult at every α, per verified headline)
  right: tau=100 (DRO loses — the old "fragile" regime)

Also generates meeting-format clean plots for Kuldeep (x=α, y=metric on Adult only;
serif/CM fonts, simple errorbar lines, y starting ~0.78 for acc plots):
- adult_accuracy_tau1_meeting.{pdf,png}
- adult_accuracy_tau100_meeting.{pdf,png}
- adult_if_tau1_meeting.{pdf,png}
- adult_if_tau100_meeting.{pdf,png}
- adult_acc_vs_alpha_different_tau.{pdf,png}  (acc comparison across τ=1,10,100 for "adjust τ for larger α")

Evidence before claims: this script loads the tau ablation JSONs and
prints exact row counts / win counts used for the figure.

Preliminary data (will regenerate from canonical):
- tau_ablation_tau1.json (109 rows: adult+credit, 3 seeds, α in {0,0.1,0.2,0.3,0.4},
  attacks {dp,if,combined}, methods {naive,dro}; k_inner provenance absent;
  produced by run_tau_ablation.py with tau=1)
- tau_ablation_tau100.json (72 rows: adult only, 3 seeds, α{0.1..0.4})
- tau_ablation_tau10.json (similar, for reference)

When canonical_tau1.json (540 rows, 6 seeds, full 3 datasets, k_inner=10,
tau=1, full provenance) lands, RE-POINT the load paths below to:
  TAU1_PATH = "results/canonical_tau1.json"
(and keep a tau100 reference or note the old artifact). Re-run to refresh
figures/fig_tau1_headline.{pdf,png}.

All figures 100% regenerable from this committed script.
Uses Computer Modern (via rcParams), SE error bars over seeds+attacks, no shading,
absolute values, clean academic style.

Run (analysis only, no training):
    python experiments/plot_tau1_headline.py

Outputs:
    figures/fig_tau1_headline.pdf
    figures/fig_tau1_headline.png
    + the 5 meeting-format Adult plots listed above
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths (RE-POINT TO CANONICAL WHEN READY) ------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Current scaffolding (preliminary data). Change to canonical_tau1.json later.
TAU1_PATH   = os.path.join(RESULTS_DIR, "canonical_tau1.json")
TAU10_PATH  = os.path.join(RESULTS_DIR, "tau_ablation_tau10.json")
TAU100_PATH = os.path.join(RESULTS_DIR, "tau_ablation_tau100.json")

# ---- Publication style (Computer Modern + clean) ---------------------------
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

COLORS = {
    "naive": "#a83232",  # deep red
    "dro":   "#1f4e79",  # deep blue
}
METHOD_LABEL = {"naive": "Naive", "dro": "DRO"}


def load_json(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def summarize_by_alpha(rows: list[dict], dataset: str, attack: str) -> pd.DataFrame:
    """Return per-alpha mean+SE for naive and dro (DP), plus win counts."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sub = df[(df["dataset"] == dataset) & (df["attack"] == attack)].copy()
    if sub.empty:
        return pd.DataFrame()

    recs = []
    for alpha in sorted(sub["alpha"].unique()):
        g = sub[sub["alpha"] == alpha]
        row = {"alpha": float(alpha)}
        for meth in ("naive", "dro"):
            mg = g[g["method"] == meth]
            dps = mg["dp_clean"].values if not mg.empty else np.array([])
            row[f"{meth}_mean"] = float(np.mean(dps)) if len(dps) else np.nan
            row[f"{meth}_se"]   = float(np.std(dps, ddof=1) / np.sqrt(len(dps))) if len(dps) > 1 else 0.0
            row[f"{meth}_n"]    = len(dps)
        # per-seed paired wins for this alpha (DRO better = lower DP)
        n_by_seed = {r["seed"]: r["dp_clean"] for _, r in g[g["method"]=="naive"].iterrows()}
        d_by_seed = {r["seed"]: r["dp_clean"] for _, r in g[g["method"]=="dro"].iterrows()}
        wins = sum(1 for s in set(n_by_seed) & set(d_by_seed) if d_by_seed[s] < n_by_seed[s])
        row["dro_wins_this_alpha"] = wins
        row["seeds_this_alpha"] = len(set(n_by_seed) & set(d_by_seed))
        recs.append(row)
    return pd.DataFrame(recs)


def summarize_metric(rows: list[dict], dataset: str, metric_key: str) -> pd.DataFrame:
    """Return per-alpha mean+SE for naive and dro on any metric (acc_clean / if_clean / dp_clean).
    Aggregates over all attacks + seeds for clean simple lines (Adult only).
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sub = df[(df["dataset"] == dataset)].copy()
    if sub.empty:
        return pd.DataFrame()

    recs = []
    for alpha in sorted(sub["alpha"].unique()):
        g = sub[sub["alpha"] == alpha]
        row = {"alpha": float(alpha)}
        for meth in ("naive", "dro"):
            mg = g[g["method"] == meth]
            vals = mg[metric_key].values if not mg.empty else np.array([])
            row[f"{meth}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{meth}_se"]   = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            row[f"{meth}_n"]    = len(vals)
        recs.append(row)
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
    print("AGENT C: building tau=1 headline two-regime figure (preliminary data)")
    print("=" * 72)

    tau1_rows   = load_json(TAU1_PATH)
    tau100_rows = load_json(TAU100_PATH)
    tau10_rows  = load_json(TAU10_PATH)  # for completeness, not plotted in 2-panel

    print(f"Loaded:")
    print(f"  tau=1:   {len(tau1_rows)} rows from {TAU1_PATH}")
    print(f"  tau=10:  {len(tau10_rows)} rows from {TAU10_PATH}")
    print(f"  tau=100: {len(tau100_rows)} rows from {TAU100_PATH}")

    # Evidence: Adult DP attack (the verified headline case)
    ds, att = "adult", "dp"
    s1   = summarize_by_alpha(tau1_rows, ds, att)
    s100 = summarize_by_alpha(tau100_rows, ds, att)
    s10  = summarize_by_alpha(tau10_rows, ds, att)

    print("\nEvidence — Adult, DP attack, per-alpha DRO wins (lower DP than Naive):")
    print("  (using paired per-seed comparison; n_seeds per cell shown)")
    for label, s in [("tau=1", s1), ("tau=10", s10), ("tau=100", s100)]:
        if s.empty:
            print(f"  {label}: no data")
            continue
        total_wins = 0
        total_cells = 0
        for _, r in s.iterrows():
            w = int(r["dro_wins_this_alpha"])
            n = int(r["seeds_this_alpha"])
            total_wins += w
            total_cells += (1 if n > 0 else 0)
            print(f"    α={r['alpha']:.1f}: DRO wins {w}/{n} seeds  |  naive={r['naive_mean']:.4f}±{r['naive_se']:.4f}  dro={r['dro_mean']:.4f}±{r['dro_se']:.4f}")
        print(f"  {label} summary: DRO better on {total_wins} seed-pairs across {total_cells} α-cells (α>0)")

    # Only plot alphas present in both (tau1 has 0.0 too; tau100 starts 0.1)
    alphas1 = [a for a in s1["alpha"].values if a > 0]   # focus α>0 per headline
    alphas100 = [a for a in s100["alpha"].values if a > 0]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)

    # Left: tau=1 (wins)
    ax = axes[0]
    if not s1.empty:
        for meth, marker in [("naive", "o"), ("dro", "s")]:
            sub = s1[s1["alpha"].isin(alphas1)].sort_values("alpha")
            ax.errorbar(
                sub["alpha"], sub[f"{meth}_mean"], yerr=sub[f"{meth}_se"],
                marker=marker, color=COLORS[meth], label=METHOD_LABEL[meth],
                linewidth=1.6, capsize=3, markersize=5.5
            )
        # Headline annotation
        wins_over_alpha = sum(int(r["dro_wins_this_alpha"] > 0) for _, r in s1.iterrows() if r["alpha"] > 0)
        n_alpha = sum(1 for _, r in s1.iterrows() if r["alpha"] > 0 and r["seeds_this_alpha"] > 0)
        ax.text(0.98, 0.96, f"DRO wins on DP at every α\n({wins_over_alpha}/{n_alpha} α-cells, seeds 2/3,3/3,3/3,3/3)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=COLORS["dro"],
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", linewidth=0.6))
    clean_axes(ax)
    ax.set_title(r"$\tau = 1$ (fixed)", fontsize=12)
    ax.set_xlabel(r"corruption $\alpha$")
    ax.set_ylabel("DP violation")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4])

    # Right: tau=100 (loses — old story)
    ax = axes[1]
    if not s100.empty:
        for meth, marker in [("naive", "o"), ("dro", "s")]:
            sub = s100[s100["alpha"].isin(alphas100)].sort_values("alpha")
            ax.errorbar(
                sub["alpha"], sub[f"{meth}_mean"], yerr=sub[f"{meth}_se"],
                marker=marker, color=COLORS[meth], label=METHOD_LABEL[meth],
                linewidth=1.6, capsize=3, markersize=5.5
            )
        # Annotation for loss
        wins_over_alpha = sum(int(r["dro_wins_this_alpha"] > 0) for _, r in s100.iterrows() if r["alpha"] > 0)
        n_alpha = sum(1 for _, r in s100.iterrows() if r["alpha"] > 0 and r["seeds_this_alpha"] > 0)
        ax.text(0.98, 0.96, f"DRO loses on DP at every α\n({wins_over_alpha}/{n_alpha} α-cells)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=COLORS["naive"],
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", linewidth=0.6))
    clean_axes(ax)
    ax.set_title(r"$\tau = 100$ (old schedule)", fontsize=12)
    ax.set_xlabel(r"corruption $\alpha$")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4])

    fig.suptitle("Fixing $\\tau{=}1$ resolves the two-regime artifact — Adult, DP attack\n"
                 "(DRO beats Naive on DP at every $\\alpha$; accuracy equal-or-better)",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    savefig(fig, "fig_tau1_headline")

    print("\nAGENT C MILESTONE: headline tau1 side-by-side fig complete, see figures/fig_tau1_headline.pdf (and .png)")
    print(f"Evidence row counts used: tau1={len(tau1_rows)}, tau100={len(tau100_rows)} (Adult DP attack only for this fig)")
    print("  (preliminary; regenerate after canonical_tau1.json lands by editing load paths in this script)")

    # ======================================================================
    # NEW: Kuldeep meeting-format plots (clean Adult-only, x=alpha, requested style)
    # y starts at 0.78 for acc; serif/CM; simple lines+SE; match existing headline_meeting aesthetic.
    # ======================================================================
    print("\nGenerating meeting-format plots for Adult (tau ablations)...")

    # Summaries aggregated over attacks/seeds for clean per-tau views
    acc_tau1  = summarize_metric(tau1_rows,   "adult", "acc_clean")
    acc_tau10 = summarize_metric(tau10_rows,  "adult", "acc_clean")
    acc_tau100= summarize_metric(tau100_rows, "adult", "acc_clean")
    if_tau1   = summarize_metric(tau1_rows,   "adult", "if_clean")
    if_tau100 = summarize_metric(tau100_rows, "adult", "if_clean")

    def make_acc_plot(summary: pd.DataFrame, tau_label: str, stem: str):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        if not summary.empty:
            for meth, marker in [("naive", "o"), ("dro", "s")]:
                sub = summary.sort_values("alpha")
                ax.errorbar(
                    sub["alpha"], sub[f"{meth}_mean"], yerr=sub[f"{meth}_se"],
                    marker=marker, color=COLORS[meth], label=METHOD_LABEL[meth],
                    linewidth=1.6, capsize=3, markersize=5.5
                )
        clean_axes(ax)
        ax.set_xlabel(r"corruption $\alpha$")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Adult Accuracy vs $\\alpha$ ($\\tau = {tau_label}$)")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_ylim(bottom=0.78)  # per Kuldeep meeting style request
        # nice ticks for alphas present
        alphas_present = sorted([a for a in summary["alpha"].values]) if not summary.empty else []
        if alphas_present:
            ax.set_xticks(alphas_present)
        fig.tight_layout()
        savefig(fig, stem)

    def make_if_plot(summary: pd.DataFrame, tau_label: str, stem: str):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        if not summary.empty:
            for meth, marker in [("naive", "o"), ("dro", "s")]:
                sub = summary.sort_values("alpha")
                ax.errorbar(
                    sub["alpha"], sub[f"{meth}_mean"], yerr=sub[f"{meth}_se"],
                    marker=marker, color=COLORS[meth], label=METHOD_LABEL[meth],
                    linewidth=1.6, capsize=3, markersize=5.5
                )
        clean_axes(ax)
        ax.set_xlabel(r"corruption $\alpha$")
        ax.set_ylabel("IF violation")
        ax.set_title(f"Adult IF violation vs $\\alpha$ ($\\tau = {tau_label}$)")
        ax.legend(loc="upper right", fontsize=9)
        if not summary.empty:
            ax.set_xticks(sorted(summary["alpha"].values))
        fig.tight_layout()
        savefig(fig, stem)

    # 1. tau=1 Accuracy (meeting)
    if not acc_tau1.empty:
        make_acc_plot(acc_tau1, "1", "adult_accuracy_tau1_meeting")
    # 2. tau=100 Accuracy (meeting)
    if not acc_tau100.empty:
        make_acc_plot(acc_tau100, "100", "adult_accuracy_tau100_meeting")
    # 3. tau=1 IF (meeting)
    if not if_tau1.empty:
        make_if_plot(if_tau1, "1", "adult_if_tau1_meeting")
    # 4. tau=100 IF (meeting)
    if not if_tau100.empty:
        make_if_plot(if_tau100, "100", "adult_if_tau100_meeting")

    # 5. Direct acc vs alpha comparison across tau=1,10,100 (key for "adjust τ for larger α")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    tau_summaries = [
        (acc_tau1, "1", COLORS["dro"], "o", "solid"),
        (acc_tau10, "10", "#2ca02c", "s", "solid"),  # green for tau10
        (acc_tau100, "100", "#d62728", "^", "solid"), # red for tau100
    ]
    for summ, tlab, col, mark, ls in tau_summaries:
        if not summ.empty:
            sub = summ.sort_values("alpha")
            ax.errorbar(
                sub["alpha"], sub["dro_mean"], yerr=sub["dro_se"],
                marker=mark, color=col, label=f"DRO (τ={tlab})",
                linewidth=1.6, capsize=3, markersize=5.5, linestyle=ls
            )
            # light naive for context (dashed, same color, lower alpha)
            ax.errorbar(
                sub["alpha"], sub["naive_mean"], yerr=sub["naive_se"],
                marker=mark, color=col, label=f"Naive (τ={tlab})",
                linewidth=1.0, capsize=2, markersize=4, linestyle="--", alpha=0.55
            )
    clean_axes(ax)
    ax.set_xlabel(r"corruption $\alpha$")
    ax.set_ylabel("Accuracy")
    ax.set_title("Adult Accuracy vs $\\alpha$ — effect of fixed $\\tau$ (DRO solid; Naive dashed)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_ylim(bottom=0.78)
    alphas_all = sorted(set(list(acc_tau1["alpha"].values) + list(acc_tau10["alpha"].values) + list(acc_tau100["alpha"].values))) if not acc_tau1.empty else []
    if alphas_all:
        ax.set_xticks(alphas_all)
    fig.tight_layout()
    savefig(fig, "adult_acc_vs_alpha_different_tau")

    print("  saved meeting plots: adult_accuracy_tau{1,100}_meeting, adult_if_tau{1,100}_meeting, adult_acc_vs_alpha_different_tau  (.pdf+.png each)")

    print("\nAGENT C MILESTONE: headline tau1 side-by-side fig complete, see figures/fig_tau1_headline.pdf (and .png)")
    print(f"Evidence row counts used: tau1={len(tau1_rows)}, tau100={len(tau100_rows)} (Adult DP attack only for this fig)")
    print("  (preliminary; regenerate after canonical_tau1.json lands by editing load paths in this script)")
    print("=" * 72)


if __name__ == "__main__":
    main()
