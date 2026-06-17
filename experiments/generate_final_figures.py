#!/usr/bin/env python3
"""
Agent C — FINAL figure regeneration from complete canonical 540 + lambda grid 72/72.

Generates ALL fig_final_* figures:
  Set 1: Constant-Predictor (3 plots: acc, dp, if vs alpha)
  Set 2: Acc-DP Tradeoff (1 plot)
  Set 3: Lambda Heatmaps (4 plots: acc@0.3, acc@0.4, dp@0.3, dp@0.4)
  Set 4: Convergence (3 plots: loss, acc, dp — if history data available)
  Set 5: Wilcoxon Table (1 plot)

Usage:
    python3 experiments/generate_final_figures.py

Style: Kuldeep preferences — CM serif, SE error bars, absolute values,
       no gridlines, legend right/upper right, PDF + PNG.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CANONICAL_PATH = os.path.join(RESULTS_DIR, "canonical_tau1.json")
LAMBDA_GRID_PATH = os.path.join(RESULTS_DIR, "lambda_lr_grid.json")
WILCOXON_PATH = os.path.join(RESULTS_DIR, "canonical_wilcoxon.csv")

CONSTANT_PREDICTOR_ACC = 0.752

# ── Publication Style (Kuldeep preferences) ──────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman",
                   "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelpad": 4,
    "grid.alpha": 0.0,          # NO gridlines (clean)
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fontsize": 10,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "errorbar.capsize": 3,
})

# Colors
C_NAIVE = "#a83232"   # warm red
C_DRO   = "#1f4e79"   # deep blue

TAU_COLORS = {
    1: "#1f4e79",
    5: "#2ca02c",
    10: "#d62728",
    20: "#9467bd",
    100: "#d4880f",
}

ALPHAS = [0.1, 0.2, 0.3, 0.4]
DATASETS = ["adult", "credit", "lsac"]
DS_LABEL = {"adult": "Adult", "credit": "Credit", "lsac": "LSAC"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _ms(vals):
    """Mean and standard error."""
    a = np.array(vals, dtype=float)
    if a.size == 0:
        return np.nan, np.nan
    return float(np.mean(a)), float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def _wilcox(a, b):
    try:
        if len(a) < 3 or np.allclose(a, b):
            return 1.0
        _, p = wilcoxon(a, b, alternative="greater")
        return float(p)
    except Exception:
        return 1.0


def savefig(fig, stem):
    pdf = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    png = os.path.join(FIGURES_DIR, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  saved {pdf}")
    print(f"  saved {png}")


# ═════════════════════════════════════════════════════════════════════════════
# SET 1: Constant-Predictor Figures (x=alpha, y=metric)
# ═════════════════════════════════════════════════════════════════════════════
def set1_constant_predictor(canonical):
    """3 plots: accuracy, DP, IF vs alpha for tau in {1,5,10,20,100} + Naive."""
    print("\n  Set 1: Constant-Predictor figures")

    # Canonical has tau=1 only (from tau_ablation_tau1.json)
    # For multi-tau, load tau_ablation files
    tau_data = {}
    for tau in [1, 5, 10, 20, 100]:
        path = os.path.join(RESULTS_DIR, f"tau_ablation_tau{tau}.json")
        rows = load_json(path)
        if rows:
            tau_data[tau] = rows
        else:
            tau_data[tau] = []

    # If canonical has multiple tau values, use that too
    if canonical:
        df_canon = pd.DataFrame(canonical)
        for tau in df_canon["tau"].unique():
            if tau not in tau_data or not tau_data[tau]:
                tau_data[int(tau)] = canonical

    # Also get naive baseline from canonical
    df_canon = pd.DataFrame(canonical) if canonical else pd.DataFrame()

    for metric, metric_key, ylabel, filename in [
        ("accuracy", "acc_clean", "Accuracy", "fig_final_constant_predictor_acc"),
        ("dp_violation", "dp_clean", "DP Violation", "fig_final_constant_predictor_dp"),
        ("if_violation", "if_clean", "IF Violation", "fig_final_constant_predictor_if"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        # Plot naive (from canonical, aggregated over datasets and attacks)
        if not df_canon.empty:
            naive_rows = df_canon[df_canon["method"] == "naive"]
            if not naive_rows.empty:
                alphas_present = sorted(naive_rows["alpha"].unique())
                means, ses = [], []
                for a in alphas_present:
                    vals = naive_rows[naive_rows["alpha"] == a][metric_key].values
                    m, s = _ms(vals)
                    means.append(m)
                    ses.append(s)
                ax.errorbar(alphas_present, means, yerr=ses, marker="o",
                            color=C_NAIVE, linewidth=1.8, capsize=3,
                            markersize=6, label="Naive")

        # Plot DRO for each tau
        for tau in sorted(tau_data.keys()):
            rows = tau_data[tau]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            dro_rows = df[df["method"] == "dro"]
            if dro_rows.empty:
                continue
            alphas_present = sorted(dro_rows["alpha"].unique())
            means, ses = [], []
            for a in alphas_present:
                vals = dro_rows[dro_rows["alpha"] == a][metric_key].values
                m, s = _ms(vals)
                means.append(m)
                ses.append(s)
            color = TAU_COLORS.get(tau, "#666")
            ax.errorbar(alphas_present, means, yerr=ses, marker="s",
                        color=color, linewidth=1.8, capsize=3,
                        markersize=6, label=f"DRO (τ={tau})")

        # Constant predictor horizontal line (accuracy only)
        if metric == "accuracy":
            ax.axhline(CONSTANT_PREDICTOR_ACC, color="#888", linestyle=":",
                       linewidth=1.2, label=f"Constant predictor ({CONSTANT_PREDICTOR_ACC})")

        ax.set_xlabel(r"Corruption level $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ALPHAS)
        ax.legend(loc="best", fontsize=9)
        ax.set_xlim(-0.02, 0.42)

        fig.tight_layout()
        savefig(fig, filename)


# ═════════════════════════════════════════════════════════════════════════════
# SET 2: Acc-DP Tradeoff
# ═════════════════════════════════════════════════════════════════════════════
def set2_tradeoff(canonical):
    """Scatter: x=DP, y=accuracy per alpha. Overlay constant predictor star."""
    print("\n  Set 2: Acc-DP Tradeoff figure")
    if not canonical:
        print("    no canonical data; skipping")
        return

    df = pd.DataFrame(canonical)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    markers = {0.1: "s", 0.2: "^", 0.3: "D", 0.4: "v"}
    sizes = {0.1: 80, 0.2: 90, 0.3: 100, 0.4: 110}

    for method, color, label in [("naive", C_NAIVE, "Naive"), ("dro", C_DRO, "DRO")]:
        mdf = df[df["method"] == method]
        for alpha in ALPHAS:
            sub = mdf[mdf["alpha"] == alpha]
            if sub.empty:
                continue
            acc_mean = sub["acc_clean"].mean()
            dp_mean = sub["dp_clean"].mean()
            ax.scatter(dp_mean, acc_mean, color=color, s=sizes.get(alpha, 80),
                       marker=markers.get(alpha, "o"), zorder=3,
                       edgecolors="white", linewidths=0.6)
            dy = 5 if alpha in [0.1, 0.3] else -12
            ax.annotate(rf"$\alpha$={alpha}", (dp_mean, acc_mean),
                        fontsize=8, color="#444",
                        textcoords="offset points", xytext=(6, dy))

    # Constant predictor star
    ax.scatter(0, CONSTANT_PREDICTOR_ACC, color="#888", s=200, marker="*",
               zorder=5, edgecolors="black", linewidths=0.8,
               label=f"Constant predictor (0, {CONSTANT_PREDICTOR_ACC})")

    ax.set_xlabel(r"DP Violation ($\Delta_{\mathrm{DP}}$)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs DP Tradeoff")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    savefig(fig, "fig_final_tradeoff_vs_constant_predictor")


# ═════════════════════════════════════════════════════════════════════════════
# SET 3: Lambda Heatmaps
# ═════════════════════════════════════════════════════════════════════════════
def set3_lambda_heatmaps(lambda_grid):
    """4 heatmaps: acc@0.3, acc@0.4, dp@0.3, dp@0.4."""
    print("\n  Set 3: Lambda Heatmap figures")
    if not lambda_grid:
        print("    no lambda grid data; skipping")
        return

    df = pd.DataFrame(lambda_grid)
    df_dp = df[df["attack"] == "dp"].copy()

    for alpha_val in [0.3, 0.4]:
        sub = df_dp[df_dp["alpha"] == alpha_val]
        if sub.empty:
            print(f"    no lambda grid data for alpha={alpha_val}; skipping")
            continue

        for metric, metric_label, vmin, vmax, cmap, threshold, stem in [
            ("acc", "Accuracy", 0.5, 0.85, "YlOrRd_r", CONSTANT_PREDICTOR_ACC,
             f"fig_final_lambda_heatmap_acc_{alpha_val}"),
            ("dp", "DP Violation", 0.0, 0.55, "YlOrRd", None,
             f"fig_final_lambda_heatmap_dp_{alpha_val}"),
        ]:
            pivot = sub.pivot_table(
                values=metric, index="lambda_init", columns="lr_lambda", aggfunc="mean"
            )
            pivot = pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)

            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            data = pivot.values
            row_labels = pivot.index.tolist()
            col_labels = pivot.columns.tolist()

            im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower",
                           vmin=vmin, vmax=vmax)

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
                    if threshold is not None and metric == "acc" and val >= threshold:
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
            ax.set_title(f"{metric_label} — $\\alpha$={alpha_val}", fontsize=12)

            cbar = fig.colorbar(im, ax=ax, shrink=0.85)
            cbar.set_label(metric_label, fontsize=10)

            fig.tight_layout()
            savefig(fig, stem)


# ═════════════════════════════════════════════════════════════════════════════
# SET 4: Convergence (if history data available)
# ═════════════════════════════════════════════════════════════════════════════
def set4_convergence(canonical):
    """3 plots: loss, acc, dp convergence. Requires trainer history."""
    print("\n  Set 4: Convergence figures")
    # Check if convergence history is stored in results
    conv_path = os.path.join(RESULTS_DIR, "convergence_history.json")
    history = load_json(conv_path)

    if not history:
        print("    no convergence_history.json found; skipping")
        print("    (convergence data requires trainer.history from runs)")
        return

    for metric, ylabel, filename in [
        ("train_loss", "Training Loss", "fig_final_convergence_loss"),
        ("val_acc", "Validation Accuracy", "fig_final_convergence_acc"),
        ("val_dp", "Validation DP Violation", "fig_final_convergence_dp"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for entry in history:
            label = entry.get("label", "unknown")
            color = entry.get("color", "#666")
            linestyle = entry.get("linestyle", "solid")
            vals = entry.get(metric, [])
            if vals:
                epochs = list(range(1, len(vals) + 1))
                ax.plot(epochs, vals, label=label, color=color, linestyle=linestyle)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)

        fig.tight_layout()
        savefig(fig, filename)


# ═════════════════════════════════════════════════════════════════════════════
# SET 5: Wilcoxon Significance Table
# ═════════════════════════════════════════════════════════════════════════════
def set5_wilcoxon(canonical):
    """Generate Wilcoxon significance table figure."""
    print("\n  Set 5: Wilcoxon Table figure")
    if not canonical:
        print("    no canonical data; skipping")
        return

    # Compute Wilcoxon from canonical data
    df = pd.DataFrame(canonical)
    results = []

    for ds in DATASETS:
        for attack in ["dp", "if", "combined"]:
            for alpha in ALPHAS:
                sub = df[(df["dataset"] == ds) & (df["attack"] == attack) &
                         (df["alpha"] == alpha)]
                naive = sub[sub["method"] == "naive"]
                dro = sub[sub["method"] == "dro"]
                merged = naive.merge(dro, on="seed", suffixes=("_naive", "_dro"))
                if len(merged) < 2:
                    continue

                diff_dp = merged["dp_clean_naive"] - merged["dp_clean_dro"]
                diff_if = merged["if_clean_naive"] - merged["if_clean_dro"]

                try:
                    _, p_dp = wilcoxon(diff_dp, alternative="greater", zero_method="wilcox")
                except ValueError:
                    p_dp = 1.0
                try:
                    _, p_if = wilcoxon(diff_if, alternative="greater", zero_method="wilcox")
                except ValueError:
                    p_if = 1.0

                results.append({
                    "dataset": ds, "attack": attack, "alpha": alpha,
                    "n_seeds": len(merged),
                    "dp_naive_mean": float(merged["dp_clean_naive"].mean()),
                    "dp_dro_mean": float(merged["dp_clean_dro"].mean()),
                    "dp_diff_mean": float(diff_dp.mean()),
                    "dp_pvalue": float(p_dp),
                    "dp_sig": "***" if p_dp < 0.001 else "**" if p_dp < 0.01 else "*" if p_dp < 0.05 else "",
                    "if_naive_mean": float(merged["if_clean_naive"].mean()),
                    "if_dro_mean": float(merged["if_clean_dro"].mean()),
                    "if_diff_mean": float(diff_if.mean()),
                    "if_pvalue": float(p_if),
                    "if_sig": "***" if p_if < 0.001 else "**" if p_if < 0.01 else "*" if p_if < 0.05 else "",
                })

    if not results:
        print("    no paired data for Wilcoxon; skipping")
        return

    wilc = pd.DataFrame(results).sort_values(["dataset", "attack", "alpha"])

    # Save CSV
    wilc.to_csv(WILCOXON_PATH, index=False, float_format="%.6f")
    print(f"    saved {WILCOXON_PATH}")

    # Create table figure
    fig, ax = plt.subplots(figsize=(14, max(4, len(wilc) * 0.35 + 1)))
    ax.axis("off")

    col_labels = ["Dataset", "Attack", r"$\alpha$", "n",
                  r"$\Delta_{\mathrm{DP}}$", "p (DP)", "sig",
                  r"$\Delta_{\mathrm{IF}}$", "p (IF)", "sig"]
    cell_data = []
    cell_colors = []

    for row_idx, (_, r) in enumerate(wilc.iterrows()):
        row = [
            r["dataset"], r["attack"], f"{r['alpha']:.1f}", int(r["n_seeds"]),
            f"{r['dp_diff_mean']:+.4f}", f"{r['dp_pvalue']:.4f}", r["dp_sig"],
            f"{r['if_diff_mean']:+.4f}", f"{r['if_pvalue']:.4f}", r["if_sig"],
        ]
        cell_data.append(row)
        # Color: green if DRO wins (positive diff), red if loses
        bg = ["#f0f0f0" if row_idx % 2 == 0 else "white" for _ in range(len(col_labels))]
        if r["dp_sig"]:
            bg[6] = "#d4edda"
        if r["if_sig"]:
            bg[9] = "#d4edda"
        cell_colors.append(bg)

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellColours=cell_colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Wilcoxon Signed-Rank Tests (DRO vs Naive)\n"
                 r"$H_a$: Naive DP $>$ DRO DP  |  $*p{<}0.05$  $**p{<}0.01$  $***p{<}0.001$",
                 fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    savefig(fig, "fig_final_wilcoxon_table")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("AGENT C: FINAL figure regeneration")
    print("=" * 72)

    canonical = load_json(CANONICAL_PATH)
    lambda_grid = load_json(LAMBDA_GRID_PATH)

    print(f"Canonical: {len(canonical)} records (target: 540)")
    print(f"Lambda grid: {len(lambda_grid)} records (target: 72)")

    # Check completion
    canon_complete = len(canonical) >= 540
    lambda_complete = len(lambda_grid) >= 72
    print(f"\nCompletion status:")
    print(f"  Canonical 540: {'COMPLETE' if canon_complete else 'INCOMPLETE'} ({len(canonical)}/540)")
    print(f"  Lambda grid 72: {'COMPLETE' if lambda_complete else 'INCOMPLETE'} ({len(lambda_grid)}/72)")

    if not canon_complete or not lambda_complete:
        print("\n  WARNING: Data incomplete. Generating figures from available data.")
        print("  Re-run this script when both datasets are complete for final figures.")

    # Generate all sets
    set1_constant_predictor(canonical)
    set2_tradeoff(canonical)
    set3_lambda_heatmaps(lambda_grid)
    set4_convergence(canonical)
    set5_wilcoxon(canonical)

    # Count output files
    fig_files = [f for f in os.listdir(FIGURES_DIR) if f.startswith("fig_final_")]
    pdf_files = [f for f in fig_files if f.endswith(".pdf")]
    png_files = [f for f in fig_files if f.endswith(".png")]

    print(f"\n{'=' * 72}")
    print(f"AGENT C: Figure regeneration complete")
    print(f"  PDF files: {len(pdf_files)}")
    print(f"  PNG files: {len(png_files)}")
    print(f"  Total fig_final_* files: {len(fig_files)}")
    print(f"\nFiles:")
    for f in sorted(fig_files):
        path = os.path.join(FIGURES_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f:60s} {size:>8,d} bytes")

    if len(pdf_files) >= 12:
        print("\n  SUCCESS: 12+ final PDF figures generated")
    else:
        print(f"\n  PARTIAL: {len(pdf_files)} PDFs (some data may be missing)")

    print("=" * 72)


if __name__ == "__main__":
    main()
