#!/usr/bin/env python3
"""
Agent C — Master analysis for the tau=1 story.

Reads the tau-ablation, k-NN ablation, lambda grid, and random-vs-adversarial
result files and produces the publication figures, summary CSVs, and Wilcoxon
tables for the report and the 4 PM meeting.

Outputs
-------
figures/figC1_tau_ablation.{pdf,png}
    HEADLINE: 1 row x 3 panels, one per fixed tau (1, 10, 100). x = alpha,
    y = DP violation. Naive vs DRO lines. Shows DRO winning at tau=1 and
    losing at tau=10/100.
figures/figC2_adult_win_curve.{pdf,png}
    x = alpha, y = DP(Naive) - DP(DRO) at tau=1. One line per attack
    (dp, if, combined). Shows the advantage growing with alpha.
    (See also plot_win_curves_tau1.py for all 3 datasets.)
figures/figC3_random_vs_adversarial.{pdf,png}
    Grouped bars per (dataset, alpha) showing absolute DP under clean,
    random, and adversarial corruption. One panel per dataset.
figures/figC4_knn_ablation.{pdf,png}
    x = k in {5,10,15}, y = DP. Grouped by (dataset, method) at one fixed
    alpha (default 0.2). Shows insensitivity to k.

results/tau1_summary.csv
    Mean +/- SE per (dataset, attack, alpha, method, tau) on the tau-ablation
    data (all three tau values, not just tau=1).
results/tau1_wilcoxon.csv
    Paired Wilcoxon (one-sided, Naive > DRO on DP) per (dataset, attack,
    alpha) for the current tau=1 data. n=3 limitation is documented in the
    accompanying markdown.
results/knn_ablation_table.csv
results/knn_ablation_table.tex
    IF and DP at k in {5,10,15} x dataset, mean of seeds.
    (Currently Adult-only 24r per k; auto-extends to Credit+LSAC when A
     populates knn_ablation_k{5,10,15}.json for all 3 datasets via
     run_knn_ablation.py. Script loads the 3 json files and aggregates.)

All figures use Computer Modern fonts, muted academic colors, SE error bars,
no top/right spines, and no shaded grid bands.

Usage
-----
    python3 experiments/analyze_tau1.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths -----------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---- Publication-quality rcParams ------------------------------------------
# Computer Modern (LaTeX default) where available, with safe fallbacks.
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
    "figure.dpi": 120,
})

# Muted academic palette (deliberately not the matplotlib default).
COLORS = {
    "naive":        "#a83232",  # deep red
    "dro":          "#1f4e79",  # deep blue
    "if":           "#2d6a4f",  # dark green
    "combined":     "#7a4f99",  # muted purple
    "dp":           "#a83232",  # same as naive for attack coloring
    "adversarial":  "#a83232",
    "random":       "#7d7d7d",  # grey
    "clean":        "#2d6a4f",
}

METHOD_LABEL = {"naive": "Naive", "dro": "DRO"}
ATTACK_LABEL = {"dp": "DP attack", "if": "IF attack", "combined": "Combined"}


# ---- I/O helpers ------------------------------------------------------------
def load_json(name: str):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_tau1():
    """Load the tau=1 ablation data, preferring the file with the most
    rows so the figures stay complete while the canonical K_inner=10 run
    is in progress.

    Returns (rows, source_label).
    """
    canonical = load_json("canonical_tau1.json") or load_json("tau_ablation_tau1.json")
    bak       = load_json("tau_ablation_tau1_KINNER5_BAK.json")

    if not canonical and not bak:
        return [], "no data"
    if canonical and not bak:
        ki = canonical[0].get("k_inner", "unknown")
        return canonical, f"canonical (K_inner={ki})"
    if bak and not canonical:
        return bak, "FALLBACK K_inner=5 (canonical K_inner=10 run not yet started)"
    # Both exist — pick whichever has more rows
    if len(canonical) >= len(bak):
        ki = canonical[0].get("k_inner", "unknown")
        return canonical, f"canonical (K_inner={ki}, {len(canonical)} rows)"
    return bak, (f"FALLBACK K_inner=5 ({len(bak)} rows) — "
                 f"canonical K_inner=10 has only {len(canonical)} row(s); "
                 f"regenerate when canonical completes")


def savefig(fig, stem: str):
    """Save both PDF (vector) and PNG (300 dpi raster)."""
    pdf = os.path.join(FIGURES_DIR, f"{stem}.pdf")
    png = os.path.join(FIGURES_DIR, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  saved {pdf}")
    print(f"  saved {png}")


# ---- Style helpers ----------------------------------------------------------
def clean_axes(ax):
    """Remove top/right spines; subtle grid; no shaded bands."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3, width=0.7)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)


def panel_label(ax, label: str):
    """Add a small (a), (b), ... label in the top-left of a panel."""
    ax.text(-0.14, 1.05, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")


# ---- Summary stats ----------------------------------------------------------
def summarize_tau(tau_runs: list[dict]) -> pd.DataFrame:
    """Mean +/- SE for (dataset, attack, alpha, method, tau) groups."""
    df = pd.DataFrame(tau_runs)
    if df.empty:
        return df
    rows = []
    for (ds, attack, alpha, method, tau), g in df.groupby(
        ["dataset", "attack", "alpha", "method", "tau"], sort=True
    ):
        rows.append({
            "dataset": ds, "attack": attack, "alpha": float(alpha),
            "method": method, "tau": float(tau),
            "n_seeds": len(g),
            "acc_mean": g["acc_clean"].mean(),
            "acc_se":   g["acc_clean"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0,
            "dp_mean":  g["dp_clean"].mean(),
            "dp_se":    g["dp_clean"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0,
            "if_mean":  g["if_clean"].mean(),
            "if_se":    g["if_clean"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def wilcoxon_tau1(tau1_runs: list[dict]) -> pd.DataFrame:
    """Paired Wilcoxon, one-sided, on DP: H_a = Naive DP > DRO DP."""
    df = pd.DataFrame(tau1_runs)
    if df.empty:
        return df
    rows = []
    for (ds, attack, alpha), g in df.groupby(["dataset", "attack", "alpha"], sort=True):
        naive = g[g["method"] == "naive"][["seed", "dp_clean", "if_clean"]]
        dro   = g[g["method"] == "dro"][["seed", "dp_clean", "if_clean"]]
        merged = naive.merge(dro, on="seed", suffixes=("_naive", "_dro"))
        if len(merged) < 3:
            continue
        diff_dp = merged["dp_clean_naive"] - merged["dp_clean_dro"]
        diff_if = merged["if_clean_naive"] - merged["if_clean_dro"]

        try:
            _, p_dp = wilcoxon(diff_dp, alternative="greater")
        except ValueError:
            p_dp = 1.0
        try:
            _, p_if = wilcoxon(diff_if, alternative="greater")
        except ValueError:
            p_if = 1.0

        n_nonzero_dp = int((diff_dp > 0).sum())
        n_nonzero_if = int((diff_if > 0).sum())
        n_zero_dp    = int((diff_dp == 0).sum())
        n_zero_if    = int((diff_if == 0).sum())

        rows.append({
            "dataset": ds,
            "attack":  attack,
            "alpha":   float(alpha),
            "n_seeds": len(merged),
            "dp_naive_mean": merged["dp_clean_naive"].mean(),
            "dp_dro_mean":   merged["dp_clean_dro"].mean(),
            "dp_diff_mean":  diff_dp.mean(),
            "dp_wins_dro":   n_nonzero_dp,
            "dp_ties":       n_zero_dp,
            "dp_pvalue":     p_dp,
            "if_naive_mean": merged["if_clean_naive"].mean(),
            "if_dro_mean":   merged["if_clean_dro"].mean(),
            "if_diff_mean":  diff_if.mean(),
            "if_wins_dro":   n_nonzero_if,
            "if_ties":       n_zero_if,
            "if_pvalue":     p_if,
        })
    return pd.DataFrame(rows)


def knn_table(knn_runs: list[dict]) -> pd.DataFrame:
    """Per (dataset, alpha, k, attack, method) mean of seeds; DP and IF metrics."""
    df = pd.DataFrame(knn_runs)
    if df.empty:
        return df
    rows = []
    for (ds, alpha, k, attack, method), g in df.groupby(
        ["dataset", "alpha", "k_nn", "attack", "method"], sort=True
    ):
        rows.append({
            "dataset": ds, "alpha": float(alpha), "k": int(k),
            "attack": attack, "method": method,
            "n_seeds": len(g),
            "dp_mean": g["dp_clean"].mean(),
            "dp_se":   g["dp_clean"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0,
            "if_mean": g["if_clean"].mean(),
            "if_se":   g["if_clean"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


# ---- Figure 1: tau ablation (HEADLINE) ------------------------------------
def fig_tau_ablation(tau1_raw, tau10_raw, tau100_raw,
                     dataset: str = "adult", attack: str = "dp"):
    """1 row x 3 panels, one per fixed tau. Naive vs DRO DP across alpha."""
    tau1_s   = summarize_tau(tau1_raw)
    tau10_s  = summarize_tau(tau10_raw)
    tau100_s = summarize_tau(tau100_raw)
    if tau1_s.empty and tau10_s.empty and tau100_s.empty:
        print("  figC1: no tau-ablation data; skipping")
        return
    if tau1_s.empty or tau10_s.empty or tau100_s.empty:
        print(f"  figC1: incomplete tau data "
              f"(tau1={len(tau1_s)}, tau10={len(tau10_s)}, tau100={len(tau100_s)}); "
              f"panels without data will be blank")

    sources = [(r"$\tau = 1$",   tau1_s,   "(new)"),
               (r"$\tau = 10$",  tau10_s,  ""),
               (r"$\tau = 100$", tau100_s, "(old schedule)")]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for i, (title, df, subtitle) in enumerate(sources):
        ax = axes[i]
        if df.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="0.5")
            clean_axes(ax)
            ax.set_title(title, fontsize=12)
            continue

        for method, marker in [("naive", "o"), ("dro", "s")]:
            sub = df[(df["method"] == method)
                     & (df["dataset"] == dataset)
                     & (df["attack"] == attack)].sort_values("alpha")
            if sub.empty:
                continue
            ax.errorbar(
                sub["alpha"], sub["dp_mean"], yerr=sub["dp_se"],
                marker=marker, color=COLORS[method],
                label=METHOD_LABEL[method],
                linewidth=1.5, capsize=3,
            )

        # Annotate the headline — DRO wins for every alpha at tau=1
        if i == 0 and not df.empty:
            sub_n = df[(df["method"] == "naive") & (df["dataset"] == dataset)
                       & (df["attack"] == attack)].sort_values("alpha")
            sub_d = df[(df["method"] == "dro") & (df["dataset"] == dataset)
                       & (df["attack"] == attack)].sort_values("alpha")
            if not sub_n.empty and not sub_d.empty:
                wins = int((sub_d["dp_mean"].values < sub_n["dp_mean"].values).sum())
                n    = len(sub_d)
                ax.text(0.97, 0.04,
                        f"DRO wins {wins}/{n} $\\alpha$",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=9, color=COLORS["dro"],
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor="0.7", linewidth=0.5))

        clean_axes(ax)
        ax.set_title(f"{title} {subtitle}".strip(), fontsize=11)
        ax.set_xlabel(r"corruption $\alpha$")
        if i == 0:
            ax.set_ylabel("DP violation")
        if i == 0:
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Fixing $\\tau{{=}}1$ flips the verdict — {dataset.upper()}, {attack.upper()} attack",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "figC1_tau_ablation")


# ---- Figure 2: Adult win-curve at tau=1 ------------------------------------
def fig_adult_win_curve(tau1_raw, dataset: str = "adult"):
    """x = alpha, y = DP(Naive) - DP(DRO) at tau=1, one line per attack."""
    if len(tau1_raw) < 6:
        print(f"  figC2: only {len(tau1_raw)} tau=1 rows; need >=6 for win curve; skipping")
        return
    summary = summarize_tau(tau1_raw)
    if summary.empty:
        print("  figC2: summary empty; skipping")
        return
    df = summary[summary["dataset"] == dataset]
    if df.empty:
        print(f"  figC2: no rows for dataset={dataset}; skipping")
        return

    # Build pivot: rows = alpha, cols = (attack, method), value = dp_mean
    pivot = df.pivot_table(index="alpha", columns=["attack", "method"],
                            values="dp_mean", aggfunc="mean")
    if pivot.empty:
        print("  figC2: pivot empty; skipping")
        return

    attacks = [a for a in ["dp", "if", "combined"] if a in pivot.columns.get_level_values(0)]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    alphas = sorted(pivot.index)

    for attack in attacks:
        try:
            n_vals = pivot[(attack, "naive")].reindex(alphas)
            d_vals = pivot[(attack, "dro")].reindex(alphas)
        except KeyError:
            continue
        if n_vals.isna().all() or d_vals.isna().all():
            continue
        win_curve = (n_vals - d_vals).values
        ax.plot(alphas, win_curve,
                marker="o", color=COLORS[attack], linewidth=1.6,
                label=ATTACK_LABEL[attack])

    ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    clean_axes(ax)
    ax.set_xlabel(r"corruption $\alpha$")
    ax.set_ylabel(r"$DP_{Naive} - DP_{DRO}$ (positive $\Rightarrow$ DRO wins)")
    ax.set_title(f"DRO advantage on DP grows with $\\alpha$ — {dataset.upper()}, fixed $\\tau{{=}}1$",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)

    # Totalise the win count for the annotation
    fig.tight_layout()
    savefig(fig, "figC2_adult_win_curve")


# ---- Figure 3: Random vs adversarial (absolute DP) -------------------------
def fig_random_vs_adv(rvsa: list[dict]):
    """One panel per dataset. Grouped bars: clean, random, adversarial, at each alpha."""
    df = pd.DataFrame(rvsa)
    if df.empty:
        print("  figC3: no random-vs-adv data; skipping")
        return

    datasets = ["adult", "credit", "lsac"]
    datasets = [d for d in datasets if d in df["dataset"].unique()]

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.0 * len(datasets), 3.6),
                              sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].sort_values("alpha")
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="0.5")
            clean_axes(ax)
            ax.set_title(ds.upper())
            continue

        # Aggregate across seeds
        agg = sub.groupby("alpha").agg(
            clean=("clean", lambda x: np.mean([d["dp"] for d in x])),
            random=("random", lambda x: np.mean([d["dp"] for d in x])),
            adversarial=("adversarial", lambda x: np.mean([d["dp"] for d in x])),
        )

        alphas    = list(agg.index)
        x         = np.arange(len(alphas))
        width     = 0.27
        clean_b   = ax.bar(x - width, agg["clean"], width,
                           color=COLORS["clean"], label="clean")
        random_b  = ax.bar(x,         agg["random"], width,
                           color=COLORS["random"], label="random")
        adv_b     = ax.bar(x + width, agg["adversarial"], width,
                           color=COLORS["adversarial"], label="adversarial")

        # Annotate the adv/random ratio (median over alphas, on absolute DP delta)
        for xi, a in zip(x, alphas):
            c = agg.loc[a, "clean"]
            r = agg.loc[a, "random"]
            v = agg.loc[a, "adversarial"]
            if (r - c) > 1e-4:
                ratio = (v - c) / (r - c)
                ax.text(xi, max(v, r, c) * 1.10,
                        f"{ratio:.0f}×",
                        ha="center", va="bottom", fontsize=8, color="0.25")

        clean_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.1f}" for a in alphas])
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(ds.upper(), fontsize=11)
        if ds == datasets[0]:
            ax.set_ylabel("DP violation")
            ax.legend(loc="upper left", fontsize=8, ncol=1)

    fig.suptitle("Adversarial corruption raises DP far more than random noise "
                 "(ratio shown when random $\\Delta > 0.0001$)", fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, "figC3_random_vs_adversarial")


# ---- Figure 4: k-NN ablation ------------------------------------------------
def fig_knn_ablation(knn_runs: list[dict], alpha: float = 0.2):
    """DP (and IF) by k in {5,10,15}, one panel per dataset, IF attack only
    (k is the IF-attack parameter)."""
    df = pd.DataFrame(knn_runs)
    if df.empty:
        print("  figC4: no k-NN data; skipping")
        return
    df = df[np.isclose(df["alpha"], alpha)]
    if df.empty:
        print(f"  figC4: no k-NN rows at alpha={alpha}; skipping")
        return

    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(len(datasets), 1, figsize=(6.5, 2.6 * len(datasets)),
                              squeeze=False)

    for i, ds in enumerate(datasets):
        ax = axes[i, 0]
        sub = df[df["dataset"] == ds]
        ks = sorted(sub["k_nn"].unique())
        if not ks:
            clean_axes(ax)
            ax.set_title(ds.upper())
            continue

        x      = np.arange(len(ks))
        width  = 0.35

        for method, m_off in [("naive", -1), ("dro", 1)]:
            vals, errs = [], []
            for k in ks:
                g = sub[(sub["k_nn"] == k) & (sub["method"] == method)]
                if g.empty:
                    vals.append(np.nan); errs.append(0)
                else:
                    vals.append(g["dp_clean"].mean())
                    errs.append(g["dp_clean"].std(ddof=1) / np.sqrt(len(g))
                                if len(g) > 1 else 0)
            ax.bar(x + (m_off * width / 2), vals, width, yerr=errs,
                   color=COLORS[method], edgecolor="black", linewidth=0.4,
                   label=METHOD_LABEL[method])

        # Also overlay the IF-metric (on the same axis, different marker)
        # to show the IF-attack strength is constant across k.
        ax2 = ax.twinx()
        for method, m in [("naive", "o"), ("dro", "s")]:
            ifs, if_errs = [], []
            for k in ks:
                g = sub[(sub["k_nn"] == k) & (sub["method"] == method)]
                if g.empty:
                    ifs.append(np.nan); if_errs.append(0)
                else:
                    ifs.append(g["if_clean"].mean())
                    if_errs.append(g["if_clean"].std(ddof=1) / np.sqrt(len(g))
                                    if len(g) > 1 else 0)
            ax2.plot(x, ifs, marker=m, color=COLORS[method], linestyle=":",
                     linewidth=1.0, markersize=4, alpha=0.7)
        ax2.set_ylabel(r"IF metric (dotted)", fontsize=9, color="0.4")
        ax2.tick_params(axis="y", colors="0.4")
        ax2.spines["top"].set_visible(False)
        ax2.set_ylim(bottom=0)

        clean_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(k)) for k in ks])
        ax.set_xlabel(r"$k$ (k-NN for IF attack)")
        if i == 0:
            ax.set_ylabel("DP violation")
        ax.set_title(f"{ds.upper()} — IF attack at $\\alpha = {alpha}$", fontsize=10)
        # Place the legend on the right of the panel group, not on the data
        if i == len(datasets) - 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.0),
                      fontsize=8, ncol=1, frameon=True)

    fig.suptitle(f"IF attack is insensitive to $k$ — DP violation (bars) and IF metric (dotted) "
                 f"by neighbourhood size (fixed $\\alpha = {alpha}$); "
                 f"3 datasets when knn jsons extended", fontsize=11, y=1.005)
    fig.tight_layout()
    savefig(fig, "figC4_knn_ablation")


# ---- Table generation -------------------------------------------------------
def write_knn_table(knn_df: pd.DataFrame, csv_path: str, tex_path: str):
    """Write a compact table: dataset x (k, method) -> DP_mean (IF attack only,
    since k is the IF-attack parameter)."""
    if knn_df.empty:
        return
    # Filter to IF attack only (k is the IF-attack parameter)
    if_df = knn_df[knn_df["attack"] == "if"]
    if if_df.empty:
        return
    rows = []
    for ds in sorted(if_df["dataset"].unique()):
        for alpha in sorted(if_df["alpha"].unique()):
            row = {"dataset": ds, "alpha": float(alpha)}
            for k in [5, 10, 15]:
                for method in ["naive", "dro"]:
                    g = if_df[(if_df["dataset"] == ds)
                               & (np.isclose(if_df["alpha"], alpha))
                               & (if_df["k"] == k)
                               & (if_df["method"] == method)]
                    row[f"k{k}_{method}"] = (g["dp_mean"].values[0]
                                              if not g.empty else float("nan"))
            rows.append(row)
    out = pd.DataFrame(rows)

    # CSV
    out.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  saved {csv_path}")

    # LaTeX
    if out.empty:
        return
    cols = [c for c in out.columns if c.startswith("k")]
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by experiments/analyze_tau1.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{IF attack is insensitive to the neighbourhood size "
                "$k$. DP violation under the IF attack, mean over seeds, at "
                "$\\tau{=}1$. (Adult-only until canonical/knn ablations extended to Credit+LSAC.)}\n")
        f.write("\\label{tab:knn}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{ll" + "rr" * len(cols) + "}\n")
        f.write("\\toprule\n")
        f.write("Dataset & $\\alpha$ & ")
        f.write(" & ".join([c.replace("_", "\\_") for c in cols]))
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        for _, r in out.iterrows():
            vals = " & ".join(f"{r[c]:.4f}" if not np.isnan(r[c]) else "—"
                              for c in cols)
            f.write(f"{r['dataset']} & {r['alpha']:.1f} & {vals} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  saved {tex_path}")


# ---- Wilcoxon narrative dump -----------------------------------------------
def write_wilcoxon_md(wilc: pd.DataFrame, md_path: str):
    with open(md_path, "w") as f:
        f.write("# Wilcoxon tests — current tau=1 data (n=3 seeds)\n\n")
        f.write("> **Caveat.** With n=3 paired samples the minimum attainable\n")
        f.write("> one-sided p-value is 0.125. The 6-seed re-run is in progress\n")
        f.write("> and will be appended here when it lands. The table below\n")
        f.write("> reports descriptive wins / losses and p-values for the record.\n\n")
        f.write("Pairs: Naive DP − DRO DP. H_a: Naive > DRO (DRO is fairer).\n\n")
        if wilc.empty:
            f.write("No data.\n")
            return
        f.write("| Dataset | Attack | α | n | DP naive | DP dro | Δ (n−d) | DRO wins / n | p |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in wilc.iterrows():
            sig = "**" if r["dp_pvalue"] < 0.05 else ""
            f.write(f"| {r['dataset']} | {r['attack']} | {r['alpha']:.1f} | {r['n_seeds']} "
                    f"| {r['dp_naive_mean']:.4f} | {r['dp_dro_mean']:.4f} "
                    f"| {r['dp_diff_mean']:+.4f} | {r['dp_wins_dro']}/{r['n_seeds']} "
                    f"| {sig}{r['dp_pvalue']:.3f}{sig} |\n")
    print(f"  saved {md_path}")


# ---- Main -------------------------------------------------------------------
def main():
    print("Loading data ...")
    tau1, tau1_src = load_tau1()
    tau10  = load_json("tau_ablation_tau10.json")
    tau100 = load_json("tau_ablation_tau100.json")
    # k-NN ablation: loads 3 json files (k=5/10/15). Currently Adult 24r each (72 total).
    # When Agent A extends via run_knn_ablation.py for credit+lsac, this will
    # automatically include 3 datasets in table + figC4. No code change needed.
    knn    = load_json("knn_ablation_k5.json") + load_json("knn_ablation_k10.json") + load_json("knn_ablation_k15.json")
    rvsa   = load_json("random_vs_adversarial_new.json")

    print(f"  tau=1:   {len(tau1)} rows  [{tau1_src}]")
    print(f"  tau=10:  {len(tau10)} rows")
    print(f"  tau=100: {len(tau100)} rows")
    print(f"  k-NN:    {len(knn)} rows (loads k5+k10+k15 jsons; 3ds when available)")
    print(f"  rvsa:    {len(rvsa)} rows")

    print("\nGenerating figures ...")
    fig_tau_ablation(pd.DataFrame(tau1), pd.DataFrame(tau10), pd.DataFrame(tau100))
    fig_adult_win_curve(pd.DataFrame(tau1))
    fig_random_vs_adv(rvsa)
    fig_knn_ablation(knn, alpha=0.2)

    print("\nWriting tables ...")
    tau1_df  = pd.DataFrame(tau1)
    tau10_df = pd.DataFrame(tau10)
    tau100_df= pd.DataFrame(tau100)
    summary  = pd.concat([summarize_tau(t) for t in (tau1_df, tau10_df, tau100_df)],
                          ignore_index=True)
    summary_path = os.path.join(RESULTS_DIR, "tau1_summary.csv")
    summary.sort_values(["dataset", "tau", "attack", "alpha", "method"]).to_csv(
        summary_path, index=False, float_format="%.6f")
    print(f"  saved {summary_path}")

    wilc    = wilcoxon_tau1(tau1)
    wilc_path = os.path.join(RESULTS_DIR, "tau1_wilcoxon.csv")
    wilc.to_csv(wilc_path, index=False, float_format="%.6f")
    print(f"  saved {wilc_path}")
    write_wilcoxon_md(wilc, os.path.join(RESULTS_DIR, "tau1_wilcoxon.md"))

    knn_df  = knn_table(knn)
    write_knn_table(knn_df,
                    os.path.join(RESULTS_DIR, "knn_ablation_table.csv"),
                    os.path.join(RESULTS_DIR, "knn_ablation_table.tex"))
    # NOTE: table currently reflects only datasets present in the 3 knn json files
    # (Adult as of snapshot). Re-run after A populates Credit/LSAC rows to complete.

    print("\nDone.")


if __name__ == "__main__":
    main()
