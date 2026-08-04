#!/usr/bin/env python3
"""
Agent C — Random-vs-adversarial bar chart (absolute DP values).

Regenerates figC3_random_vs_adversarial.{pdf,png} from the 27-row
results/random_vs_adversarial_new.json (3 datasets × 3 alphas × 3 seeds).

Key narrative (Adult α=0.2/0.3): adversarial corruption causes huge DP lift;
random noise causes tiny lift. (Adversarial is the only corruption used for
main results; random is baseline comparator only.)

Data format: each row has 'clean', 'random', 'adversarial' dicts with 'dp'/'acc'.

PRELIMINARY: this is the committed 27-row snapshot. When canonical lands
we keep this as the dedicated ablation (it is orthogonal to the main
canonical_tau1 grid). Re-run this script to refresh figure.

All figures 100% regenerable. CM fonts, error bars over seeds, absolute
DP (not percentages), clean academic style matching paper.

Run:
    python3 experiments/plot_random_vs_adversarial.py

Outputs:
    figures/figC3_random_vs_adversarial.pdf
    figures/figC3_random_vs_adversarial.png
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

RVA_PATH = os.path.join(RESULTS_DIR, "random_vs_adversarial_new.json")

# ---- CM style --------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman",
                   "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "errorbar.capsize": 3,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 150,
})

COLORS = {
    "clean":        "#2d6a4f",
    "random":       "#7d7d7d",
    "adversarial":  "#a83232",
}


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


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
    print("AGENT C: regenerating random-vs-adversarial absolute DP bar chart")
    print("=" * 72)

    rows = load_json(RVA_PATH)
    print(f"Loaded {len(rows)} rows from {RVA_PATH}")
    df = pd.DataFrame(rows)
    print(f"  datasets: {sorted(df['dataset'].unique())}")
    print(f"  alphas: {sorted(df['alpha'].unique())}")
    print(f"  seeds: {sorted(df['seed'].unique())}")

    # Evidence for headline: Adult α=0.2, 0.3 absolute DP and deltas
    print("\nEvidence — Adult α=0.2/0.3 absolute DP (mean over 3 seeds):")
    for alpha in [0.2, 0.3]:
        sub = df[(df["dataset"] == "adult") & (np.isclose(df["alpha"], alpha))]
        c = np.mean([r["clean"]["dp"] for _, r in sub.iterrows()])
        r = np.mean([r["random"]["dp"] for _, r in sub.iterrows()])
        a = np.mean([r["adversarial"]["dp"] for _, r in sub.iterrows()])
        print(f"  α={alpha}: clean={c:.4f}  random={r:.4f} (Δ{r-c:+.4f})  adversarial={a:.4f} (Δ{a-c:+.4f})")
        print(f"         adv vs random lift ratio (when randΔ>0): {(a-c)/(r-c):.1f}×")

    datasets = ["adult", "credit", "lsac"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.7),
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

        # Aggregate means + SE over seeds per alpha
        agg_rows = []
        for alpha, g in sub.groupby("alpha"):
            clean_dps = [e["clean"]["dp"] for e in g.to_dict("records")]
            rand_dps  = [e["random"]["dp"] for e in g.to_dict("records")]
            adv_dps   = [e["adversarial"]["dp"] for e in g.to_dict("records")]
            agg_rows.append({
                "alpha": alpha,
                "clean_mean": np.mean(clean_dps),
                "clean_se": np.std(clean_dps, ddof=1)/np.sqrt(len(clean_dps)) if len(clean_dps)>1 else 0,
                "rand_mean": np.mean(rand_dps),
                "rand_se": np.std(rand_dps, ddof=1)/np.sqrt(len(rand_dps)) if len(rand_dps)>1 else 0,
                "adv_mean": np.mean(adv_dps),
                "adv_se": np.std(adv_dps, ddof=1)/np.sqrt(len(adv_dps)) if len(adv_dps)>1 else 0,
            })
        agg = pd.DataFrame(agg_rows)

        x = np.arange(len(agg))
        width = 0.26
        ax.bar(x - width, agg["clean_mean"], width, yerr=agg["clean_se"],
               color=COLORS["clean"], label="clean", capsize=2)
        ax.bar(x,         agg["rand_mean"], width, yerr=agg["rand_se"],
               color=COLORS["random"], label="random", capsize=2)
        ax.bar(x + width, agg["adv_mean"], width, yerr=agg["adv_se"],
               color=COLORS["adversarial"], label="adversarial", capsize=2)

        # Ratio annotations (adv / rand delta) for cells where rand delta >0
        for xi, (_, r) in enumerate(agg.iterrows()):
            dc = r["clean_mean"]
            dr = r["rand_mean"] - dc
            da = r["adv_mean"] - dc
            if dr > 1e-4:
                ratio = da / dr
                ax.text(xi, max(r["adv_mean"], r["rand_mean"], r["clean_mean"]) * 1.08,
                        f"{ratio:.0f}×", ha="center", va="bottom", fontsize=8, color="0.3")

        clean_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.1f}" for a in agg["alpha"]])
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(ds.upper(), fontsize=11)
        if ds == "adult":
            ax.set_ylabel("DP violation (absolute)")
        if ds == datasets[0]:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Adversarial corruption (FairnessTargetedPGD) raises DP far more than random noise\n"
                 "(ratio shown where random ΔDP > 0.0001; absolute values; 27 rows)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, "figC3_random_vs_adversarial")

    print("\nAGENT C MILESTONE: random-vs-adversarial fig regenerated (absolute DP), see figures/figC3_random_vs_adversarial.pdf")
    print(f"Evidence: 27 rows used. Adult α=0.2/0.3 adversarial >> random as shown above.")
    print("  (standalone committed generator; rerun to refresh)")
    print("=" * 72)


if __name__ == "__main__":
    main()
