#!/usr/bin/env python3
"""Accuracy vs alpha in win-curves format (3 panels, one per dataset, lines per attack)."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
FIGURES_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

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

COLORS = {
    "dp":       "#a83232",
    "if":       "#2d6a4f",
    "combined": "#7a4f99",
}
ATTACK_LABEL = {"dp": "DP attack", "if": "IF attack", "combined": "Combined"}

with open("results/canonical_tau1.json") as f:
    rows = json.load(f)

def compute_acc_advantage(rows, dataset):
    """For each (alpha, attack) compute mean accuracy for naive and dro."""
    df = pd.DataFrame(rows)
    sub = df[df["dataset"] == dataset]
    recs = []
    for attack in ["dp", "if", "combined"]:
        for alpha in sorted(sub["alpha"].unique()):
            g = sub[(sub["attack"] == attack) & (sub["alpha"] == alpha)]
            naive = g[g["method"] == "naive"]["acc_clean"].values
            dro = g[g["method"] == "dro"]["acc_clean"].values
            if len(naive) and len(dro):
                recs.append({
                    "alpha": float(alpha), "attack": attack,
                    "naive_mean": float(np.mean(naive)),
                    "naive_se": float(np.std(naive, ddof=1) / np.sqrt(len(naive))) if len(naive) > 1 else 0,
                    "dro_mean": float(np.mean(dro)),
                    "dro_se": float(np.std(dro, ddof=1) / np.sqrt(len(dro))) if len(dro) > 1 else 0,
                    "n": len(dro),
                })
    return pd.DataFrame(recs)

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3, width=0.7)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

datasets = ["adult", "credit", "lsac"]
all_curves = {}
for ds in datasets:
    all_curves[ds] = compute_acc_advantage(rows, ds)
    print(f"  {ds}: {len(all_curves[ds])} curve points")

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)

for i, ds in enumerate(datasets):
    ax = axes[i]
    curve = all_curves[ds]
    if curve.empty:
        ax.text(0.5, 0.5, "no data (LSAC pending canonical)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.5")
        clean_axes(ax)
        ax.set_title(ds.upper(), fontsize=11)
        ax.set_xlabel(r"corruption $\alpha$")
        continue

    for attack in ["dp", "if", "combined"]:
        sub = curve[curve["attack"] == attack].sort_values("alpha")
        if sub.empty:
            continue
        # DRO accuracy (solid)
        ax.errorbar(sub["alpha"], sub["dro_mean"], yerr=sub["dro_se"],
                    marker="o", color=COLORS[attack], linewidth=1.5, capsize=2.5,
                    label=f'{ATTACK_LABEL[attack]} DRO')
        # Naive accuracy (dashed, lighter)
        ax.errorbar(sub["alpha"], sub["naive_mean"], yerr=sub["naive_se"],
                    marker="o", color=COLORS[attack], linewidth=1.0, capsize=2, alpha=0.5,
                    linestyle="--", label=f'{ATTACK_LABEL[attack]} Naive')
    clean_axes(ax)
    ax.set_title(ds.upper(), fontsize=11)
    ax.set_xlabel(r"corruption $\alpha$")
    if i == 0:
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])

fig.suptitle("Accuracy vs $\\alpha$ by attack type (fixed $\\tau{=}1$)\n"
             "(solid = DRO, dashed = Naive)",
             fontsize=11, y=1.03)
fig.tight_layout()

pdf = os.path.join(FIGURES_DIR, "fig_acc_win_curves_tau1.pdf")
png = os.path.join(FIGURES_DIR, "fig_acc_win_curves_tau1.png")
fig.savefig(pdf)
fig.savefig(png, dpi=300)
plt.close(fig)
print(f"\nSaved {pdf}")
print(f"Saved {png}")

# Also print evidence
print("\nEvidence (Adult accuracy by attack):")
adult = all_curves["adult"]
for att in ["dp", "if", "combined"]:
    print(f"  {att}:")
    sub = adult[adult["attack"] == att].sort_values("alpha")
    for _, r in sub.iterrows():
        print(f"    α={r['alpha']:.1f}: Naive={r['naive_mean']:.4f}  DRO={r['dro_mean']:.4f}  (n={int(r['n'])})")
