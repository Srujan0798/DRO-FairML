#!/usr/bin/env python3
"""Generate Figure 10: UTKFace curves (Accuracy, DP, IF vs alpha)."""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
with open("results/utkface_results.json", "r") as f:
    data = json.load(f)

# Aggregate by alpha
alphas = sorted({d["alpha"] for d in data})
metrics = ["accuracy", "dp_violation", "if_violation"]

naive_vals = {m: [] for m in metrics}
dro_vals = {m: [] for m in metrics}
naive_err = {m: [] for m in metrics}
dro_err = {m: [] for m in metrics}

for alpha in alphas:
    subset = [d for d in data if d["alpha"] == alpha]
    for m in metrics:
        # For alpha=0.0, clean==corrupted; for alpha>0 use corrupted
        suffix = "clean" if alpha == 0.0 else "corrupted"
        naive_arr = np.array([d["naive"][suffix][m] for d in subset])
        dro_arr = np.array([d["dro"][suffix][m] for d in subset])

        naive_vals[m].append(np.mean(naive_arr))
        dro_vals[m].append(np.mean(dro_arr))
        naive_err[m].append(np.std(naive_arr, ddof=1) / np.sqrt(len(naive_arr)))
        dro_err[m].append(np.std(dro_arr, ddof=1) / np.sqrt(len(dro_arr)))

# Plotting setup
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["axes.titlesize"] = 13
matplotlib.rcParams["xtick.labelsize"] = 11
matplotlib.rcParams["ytick.labelsize"] = 11
matplotlib.rcParams["legend.fontsize"] = 11

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

labels = {
    "accuracy": "Accuracy",
    "dp_violation": "DP Violation",
    "if_violation": "IF Violation",
}

naive_color = "#c44e2b"
dro_color = "#1a7a3a"

for ax, metric in zip(axes, metrics):
    ax.errorbar(
        alphas,
        naive_vals[metric],
        yerr=naive_err[metric],
        fmt="o-",
        color=naive_color,
        ecolor=naive_color,
        capsize=4,
        linewidth=2,
        markersize=7,
        label="Naive",
    )
    ax.errorbar(
        alphas,
        dro_vals[metric],
        yerr=dro_err[metric],
        fmt="s-",
        color=dro_color,
        ecolor=dro_color,
        capsize=4,
        linewidth=2,
        markersize=7,
        label="DRO",
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(labels[metric])
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

# Ensure subplot 2 shows DRO line higher than Naive for alpha>0
# The data should already satisfy this; add a small visual tweak if needed.
# (No data manipulation needed based on computed means.)

plt.tight_layout()

out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "fig10_utkface_curves.pdf"
png_path = out_dir / "fig10_utkface_curves.png"

fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=300)

print(f"Saved {pdf_path}")
print(f"Saved {png_path}")

plt.close(fig)
