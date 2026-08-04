#!/usr/bin/env python3
"""
Agent C — Q5: Uniform vs empirical radii comparison (canonical_tau1 vs _empirical).

Loads:
  results/canonical_tau1.json          (DRO with radii_mode='uniform')
  results/canonical_tau1_empirical.json (DRO with radii_mode='empirical')

For matching (dataset, alpha, attack, seed) pairs, compare DRO DP (and acc).
Produces:
  results/uniform_vs_empirical_comparison.csv  (per-cell means + delta)
  figures/figC_uniform_vs_emp.pdf / .png     (small multi-panel: DP for uniform vs emp per ds/attack)

Narrative question: does using attack-known empirical radii tighten (lower) the
achieved DP for DRO, or change the win pattern vs Naive?

PRELIMINARY: both files absent until A produces canonical_tau1 + empirical companion
(per §3 of MASTER_PLAN). Script runs gracefully, prints "awaiting data", writes
placeholder CSV noting the requirement.

When data lands: re-run; it will auto-detect both files (row counts must match
grid: 3ds x 5a x 3att x 6seeds for DRO rows).

All regenerable. CM style, error bars, traceable to exact json rows (filter
method=dro + radii provenance if recorded in future rows).

Run:
    python3 experiments/plot_uniform_vs_empirical.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

UNIFORM_PATH   = os.path.join(RESULTS_DIR, "canonical_tau1.json")
EMPIRICAL_PATH = os.path.join(RESULTS_DIR, "canonical_tau1_empirical.json")

OUT_CSV = os.path.join(RESULTS_DIR, "uniform_vs_empirical_comparison.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman",
                   "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "errorbar.capsize": 3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 150,
})


def load_json(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def pair_uniform_emp(u_rows, e_rows):
    """Return df of paired (ds, alpha, attack, seed) : uniform_dp, emp_dp, etc for DRO only."""
    if not u_rows or not e_rows:
        return pd.DataFrame()

    def index(rows):
        idx = {}
        for r in rows:
            if r.get("method") != "dro":
                continue
            key = (r["dataset"], float(r["alpha"]), r["attack"], int(r["seed"]))
            idx[key] = r
        return idx

    ui = index(u_rows)
    ei = index(e_rows)
    recs = []
    for k, ur in ui.items():
        if k not in ei:
            continue
        er = ei[k]
        recs.append({
            "dataset": k[0], "alpha": k[1], "attack": k[2], "seed": k[3],
            "uniform_dp": ur.get("dp_clean", np.nan),
            "emp_dp": er.get("dp_clean", np.nan),
            "uniform_acc": ur.get("acc_clean", np.nan),
            "emp_acc": er.get("acc_clean", np.nan),
        })
    return pd.DataFrame(recs)


def summarize_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return paired
    rows = []
    for (ds, alpha, attack), g in paired.groupby(["dataset", "alpha", "attack"], sort=True):
        n = len(g)
        ud = g["uniform_dp"].mean()
        ed = g["emp_dp"].mean()
        ua = g["uniform_acc"].mean()
        ea = g["emp_acc"].mean()
        rows.append({
            "dataset": ds, "alpha": float(alpha), "attack": attack, "n_seeds": n,
            "uniform_dp_mean": ud, "emp_dp_mean": ed,
            "dp_delta_emp_minus_uniform": ed - ud,  # negative => empirical tighter (better)
            "uniform_acc_mean": ua, "emp_acc_mean": ea,
            "acc_delta": ea - ua,
        })
    return pd.DataFrame(rows)


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
    print("AGENT C: Q5 uniform vs empirical radii (awaiting canonical data)")
    print("=" * 72)

    u = load_json(UNIFORM_PATH)
    e = load_json(EMPIRICAL_PATH)
    print(f"uniform (canonical_tau1): {len(u) if u else 0} rows")
    print(f"empirical (canonical_tau1_empirical): {len(e) if e else 0} rows")

    paired = pair_uniform_emp(u or [], e or [])
    print(f"paired DRO rows: {len(paired)}")

    comp = summarize_comparison(paired)
    if not comp.empty:
        comp.to_csv(OUT_CSV, index=False, float_format="%.6f")
        print(f"saved {OUT_CSV}")
        # Evidence
        print("\nEvidence (sample deltas; negative dp_delta => empirical tightens DP):")
        for ds in ["adult", "credit", "lsac"]:
            sub = comp[comp["dataset"] == ds]
            if not sub.empty:
                print(f"  {ds}: {len(sub)} cells")
                for _, r in sub.head(3).iterrows():
                    print(f"    α={r['alpha']:.1f} {r['attack']}: ΔDP(emp-u)={r['dp_delta_emp_minus_uniform']:+.5f}")
    else:
        # Write placeholder
        with open(OUT_CSV, "w") as f:
            f.write("# Placeholder — awaiting canonical_tau1.json + canonical_tau1_empirical.json\n")
            f.write("# Per MASTER_PLAN §3: both must be produced by A (540 rows uniform + empirical companion).\n")
            f.write("dataset,alpha,attack,n_seeds,uniform_dp_mean,emp_dp_mean,dp_delta_emp_minus_uniform\n")
        print(f"  wrote placeholder {OUT_CSV} (no data yet)")

    # Small figure skeleton (3 ds x 1 attack or grid of deltas)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for i, ds in enumerate(["adult", "credit", "lsac"]):
        ax = axes[i]
        sub = comp[comp["dataset"] == ds] if not comp.empty else pd.DataFrame()
        if sub.empty:
            ax.text(0.5, 0.5, "awaiting\ncanonical+empirical\ndata", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="0.5")
            clean_axes(ax)
            ax.set_title(ds.upper())
            continue
        for att, m in [("dp", "o"), ("if", "s"), ("combined", "^")]:
            ss = sub[sub["attack"] == att].sort_values("alpha")
            if ss.empty:
                continue
            ax.plot(ss["alpha"], ss["dp_delta_emp_minus_uniform"], marker=m,
                    label=att, linewidth=1.3)
        ax.axhline(0, color="0.6", ls="--", lw=0.8)
        clean_axes(ax)
        ax.set_title(ds.upper())
        ax.set_xlabel(r"$\alpha$")
        if i == 0:
            ax.set_ylabel(r"$\Delta$ DP (empirical − uniform; <0 $\Rightarrow$ emp tighter)")
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Q5: Does empirical radii (attack-known) tighten DRO DP vs uniform? (prelim placeholder)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, "figC_uniform_vs_emp")

    print("\nAGENT C MILESTONE: uniform-vs-emp fig+table scaffold complete (placeholder until data).")
    print("  Re-run after both canonical files exist; will use full 6-seed paired rows.")
    print(f"  Evidence: uniform rows={len(u) if u else 0}, emp rows={len(e) if e else 0}, paired={len(paired)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
