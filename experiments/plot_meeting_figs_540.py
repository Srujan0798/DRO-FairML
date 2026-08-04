#!/usr/bin/env python3
"""Regenerate meeting headline + Wilcoxon table from the 540-row canonical grid.

Outputs:
  figures/fig_tau1_headline.{pdf,png}
  figures/fig_final_wilcoxon_table.{pdf,png}
  figures/figD10_final_wilcoxon_table.{pdf,png}

Run from repo root:
  python3 experiments/plot_meeting_figs_540.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "results" / "canonical_tau1.json"
FIG = ROOT / "figures"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
DATASETS = ["adult", "credit", "lsac"]


def main() -> None:
    d = json.loads(CANON.read_text())
    if len(d) != 540:
        raise SystemExit(f"expected 540 rows, got {len(d)}")
    FIG.mkdir(exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )

    # Adult DP headline
    g = collections.defaultdict(dict)
    for r in d:
        if r["dataset"] == "adult" and r["attack"] == "dp":
            g[(float(r["alpha"]), int(r["seed"]))][r["method"]] = r

    nv_m, dv_m, nv_se, dv_se, wins = [], [], [], [], []
    for a in ALPHAS:
        pairs = [
            (v["naive"], v["dro"])
            for (aa, s), v in g.items()
            if aa == a and "naive" in v and "dro" in v
        ]
        if len(pairs) != 6:
            raise SystemExit(f"adult dp a={a}: expected 6 pairs, got {len(pairs)}")
        nvs = [p[0]["dp_clean"] for p in pairs]
        dvs = [p[1]["dp_clean"] for p in pairs]
        nv_m.append(mean(nvs))
        dv_m.append(mean(dvs))
        nv_se.append(float(np.std(nvs, ddof=1) / np.sqrt(6)))
        dv_se.append(float(np.std(dvs, ddof=1) / np.sqrt(6)))
        wins.append(sum(1 for n, x in zip(nvs, dvs) if n > x))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(ALPHAS, nv_m, yerr=nv_se, marker="o", label="Naive-FAIR", color="#d62728", capsize=3)
    ax.errorbar(ALPHAS, dv_m, yerr=dv_se, marker="s", label="DRO-FAIR", color="#1f77b4", capsize=3)
    ax.axvspan(-0.05, 0.25, color="#e8f5e9", alpha=0.5, zorder=0)
    ax.text(0.08, max(nv_m) * 0.95, r"defensible $\alpha\leq0.2$", fontsize=8, color="#2e7d32")
    ax.set_xlabel(r"Corruption budget $\alpha$")
    ax.set_ylabel("DP violation (lower is better)")
    ax.set_title(r"Adult / DP attack — $\tau=1$, $n=6$ (from 540-row canonical)")
    ax.legend(loc="upper left", frameon=False)
    for a, w, ym in zip(ALPHAS, wins, nv_m):
        ax.annotate(
            f"{w}/6",
            (a, ym),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7,
            color="#555",
        )
    ax.set_xticks(ALPHAS)
    fig.savefig(FIG / "fig_tau1_headline.pdf")
    fig.savefig(FIG / "fig_tau1_headline.png", dpi=300)
    plt.close(fig)
    print("fig_tau1_headline wins", wins)

    # Wilcoxon win matrix (DP attack)
    g2 = collections.defaultdict(dict)
    for r in d:
        if r["attack"] == "dp":
            g2[(r["dataset"], float(r["alpha"]), int(r["seed"]))][r["method"]] = r

    win_mat = np.zeros((3, 5))
    p_mat = np.full((3, 5), np.nan)
    for i, ds in enumerate(DATASETS):
        for j, a in enumerate(ALPHAS):
            pairs = [
                (v["naive"], v["dro"])
                for (d0, aa, s), v in g2.items()
                if d0 == ds and aa == a and "naive" in v and "dro" in v
            ]
            nv = [p[0]["dp_clean"] for p in pairs]
            dv = [p[1]["dp_clean"] for p in pairs]
            win_mat[i, j] = sum(1 for n, x in zip(nv, dv) if n > x)
            try:
                if any(abs(n - x) > 1e-12 for n, x in zip(nv, dv)):
                    p_mat[i, j] = wilcoxon(nv, dv, alternative="greater").pvalue
                else:
                    p_mat[i, j] = 1.0
            except Exception:
                p_mat[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    im = ax.imshow(win_mat / 6.0, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_yticks(range(3))
    ax.set_yticklabels([ds.capitalize() for ds in DATASETS])
    ax.set_xlabel(r"$\alpha$")
    ax.set_title(r"DP attack: DRO win count / 6 (Wilcoxon $p$) — $\tau=1$, 540-grid")
    for i in range(3):
        for j in range(5):
            w = int(win_mat[i, j])
            p = p_mat[i, j]
            txt = f"{w}/6\np={p:.3f}" if np.isfinite(p) else f"{w}/6"
            color = "white" if win_mat[i, j] in (0, 6) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    ax.text(
        0.5,
        -0.22,
        r"Note: Adult $\alpha=0.1$ is 5/6 (not 6/6). LSAC is 0/6 at all $\alpha$ (degenerate).",
        transform=ax.transAxes,
        fontsize=8,
        color="#333",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("win fraction")
    for stem in ("fig_final_wilcoxon_table", "figD10_final_wilcoxon_table"):
        fig.savefig(FIG / f"{stem}.pdf")
        fig.savefig(FIG / f"{stem}.png", dpi=300)
    plt.close(fig)
    print("win_mat\n", win_mat)


if __name__ == "__main__":
    main()
