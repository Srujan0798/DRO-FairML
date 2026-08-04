#!/usr/bin/env python3
"""Agent A4 — Random vs Adversarial summary (analysis-only).

Reads results/random_vs_adversarial.json (corruptor_type ∈ {adversarial,
random}; attack='dp'; α ∈ {0.1, 0.2}; 3 datasets × 6 seeds × 2 methods ×
2 corruptors = 144 configs).

Per (dataset, α): ΔDP = naive_dp - dro_dp for BOTH corruptor types;
multiplier = ΔDP(adv) / ΔDP(random). The historical claim (paper
S\ref{sec:results-tabular}) is that adversarial corruption raises DP
"12-40×" more than random corruption. This script tests that claim on the
new n=6 data. If the multiplier is < 12 or > 40 it is flagged explicitly.

Outputs:
    results/random_vs_adversarial_summary.md

Run:
    PYTHONPATH=. python3 experiments/summarize_a4_rva.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

ABLATION_PATH = os.path.join(RESULTS_DIR, "random_vs_adversarial.json")
OUT_MD = os.path.join(RESULTS_DIR, "random_vs_adversarial_summary.md")

ALPHA_LEVEL = 0.05
DATASETS = ["adult", "credit", "lsac"]
ALPHAS = [0.1, 0.2]
CORRUPTORS = ["adversarial", "random"]
CLAIM_LOWER, CLAIM_UPPER = 12.0, 40.0


def _expected_total():
    """3 ds x 2 alpha x 6 seeds x 2 methods x 2 corruptor = 144."""
    return 3 * 2 * 6 * 2 * 2


def _load_ablation():
    if not os.path.exists(ABLATION_PATH):
        return [], f"MISSING: {ABLATION_PATH}"
    with open(ABLATION_PATH) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{ABLATION_PATH} is not a JSON list")
    return rows, f"results/random_vs_adversarial.json ({len(rows)} rows)"


def _mean(rows, key):
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def _per_cell_means(rows):
    """(ds, alpha, method, corruptor) -> {dp, acc, n}."""
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], float(r["alpha"]), r["method"],
           r.get("corruptor_type", "adversarial"))].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = {"dp": _mean(rs, "dp_clean"), "acc": _mean(rs, "acc_clean"), "n": len(rs)}
    return out


def _wilcoxon_greater(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0.0):
        return 1.0
    try:
        _, p = wilcoxon(d, alternative="greater", zero_method="wilcox")
        return float(p)
    except ValueError:
        return 1.0


def _delta_dp_per_cell(rows):
    """Per (ds, alpha, corruptor): ΔDP = naive_dp - dro_dp; paired Wilcoxon
    H1: naive_dp > dro_dp (DRO improves fairness). Returns DataFrame."""
    df = pd.DataFrame(rows)
    out = []
    if df.empty:
        return pd.DataFrame(out)
    for (ds, alpha, corr), g in df.groupby(["dataset", "alpha", "corruptor_type"], sort=True):
        naive = g[g["method"] == "naive"][["seed", "dp_clean"]]
        dro = g[g["method"] == "dro"][["seed", "dp_clean"]]
        merged = naive.merge(dro, on="seed", suffixes=("_naive", "_dro"))
        n = len(merged)
        if n < 1:
            continue
        diff = merged["dp_clean_naive"] - merged["dp_clean_dro"]
        p = _wilcoxon_greater(diff) if n >= 2 else 1.0
        out.append({
            "dataset": ds,
            "alpha": float(alpha),
            "corruptor": corr,
            "n_pairs": int(n),
            "dp_naive_mean": float(merged["dp_clean_naive"].mean()),
            "dp_dro_mean": float(merged["dp_clean_dro"].mean()),
            "delta_dp": float(diff.mean()),
            "p_value": float(p),
            "wins_dro": int((diff > 0).sum()),
            "sig": "*" if p < ALPHA_LEVEL else "",
        })
    return pd.DataFrame(out).sort_values(["dataset", "alpha", "corruptor"]).reset_index(drop=True)


def _multiplier_table(delta_df):
    """Per (ds, alpha): multiplier = ΔDP(adv) / ΔDP(random)."""
    out = []
    if delta_df.empty:
        return pd.DataFrame(out)
    for (ds, alpha), g in delta_df.groupby(["dataset", "alpha"], sort=True):
        adv = g[g["corruptor"] == "adversarial"]
        rnd = g[g["corruptor"] == "random"]
        if adv.empty or rnd.empty:
            continue
        ddp_adv = float(adv["delta_dp"].iloc[0])
        ddp_rnd = float(rnd["delta_dp"].iloc[0])
        if abs(ddp_rnd) < 1e-9:
            mult = float("inf") if ddp_adv > 0 else float("nan")
            mult_s = "inf (ΔDP_random≈0)"
        else:
            mult = ddp_adv / ddp_rnd
            mult_s = f"{mult:.2f}"
        out.append({
            "dataset": ds,
            "alpha": float(alpha),
            "delta_dp_adv": ddp_adv,
            "delta_dp_random": ddp_rnd,
            "multiplier": mult,
            "multiplier_str": mult_s,
            "n_adv": int(adv["n_pairs"].iloc[0]),
            "n_random": int(rnd["n_pairs"].iloc[0]),
            "in_claim_range": (CLAIM_LOWER <= mult <= CLAIM_UPPER) if np.isfinite(mult) else False,
        })
    return pd.DataFrame(out)


def write_md(rows, source, n_expected, means, delta_df, mult_df):
    lines = []
    lines.append("# Agent A4 — Random vs Adversarial summary (τ=1, DP attack)")
    lines.append("")
    lines.append("Analysis-only. No new training. Source: `results/random_vs_adversarial.json` ")
    lines.append(f"({len(rows)}/{n_expected} rows). Historical claim in paper: adversarial "
                 f"corruption raises DP **12-40×** more than random corruption.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- RvA rows present: **{len(rows)}/{n_expected}** "
                 f"({100.0*len(rows)/n_expected:.1f}%)" if n_expected else "- no rows")
    if len(rows) < n_expected:
        lines.append("- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).")
    lines.append("")
    lines.append("## ΔDP = naive_dp − dro_dp per (dataset, α, corruptor)")
    lines.append("")
    lines.append("H1 (Wilcoxon): naive_dp > dro_dp. * marks p<0.05.")
    lines.append("")
    if delta_df.empty:
        lines.append("_(no paired rows yet)_")
    else:
        lines.append("| dataset | α | corruptor | n | DP_naive | DP_dro | ΔDP | wins_dro | p | sig |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in delta_df.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['alpha']:.1f} | {r['corruptor']} "
                f"| {int(r['n_pairs'])} | {r['dp_naive_mean']:.4f} "
                f"| {r['dp_dro_mean']:.4f} | {r['delta_dp']:+.4f} "
                f"| {int(r['wins_dro'])}/{int(r['n_pairs'])} "
                f"| {r['p_value']:.4f} | {r['sig']} |"
            )
    lines.append("")
    lines.append("## Multiplier table — ΔDP(adv) / ΔDP(random)")
    lines.append("")
    lines.append(f"Claimed range: **{CLAIM_LOWER:.0f}-{CLAIM_UPPER:.0f}×**. "
                 "Cells outside this range are flagged explicitly.")
    lines.append("")
    if mult_df.empty:
        lines.append("_(no multiplier cells yet — need both adv and random arms for the same (ds,α))_")
    else:
        lines.append("| dataset | α | n_adv | n_random | ΔDP_adv | ΔDP_random | multiplier | in 12-40×? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in mult_df.iterrows():
            in_range_s = "yes" if r["in_claim_range"] else (
                "**NO — below 12×**" if (np.isfinite(r["multiplier"]) and r["multiplier"] < CLAIM_LOWER)
                else ("**NO — above 40×**" if (np.isfinite(r["multiplier"]) and r["multiplier"] > CLAIM_UPPER)
                      else "**NO — ΔDP_random≈0**"))
            lines.append(
                f"| {r['dataset']} | {r['alpha']:.1f} "
                f"| {int(r['n_adv'])} | {int(r['n_random'])} "
                f"| {r['delta_dp_adv']:+.4f} | {r['delta_dp_random']:+.4f} "
                f"| {r['multiplier_str']} | {in_range_s} |"
            )
    lines.append("")
    # Verdict.
    if mult_df.empty:
        verdict = ("Not yet answerable — need both adversarial and random arms "
                   "for at least one (ds,α) cell (currently INCOMPLETE).")
    else:
        finite = mult_df[np.isfinite(mult_df["multiplier"])]
        if finite.empty:
            verdict = ("Multiplier not computable — ΔDP(random) ≈ 0 in every cell "
                       "(random corruption barely moves DP). The qualitative "
                       "direction adv ≫ random holds; the 12-40× factor cannot be "
                       "pinned down from this data.")
        else:
            med = float(finite["multiplier"].median())
            mn = float(finite["multiplier"].min())
            mx = float(finite["multiplier"].max())
            in_count = int(finite["in_claim_range"].sum())
            n_cells = len(finite)
            below = int((finite["multiplier"] < CLAIM_LOWER).sum())
            above = int((finite["multiplier"] > CLAIM_UPPER).sum())
            if in_count == n_cells:
                verdict = (f"Substantiated: all {n_cells} finite cells fall in the "
                           f"12-40× range (min={mn:.1f}×, median={med:.1f}×, max={mx:.1f}×).")
            else:
                verdict = (f"Corrected: of {n_cells} finite cells, {in_count} fall in "
                           f"12-40×, {below} are below 12×, {above} are above 40× "
                           f"(min={mn:.1f}×, median={med:.1f}×, max={mx:.1f}×). "
                           f"The '12-40×' claim should be revised to the observed range.")
    lines.append("## Verdict — is the 12-40× claim substantiated?")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines), verdict


def main():
    print("AGENT A4: Random vs Adversarial summary (analysis-only)")
    print("=" * 78)

    rows, source = _load_ablation()
    n_expected = _expected_total()
    print(f"Loaded: {source}")
    if len(rows) < n_expected:
        print(f"  INCOMPLETE: {len(rows)}/{n_expected} rows "
              f"({100.0*len(rows)/n_expected:.1f}%) — partial-data mode.")

    means = _per_cell_means(rows)
    delta_df = _delta_dp_per_cell(rows)
    mult_df = _multiplier_table(delta_df)
    print(f"Computed {len(delta_df)} ΔDP rows; {len(mult_df)} multiplier rows.")

    md_text, verdict = write_md(rows, source, n_expected, means, delta_df, mult_df)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print(f"  saved {OUT_MD}")

    print("\nVerdict:")
    print("  " + verdict)

    if not mult_df.empty:
        print("\nMultiplier table:")
        print(f"{'dataset':<8} {'α':<5} {'ΔDP_adv':<10} {'ΔDP_rnd':<10} {'mult':<10} in_12-40?")
        for _, r in mult_df.iterrows():
            in_s = "yes" if r["in_claim_range"] else "NO"
            print(f"{r['dataset']:<8} {r['alpha']:<5.1f} "
                  f"{r['delta_dp_adv']:<+10.4f} {r['delta_dp_random']:<+10.4f} "
                  f"{r['multiplier_str']:<10} {in_s}")

    print("\n" + "=" * 78)
    print("AGENT A4 SUMMARY MILESTONE complete (analysis-only).")
    print(f"  output: {OUT_MD}")
    print("=" * 78)


if __name__ == "__main__":
    main()