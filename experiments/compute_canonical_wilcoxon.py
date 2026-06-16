#!/usr/bin/env python3
"""
Agent C — n=6 (or current) paired Wilcoxon significance for canonical_tau1.

H1 (one-sided): Naive_DP > DRO_DP  i.e. DRO improves fairness (lower DP).

Produces:
    results/canonical_wilcoxon.csv
    (and .md narrative)

For every (dataset, attack, alpha) with paired seeds:
- dp_naive_mean, dp_dro_mean, dp_diff_mean (naive-dro >0 => DRO wins)
- dp_wins_dro (count of seeds where DRO DP < Naive DP)
- dp_pvalue (scipy.stats.wilcoxon, alternative="greater")
- sig marker: * if p < 0.05

Same for IF metric (secondary).

PRELIMINARY: currently falls back to tau_ablation_tau1.json (n=3 seeds) so min
p=0.125; no * will appear. After A delivers results/canonical_tau1.json
(6 seeds, 540 rows, full provenance) this script will automatically use it
(first check) and n=6 will allow p<0.05 in cells where the effect is consistent.

RE-POINT / auto: this script prefers canonical_tau1.json if present, else
tau_ablation_tau1.json. On canonical, also consider writing a version of
fairness_pgd_wilcoxon.csv if needed for legacy figs, but canonical_wilcoxon
is the authoritative for the paper.

All numbers traceable to specific json rows (per (ds,alpha,attack,seed,method)).

Run (analysis only):
    python3 experiments/compute_canonical_wilcoxon.py

Also run by analyze_tau1 or other generators.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

CANONICAL_PATH = os.path.join(RESULTS_DIR, "canonical_tau1.json")
TAU1_FALLBACK  = os.path.join(RESULTS_DIR, "tau_ablation_tau1.json")

OUT_CSV = os.path.join(RESULTS_DIR, "canonical_wilcoxon.csv")
OUT_MD  = os.path.join(RESULTS_DIR, "canonical_wilcoxon.md")


def load_preferred() -> tuple[list[dict], str]:
    if os.path.exists(CANONICAL_PATH):
        with open(CANONICAL_PATH) as f:
            rows = json.load(f)
        return rows, f"canonical ({len(rows)} rows, 6 seeds expected)"
    if os.path.exists(TAU1_FALLBACK):
        with open(TAU1_FALLBACK) as f:
            rows = json.load(f)
        return rows, f"fallback tau_ablation_tau1 ({len(rows)} rows, n<=3 seeds)"
    return [], "no data"


def compute_wilcoxon(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    for (ds, attack, alpha), g in df.groupby(["dataset", "attack", "alpha"], sort=True):
        naive = g[g["method"] == "naive"][["seed", "dp_clean", "if_clean"]]
        dro   = g[g["method"] == "dro"][["seed", "dp_clean", "if_clean"]]
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

        n_nonzero_dp = int((diff_dp > 0).sum())
        n_zero_dp    = int((diff_dp == 0).sum())
        n_nonzero_if = int((diff_if > 0).sum())
        n_zero_if    = int((diff_if == 0).sum())

        out.append({
            "dataset": ds,
            "attack": attack,
            "alpha": float(alpha),
            "n_seeds": len(merged),
            "dp_naive_mean": float(merged["dp_clean_naive"].mean()),
            "dp_dro_mean": float(merged["dp_clean_dro"].mean()),
            "dp_diff_mean": float(diff_dp.mean()),
            "dp_wins_dro": n_nonzero_dp,
            "dp_ties": n_zero_dp,
            "dp_pvalue": float(p_dp),
            "dp_sig": "*" if p_dp < 0.05 else "",
            "if_naive_mean": float(merged["if_clean_naive"].mean()),
            "if_dro_mean": float(merged["if_clean_dro"].mean()),
            "if_diff_mean": float(diff_if.mean()),
            "if_wins_dro": n_nonzero_if,
            "if_ties": n_zero_if,
            "if_pvalue": float(p_if),
            "if_sig": "*" if p_if < 0.05 else "",
        })
    return pd.DataFrame(out).sort_values(["dataset", "attack", "alpha"])


def write_md(wilc: pd.DataFrame, source: str, n_note: str):
    with open(OUT_MD, "w") as f:
        f.write("# Wilcoxon signed-rank tests (one-sided) — canonical_tau1 data\n\n")
        f.write(f"Source: {source}\n")
        f.write(f"{n_note}\n\n")
        f.write("H_a: Naive_DP > DRO_DP  (i.e., DRO yields strictly lower DP violation)\n")
        f.write("Paired by seed. * marks p<0.05.\n\n")
        f.write("Columns: n_seeds, means, diff=naive-dro (positive good for DRO), ")
        f.write("wins_dro = #seeds DRO strictly better, p, sig.\n\n")
        if wilc.empty:
            f.write("No data.\n")
            return
        f.write("| dataset | attack | α | n | DP_naive | DP_dro | ΔDP | wins | p | sig | IF_Δ | IF_p | IF_sig |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in wilc.iterrows():
            f.write(f"| {r['dataset']} | {r['attack']} | {r['alpha']:.1f} | {int(r['n_seeds'])} "
                    f"| {r['dp_naive_mean']:.4f} | {r['dp_dro_mean']:.4f} | {r['dp_diff_mean']:+.4f} "
                    f"| {int(r['dp_wins_dro'])}/{int(r['n_seeds'])} | {r['dp_pvalue']:.4f} | {r['dp_sig']} "
                    f"| {r['if_diff_mean']:+.4f} | {r['if_pvalue']:.4f} | {r['if_sig']} |\n")
    print(f"  saved {OUT_MD}")


def main():
    print("AGENT C: computing Wilcoxon (n=6 target) for canonical_tau1 / fallback")
    print("=" * 72)

    rows, source = load_preferred()
    print(f"Loaded: {source}")
    print(f"  row count: {len(rows)}")

    wilc = compute_wilcoxon(rows)
    print(f"Computed {len(wilc)} (dataset,attack,alpha) test rows")

    # Evidence for Adult DP (headline)
    print("\nEvidence — Adult DP attack (H1: Naive > DRO):")
    adult_dp = wilc[(wilc["dataset"] == "adult") & (wilc["attack"] == "dp")]
    for _, r in adult_dp.iterrows():
        marker = " *" if r["dp_sig"] else ""
        print(f"  α={r['alpha']:.1f} n={int(r['n_seeds'])}: ΔDP={r['dp_diff_mean']:+.5f}  "
              f"wins={int(r['dp_wins_dro'])}/{int(r['n_seeds'])}  p={r['dp_pvalue']:.4f}{marker}")

    if not wilc.empty:
        wilc.to_csv(OUT_CSV, index=False, float_format="%.6f")
        print(f"\n  saved {OUT_CSV}")
    else:
        print("  no wilc rows; csv not written")

    n_note = ("n=6 (canonical) — p<0.05 achievable for consistent effects. "
              "Previously n=3 limited min p~0.125." if "canonical" in source.lower()
              else "PRELIMINARY: n<=3 (tau_ablation fallback); min attainable p=0.125. "
                   "Regenerate after canonical_tau1.json (6 seeds) lands.")
    write_md(wilc, source, n_note)

    sig_count = int((wilc["dp_sig"] == "*").sum()) if not wilc.empty else 0
    print(f"\nAGENT C MILESTONE: canonical_wilcoxon complete (from {source}), {sig_count} significant DP cells (p<0.05). See results/canonical_wilcoxon.csv")
    print(f"  row counts used: {len(rows)}; will auto-upgrade on canonical presence.")
    print("=" * 72)


if __name__ == "__main__":
    main()
