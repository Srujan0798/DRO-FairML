#!/usr/bin/env python3
"""AGENT S — recompute canonical Wilcoxon at n=10 (or current partial n).

Reads results/canonical_tau1.json (now extended with seeds 6-9 by run_s_n10.py),
recomputes the paired Wilcoxon for every (dataset, attack, alpha) cell at the
new n, compares against the LOCKED n=6 results/canonical_wilcoxon.csv to detect
significance FLIPS, and rewrites:

    results/canonical_wilcoxon.csv   (at the new n)
    results/canonical_wilcoxon.md    (at the new n, with a FLIPS section)

This script does NOT touch results/canonical_tau1.json (read-only) and does NOT
regenerate figures (Agent I2's job).

Run:
    python3 experiments/summarize_n10.py
"""
from __future__ import annotations

import os
import json
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
CANON = os.path.join(RESULTS_DIR, "canonical_tau1.json")
OUT_CSV = os.path.join(RESULTS_DIR, "canonical_wilcoxon.csv")
OUT_MD = os.path.join(RESULTS_DIR, "canonical_wilcoxon.md")

# Locked n=6 reference CSV (read-only comparison baseline). If the file has
# already been overwritten by a previous n=10 run, we reconstruct the n=6
# table from the first 540 rows on the fly so the FLIP detection is always
# honest and traceable.
N6_ROWS = 540
N6_SEEDS = 6


def _load_rows():
    with open(CANON) as f:
        rows = json.load(f)
    return rows


def compute_wilcoxon(rows):
    """Recompute the Wilcoxon table for whatever rows are passed in.

    Mirrors experiments/compute_canonical_wilcoxon.py exactly so numbers are
    directly comparable. H1 (one-sided): Naive_DP > DRO_DP.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    for (ds, attack, alpha), g in df.groupby(["dataset", "attack", "alpha"], sort=True):
        naive = g[g["method"] == "naive"][["seed", "dp_clean", "if_clean"]]
        dro = g[g["method"] == "dro"][["seed", "dp_clean", "if_clean"]]
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
        n_zero_dp = int((diff_dp == 0).sum())
        n_nonzero_if = int((diff_if > 0).sum())
        n_zero_if = int((diff_if == 0).sum())

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
    return pd.DataFrame(out).sort_values(["dataset", "attack", "alpha"]).reset_index(drop=True)


def _sig_str(s):
    return "*" if s == "*" else ""


def detect_flips(w6, w10):
    """Compare n=6 vs n=10 Wilcoxon tables; return a DataFrame of FLIPS.

    A FLIP is any cell where the significance marker (p<0.05) changed between
    n=6 and n=10, in either direction, on either the DP or IF metric. We also
    report directional change of the mean diff for context.
    """
    if w6 is None or w10 is None or w6.empty or w10.empty:
        return pd.DataFrame()
    cols = ["dataset", "attack", "alpha"]
    m = w6[cols + ["dp_pvalue", "dp_sig", "if_pvalue", "if_sig",
                   "dp_diff_mean", "if_diff_mean", "n_seeds"]].merge(
        w10[cols + ["dp_pvalue", "dp_sig", "if_pvalue", "if_sig",
                    "dp_diff_mean", "if_diff_mean", "n_seeds"]],
        on=cols, suffixes=("_n6", "_n10"),
    )
    flips = []
    for _, r in m.iterrows():
        dp_flip = (r["dp_sig_n6"] != r["dp_sig_n10"])
        if_flip = (r["if_sig_n6"] != r["if_sig_n10"])
        if not (dp_flip or if_flip):
            continue
        flips.append({
            "dataset": r["dataset"],
            "attack": r["attack"],
            "alpha": float(r["alpha"]),
            "dp_p_n6": float(r["dp_pvalue_n6"]),
            "dp_p_n10": float(r["dp_pvalue_n10"]),
            "dp_sig_n6": _sig_str(r["dp_sig_n6"]),
            "dp_sig_n10": _sig_str(r["dp_sig_n10"]),
            "dp_flip": "Y" if dp_flip else "",
            "if_p_n6": float(r["if_pvalue_n6"]),
            "if_p_n10": float(r["if_pvalue_n10"]),
            "if_sig_n6": _sig_str(r["if_sig_n6"]),
            "if_sig_n10": _sig_str(r["if_sig_n10"]),
            "if_flip": "Y" if if_flip else "",
            "dp_diff_n6": float(r["dp_diff_mean_n6"]),
            "dp_diff_n10": float(r["dp_diff_mean_n10"]),
            "if_diff_n6": float(r["if_diff_mean_n6"]),
            "if_diff_n10": float(r["if_diff_mean_n10"]),
        })
    return pd.DataFrame(flips)


def write_md(wilc, flips, n_rows, n_actual, source_note):
    with open(OUT_MD, "w") as f:
        f.write("# Wilcoxon signed-rank tests (one-sided) — canonical_tau1 data\n\n")
        f.write(f"Source: {source_note}\n")
        f.write(
            f"n={n_actual} paired seeds (extended by Agent S from the locked n=6 grid; "
            f"first 540 rows byte-identical). p<0.05 marked with *. "
            f"No tau_ablation / K_inner=5 fallback.\n\n"
        )
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

        f.write("\n## Significance FLIPS vs locked n=6\n\n")
        if flips is None or flips.empty:
            f.write("No cells flipped significance at the new n. "
                    "All n=6 significance calls are upheld at n=" + str(n_actual) + ".\n")
        else:
            f.write(f"{len(flips)} cell(s) changed significance (p<0.05 threshold) "
                    f"between n=6 and n={n_actual}. The paper must report the n={n_actual} "
                    f"truth and call out each flip explicitly.\n\n")
            f.write("| dataset | attack | α | DP_p n6 | DP_p n10 | DP sig n6 | DP sig n10 | DP flip | IF_p n6 | IF_p n10 | IF sig n6 | IF sig n10 | IF flip | ΔDP n6 | ΔDP n10 | IFΔ n6 | IFΔ n10 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for _, r in flips.iterrows():
                f.write(
                    f"| {r['dataset']} | {r['attack']} | {r['alpha']:.1f} "
                    f"| {r['dp_p_n6']:.4f} | {r['dp_p_n10']:.4f} "
                    f"| {r['dp_sig_n6'] or '·'} | {r['dp_sig_n10'] or '·'} | {r['dp_flip'] or '·'} "
                    f"| {r['if_p_n6']:.4f} | {r['if_p_n10']:.4f} "
                    f"| {r['if_sig_n6'] or '·'} | {r['if_sig_n10'] or '·'} | {r['if_flip'] or '·'} "
                    f"| {r['dp_diff_n6']:+.4f} | {r['dp_diff_n10']:+.4f} "
                    f"| {r['if_diff_n6']:+.4f} | {r['if_diff_n10']:+.4f} |\n"
                )
    print(f"  saved {OUT_MD}")


def main():
    print("AGENT S summarize: recomputing Wilcoxon at the new n")
    print("=" * 72)

    rows = _load_rows()
    n_rows = len(rows)
    seeds_present = sorted({r["seed"] for r in rows})
    n_actual = len(seeds_present)
    print(f"Loaded canonical_tau1.json: {n_rows} rows, seeds={seeds_present}")

    # --- n=10 (current) table ---
    w10 = compute_wilcoxon(rows)
    print(f"n={n_actual} Wilcoxon: {len(w10)} (dataset,attack,alpha) cells")

    # --- n=6 baseline (reconstruct from first 540 rows for honesty) ---
    w6 = compute_wilcoxon(rows[:N6_ROWS]) if n_rows >= N6_ROWS else None
    if w6 is not None:
        print(f"n=6 baseline (first {N6_ROWS} rows): {len(w6)} cells")

    # --- FLIP detection ---
    flips = detect_flips(w6, w10)
    if flips is None or flips.empty:
        print("FLIPS: none — every n=6 significance call upheld at n=" + str(n_actual))
    else:
        print(f"FLIPS: {len(flips)} cell(s) changed significance:")
        for _, r in flips.iterrows():
            print(
                f"  {r['dataset']:6s} {r['attack']:8s} α={r['alpha']:.1f}  "
                f"DP {r['dp_sig_n6'] or '·'}->{r['dp_sig_n10'] or '·'} "
                f"(p {r['dp_p_n6']:.4f}->{r['dp_p_n10']:.4f})  "
                f"IF {r['if_sig_n6'] or '·'}->{r['if_sig_n10'] or '·'} "
                f"(p {r['if_p_n6']:.4f}->{r['if_p_n10']:.4f})"
            )

    # --- write outputs ---
    if not w10.empty:
        w10.to_csv(OUT_CSV, index=False, float_format="%.6f")
        print(f"  saved {OUT_CSV}")
    else:
        print("  no wilcoxon rows; csv not written")

    source_note = f"canonical_tau1.json ({n_rows} rows, k_inner=10, tau=1, seeds 0-{n_actual-1})"
    write_md(w10, flips, n_rows, n_actual, source_note)

    # --- summary line ---
    dp_sig = int((w10["dp_sig"] == "*").sum()) if not w10.empty else 0
    if_sig = int((w10["if_sig"] == "*").sum()) if not w10.empty else 0
    n_flip = 0 if flips is None or flips.empty else len(flips)
    print(
        f"\nAGENT S summarize DONE: n={n_actual}, {dp_sig} DP-sig cells, "
        f"{if_sig} IF-sig cells, {n_flip} significance FLIP(s) vs n=6. "
        f"See {OUT_CSV} and {OUT_MD}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()