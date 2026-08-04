#!/usr/bin/env python3
"""Agent N4 (MASTER_PROTOCOL_AUG10.md, Part 3, Wave 1.5, deliverable D4).

ANALYSIS-ONLY. No new training. Reads results/canonical_tau1.json only.

Kuldeep (verbatim, Jun 30): "if individual fairness is good for α=0.3, then we
can state this clearly." This script tests that claim formally.

What it does
------------
1. Loads results/canonical_tau1.json via experiments.loaders.load_canonical_tau1()
   (fail-loud, no fallback to stale/tau_ablation files).
2. Filters to attack == 'if' rows only (the IF-attack grid: 180 rows =
   3 datasets × 5 alpha × 6 seeds × 2 methods).
3. For each (dataset, alpha) cell, pairs naive vs dro BY SEED and computes:
     - mean IF violation (if_clean) for naive and dro
     - paired Wilcoxon one-sided, H1: naive_if > dro_if
       (i.e. DRO has STRICTLY LOWER IF violation — better individual fairness)
     - p-value, n_pairs, win count (seeds where dro_if < naive_if)
4. Verifies the protocol-stated means:
     Adult  α=0.3: DRO 0.0258 vs Naive 0.0334
     Credit α=0.3: DRO 0.1011 vs Naive 0.1212
5. Writes results/if_violation_wilcoxon.csv with columns:
     dataset, alpha, n_pairs, if_naive_mean, if_dro_mean, if_diff_mean,
     p_value, win_count_dro
6. Coupling caveat check (D4 verbatim): the protocol notes
   "Adult α=0.3 still DP loss under IF (coupling)". We verify this directly
   by ALSO computing the DP metric (dp_clean) under IF attack at α=0.3 for
   Adult: is dro_dp > naive_dp there? (DRO losing on DP).
7. Prints a readable summary to stdout AND writes
   results/if_wilcoxon_n4_summary.md.

Outputs (only two new files; canonical_tau1.json untouched):
    results/if_violation_wilcoxon.csv
    results/if_wilcoxon_n4_summary.md

Run:
    PYTHONPATH=. python3 experiments/agent_n4_if_wilcoxon.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

OUT_CSV = os.path.join(RESULTS_DIR, "if_violation_wilcoxon.csv")
OUT_MD = os.path.join(RESULTS_DIR, "if_wilcoxon_n4_summary.md")

# Protocol-stated means (D4) for verification.
EXPECTED = {
    ("adult", 0.3): {"dro": 0.0258, "naive": 0.0334},
    ("credit", 0.3): {"dro": 0.1011, "naive": 0.1212},
}
# Tolerance: means can differ slightly by float aggregation; allow 5e-3.
TOL = 5e-3

ALPHA_LEVEL = 0.05


def load_if_rows() -> tuple[list[dict], str]:
    """Load canonical_tau1.json (fail-loud) and filter to attack=='if' rows."""
    from experiments.loaders import load_canonical_tau1
    rows = load_canonical_tau1()
    if_rows = [r for r in rows if r.get("attack") == "if"]
    if not if_rows:
        raise RuntimeError(
            "No attack=='if' rows in canonical_tau1.json. "
            "IF grid should be 180 rows (3 ds × 5 α × 6 seeds × 2 methods)."
        )
    # Sanity: required fields present.
    required = {"dataset", "alpha", "seed", "method", "if_clean", "dp_clean"}
    for r in if_rows:
        missing = required - set(r.keys())
        if missing:
            raise RuntimeError(
                f"IF row missing required fields {missing}: {r}"
            )
    return if_rows, f"canonical_tau1.json ({len(rows)} total, {len(if_rows)} IF rows)"


def compute_if_wilcoxon(if_rows: list[dict]) -> pd.DataFrame:
    """Per (dataset, alpha) paired Wilcoxon on IF metric, H1: naive_if > dro_if."""
    df = pd.DataFrame(if_rows)
    out = []
    for (ds, alpha), g in df.groupby(["dataset", "alpha"], sort=True):
        naive = g[g["method"] == "naive"][["seed", "if_clean", "dp_clean"]]
        dro = g[g["method"] == "dro"][["seed", "if_clean", "dp_clean"]]
        merged = naive.merge(dro, on="seed", suffixes=("_naive", "_dro"))
        n = len(merged)
        if n < 2:
            continue

        diff_if = merged["if_clean_naive"] - merged["if_clean_dro"]
        # one-sided: H1 naive_if > dro_if  =>  diff_if > 0  => alternative="greater"
        try:
            _, p_if = wilcoxon(diff_if, alternative="greater", zero_method="wilcox")
        except ValueError:
            # All diffs zero or insufficient non-zero — not significant.
            p_if = 1.0

        out.append({
            "dataset": ds,
            "alpha": float(alpha),
            "n_pairs": int(n),
            "if_naive_mean": float(merged["if_clean_naive"].mean()),
            "if_dro_mean": float(merged["if_clean_dro"].mean()),
            "if_diff_mean": float(diff_if.mean()),
            "p_value": float(p_if),
            "win_count_dro": int((diff_if > 0).sum()),
        })
    return pd.DataFrame(out).sort_values(["dataset", "alpha"]).reset_index(drop=True)


def compute_adult_dp_coupling(if_rows: list[dict]) -> dict:
    """Adult α=0.3 DP metric under IF attack: is dro_dp > naive_dp? (DRO losing DP)"""
    df = pd.DataFrame(if_rows)
    sub = df[(df["dataset"] == "adult") & (df["alpha"] == 0.3)]
    naive = sub[sub["method"] == "naive"][["seed", "dp_clean"]]
    dro = sub[sub["method"] == "dro"][["seed", "dp_clean"]]
    merged = naive.merge(dro, on="seed", suffixes=("_naive", "_dro"))
    diff_dp = merged["dp_clean_naive"] - merged["dp_clean_dro"]  # <0 => DRO worse on DP
    try:
        _, p_dp = wilcoxon(diff_dp, alternative="greater", zero_method="wilcox")
    except ValueError:
        p_dp = 1.0
    return {
        "n_pairs": int(len(merged)),
        "dp_naive_mean": float(merged["dp_clean_naive"].mean()),
        "dp_dro_mean": float(merged["dp_clean_dro"].mean()),
        "dp_diff_mean_naive_minus_dro": float(diff_dp.mean()),
        "p_value_h1_naive_gt_dro": float(p_dp),
        "dro_dp_wins": int((diff_dp > 0).sum()),  # seeds DRO strictly better on DP
        "dp_loss_for_dro": bool(merged["dp_clean_dro"].mean() > merged["dp_clean_naive"].mean()),
    }


def verify_expected(wilc: pd.DataFrame) -> list[str]:
    """Cross-check the protocol-stated means against the computed ones."""
    notes = []
    for (ds, alpha), exp in EXPECTED.items():
        row = wilc[(wilc["dataset"] == ds) & (wilc["alpha"] == alpha)]
        if row.empty:
            notes.append(f"  [FAIL] ({ds}, {alpha}): no computed row.")
            continue
        got_dro = float(row["if_dro_mean"].iloc[0])
        got_naive = float(row["if_naive_mean"].iloc[0])
        ok_dro = abs(got_dro - exp["dro"]) <= TOL
        ok_naive = abs(got_naive - exp["naive"]) <= TOL
        flag = "OK" if (ok_dro and ok_naive) else "MISMATCH"
        notes.append(
            f"  [{flag}] {ds} α={alpha}: "
            f"DRO got={got_dro:.4f} expected={exp['dro']:.4f} | "
            f"Naive got={got_naive:.4f} expected={exp['naive']:.4f}"
        )
    return notes


def write_md(wilc: pd.DataFrame, source: str, coupling: dict,
             verify_notes: list[str]) -> tuple[str, str]:
    """Write results/if_wilcoxon_n4_summary.md. Return (headline_p05, alpha03_stmt)."""
    sig_cells = wilc[wilc["p_value"] < ALPHA_LEVEL]
    headline_lines = []
    for _, r in sig_cells.iterrows():
        headline_lines.append(
            f"- **{r['dataset']} α={r['alpha']:.1f}** — "
            f"IF naive={r['if_naive_mean']:.4f} vs dro={r['if_dro_mean']:.4f}, "
            f"n={int(r['n_pairs'])}, wins_dro={int(r['win_count_dro'])}/{int(r['n_pairs'])}, "
            f"p={r['p_value']:.4f} (<0.05)"
        )
    headline_p05 = "\n".join(headline_lines) if headline_lines else "_(none)_"

    # α=0.3 explicit statement (Adult + Credit).
    alpha03_rows = wilc[wilc["alpha"] == 0.3].set_index("dataset")
    a_adult = alpha03_rows.loc["adult"]
    a_credit = alpha03_rows.loc["credit"]
    adult_sig = a_adult["p_value"] < ALPHA_LEVEL
    credit_sig = a_credit["p_value"] < ALPHA_LEVEL
    alpha03_stmt = (
        "Kuldeep claim — \"if individual fairness is good for α=0.3, "
        "then we can state this clearly\":\n"
        f"- Adult α=0.3 (IF attack, IF metric): DRO wins "
        f"{int(a_adult['win_count_dro'])}/{int(a_adult['n_pairs'])} seeds, "
        f"dro_if={a_adult['if_dro_mean']:.4f} < naive_if={a_adult['if_naive_mean']:.4f}, "
        f"p={a_adult['p_value']:.4f} → "
        f"{'SIGNIFICANT (DRO IF strictly lower).' if adult_sig else 'NOT significant.'}\n"
        f"- Credit α=0.3 (IF attack, IF metric): DRO wins "
        f"{int(a_credit['win_count_dro'])}/{int(a_credit['n_pairs'])} seeds, "
        f"dro_if={a_credit['if_dro_mean']:.4f} < naive_if={a_credit['if_naive_mean']:.4f}, "
        f"p={a_credit['p_value']:.4f} → "
        f"{'SIGNIFICANT (DRO IF strictly lower).' if credit_sig else 'NOT significant.'}\n"
        f"- Kuldeep claim SUPPORTED on Adult and Credit at α=0.3: "
        f"{'YES' if (adult_sig and credit_sig) else 'PARTIAL' if (adult_sig or credit_sig) else 'NO'}."
    )

    # Coupling caveat — Adult α=0.3 DP under IF attack.
    dp_loss = coupling["dp_loss_for_dro"]
    coup_stmt = (
        "Coupling caveat (D4): Adult α=0.3 under IF attack — DP metric under IF attack.\n"
        f"- dp_naive_mean={coupling['dp_naive_mean']:.4f}, "
        f"dp_dro_mean={coupling['dp_dro_mean']:.4f}, "
        f"Δ(naive-dro)={coupling['dp_diff_mean_naive_minus_dro']:+.4f}, "
        f"n={coupling['n_pairs']}, "
        f"dro_dp_wins={coupling['dro_dp_wins']}/{coupling['n_pairs']}, "
        f"p(H1 naive>dro)={coupling['p_value_h1_naive_gt_dro']:.4f}\n"
        f"- DP loss for DRO at Adult α=0.3 under IF attack: "
        f"{'YES — dro_dp > naive_dp (DRO loses on DP), confirming the IF↔DP coupling caveat.' if dp_loss else 'NO — DRO also wins DP at this cell.'}"
    )

    with open(OUT_MD, "w") as f:
        f.write("# Agent N4 — IF-violation paired Wilcoxon under IF attack (D4)\n\n")
        f.write("Analysis-only. No new training. Source of truth: ")
        f.write(f"`results/canonical_tau1.json` (IF-attack rows only).\n\n")
        f.write(f"Source: {source}\n\n")
        f.write("Test: per (dataset, α), paired Wilcoxon one-sided on the ")
        f.write("**IF-violation** metric itself (`if_clean`), H1: ")
        f.write("`naive_if > dro_if` (DRO has **strictly lower** IF violation = ")
        f.write("better individual fairness). Paired by seed. ")
        f.write(f"Significance level α={ALPHA_LEVEL}.\n\n")

        f.write("## Protocol-mean verification\n\n")
        f.write("Expected (D4): Adult α=0.3 DRO 0.0258 vs Naive 0.0334; ")
        f.write("Credit 0.1011 vs 0.1212.\n\n")
        for n in verify_notes:
            f.write(f"{n}\n")
        f.write("\n")

        f.write("## All (dataset, α) cells — IF violation Wilcoxon\n\n")
        f.write("| dataset | α | n_pairs | IF_naive | IF_dro | ΔIF(naive-dro) | wins_dro | p_value | sig |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in wilc.iterrows():
            sig = "*" if r["p_value"] < ALPHA_LEVEL else ""
            f.write(
                f"| {r['dataset']} | {r['alpha']:.1f} | {int(r['n_pairs'])} "
                f"| {r['if_naive_mean']:.4f} | {r['if_dro_mean']:.4f} "
                f"| {r['if_diff_mean']:+.4f} | {int(r['win_count_dro'])}/{int(r['n_pairs'])} "
                f"| {r['p_value']:.4f} | {sig} |\n"
            )
        f.write("\n* marks p<0.05 (DRO IF violation significantly lower than naive).\n\n")

        f.write("## Headline — cells where DRO is significantly lower on IF (p<0.05)\n\n")
        f.write(headline_p05 + "\n\n")

        f.write("## Kuldeep α=0.3 claim — explicit\n\n")
        f.write(alpha03_stmt + "\n\n")

        f.write("## Coupling caveat — Adult α=0.3 DP under IF attack\n\n")
        f.write("Protocol (D4): \"Adult α=0.3 still DP loss under IF (coupling)\". ")
        f.write("We verify directly: under IF attack at Adult α=0.3, is ")
        f.write("`dro_dp > naive_dp`?\n\n")
        f.write(coup_stmt + "\n\n")

        f.write("## Interpretation\n\n")
        if adult_sig and credit_sig and dp_loss:
            f.write(
                "DRO delivers significantly lower IF violation at α=0.3 on both "
                "Adult and Credit (Kuldeep claim supported). However, on Adult at "
                "α=0.3 the DP metric under the same IF attack is WORSE for DRO "
                "(dro_dp > naive_dp): the IF gain is coupled with a DP loss. "
                "State the IF result clearly, but pair it with this DP-coupling "
                "caveat in the paper's IF section.\n"
            )
        elif adult_sig and credit_sig and not dp_loss:
            f.write(
                "DRO delivers significantly lower IF violation at α=0.3 on both "
                "Adult and Credit, and the DP-coupling caveat is NOT reproduced "
                "in this data (DRO does not lose DP at Adult α=0.3). State the IF "
                "result clearly and note the coupling caveat does not hold here.\n"
            )
        else:
            f.write(
                "DRO does not reach significance on IF at α=0.3 on both datasets. "
                "State the result as directional only where p is small but ≥0.05.\n"
            )

    return headline_p05, alpha03_stmt + "\n\n" + coup_stmt


def main():
    print("AGENT N4: IF-violation paired Wilcoxon under IF attack (D4, analysis-only)")
    print("=" * 78)

    if_rows, source = load_if_rows()
    print(f"Loaded: {source}")
    # quick coverage print
    cov = {}
    for r in if_rows:
        cov.setdefault((r["dataset"], r["alpha"]), set()).add(r["seed"])
    ds_alpha = sorted(cov.keys())
    print(f"  (dataset, α) cells: {len(ds_alpha)}; "
          f"seeds per cell: {sorted(len(s) for s in cov.values())}")

    wilc = compute_if_wilcoxon(if_rows)
    print(f"Computed {len(wilc)} (dataset, α) IF-Wilcoxon rows")

    # Verify protocol-stated means.
    print("\nProtocol-mean verification (Adult/Credit α=0.3):")
    verify_notes = verify_expected(wilc)
    for n in verify_notes:
        print(n)

    # Coupling check.
    coupling = compute_adult_dp_coupling(if_rows)
    print("\nAdult α=0.3 DP under IF attack (coupling caveat):")
    print(f"  dp_naive_mean={coupling['dp_naive_mean']:.4f} "
          f"dp_dro_mean={coupling['dp_dro_mean']:.4f} "
          f"Δ(naive-dro)={coupling['dp_diff_mean_naive_minus_dro']:+.4f} "
          f"dro_dp_wins={coupling['dro_dp_wins']}/{coupling['n_pairs']} "
          f"p={coupling['p_value_h1_naive_gt_dro']:.4f}")
    print(f"  DP loss for DRO at Adult α=0.3 under IF: "
          f"{coupling['dp_loss_for_dro']}")

    # Write CSV.
    wilc.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"\n  saved {OUT_CSV}")

    # Write MD + collect headline / α=0.3 / coupling strings for stdout.
    headline_p05, alpha03_stmt = write_md(wilc, source, coupling, verify_notes)
    print(f"  saved {OUT_MD}")

    # Readable stdout summary.
    print("\nIF-violation Wilcoxon per cell (H1: naive_if > dro_if):")
    print("-" * 78)
    print(f"{'dataset':<8} {'α':<5} {'n':<3} {'IF_naive':<10} {'IF_dro':<10} "
          f"{'ΔIF':<10} {'wins':<8} {'p':<8} sig")
    for _, r in wilc.iterrows():
        sig = "*" if r["p_value"] < ALPHA_LEVEL else ""
        print(f"{r['dataset']:<8} {r['alpha']:<5.1f} {int(r['n_pairs']):<3d} "
              f"{r['if_naive_mean']:<10.4f} {r['if_dro_mean']:<10.4f} "
              f"{r['if_diff_mean']:<+10.4f} "
              f"{int(r['win_count_dro']):<3d}/{int(r['n_pairs']):<3d}   "
              f"{r['p_value']:<8.4f} {sig}")

    print("\nHeadline (p<0.05, DRO IF strictly lower):")
    print(headline_p05)

    print("\nα=0.3 + coupling statement:")
    print(alpha03_stmt)

    print("\n" + "=" * 78)
    print("AGENT N4 MILESTONE: IF-violation Wilcoxon complete (analysis-only).")
    print(f"  outputs: {OUT_CSV}")
    print(f"           {OUT_MD}")
    print("=" * 78)


if __name__ == "__main__":
    main()