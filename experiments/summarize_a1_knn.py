#!/usr/bin/env python3
"""Agent A1 — kNN ablation summary (analysis-only).

Reads results/knn_ablation.json (attack='if', attack_k in {5,15}) and
results/canonical_tau1.json (the IF-attack rows are the implicit k=5 reference:
run_fairness_pgd.run_single_experiment has attack_k=5 default, and the canonical
grid passes no attack_k, so every canonical IF row is k=5). The driver's
missing_configs() therefore SKIPS attack_k=5 rows that overlap canonical — the
k=15 rows are the genuinely NEW ones. This script verifies that overlap and
reports k=5 vs k=15 head-to-head.

Per (dataset, alpha): DP, IF, accuracy at attack_k=5 and attack_k=15 (and
attack_k=10 if present in the ablation file). Paired Wilcoxon (one-sided) on
IF-violation and on DP, H1: k15 > k5 (i.e. larger neighbourhood raises the
attack's measured violation). Honest either way.

Outputs:
    results/knn_ablation_summary.md

Run:
    PYTHONPATH=. python3 experiments/summarize_a1_knn.py
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

ABLATION_PATH = os.path.join(RESULTS_DIR, "knn_ablation.json")
OUT_MD = os.path.join(RESULTS_DIR, "knn_ablation_summary.md")

ALPHA_LEVEL = 0.05
ATTACK = "if"
DEFAULT_ATTACK_K = 5  # run_fairness_pgd.run_single_experiment default; canonical uses this

# Canonical IF-attack reference (k=5 implicit). Pulled via loader.
CANON_IF_K = DEFAULT_ATTACK_K


def _load_ablation():
    """Return (rows, source_label). Missing file => ([], 'MISSING')."""
    if not os.path.exists(ABLATION_PATH):
        return [], f"MISSING: {ABLATION_PATH}"
    with open(ABLATION_PATH) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{ABLATION_PATH} is not a JSON list")
    return rows, f"results/knn_ablation.json ({len(rows)} rows)"


def _load_canonical_if():
    """Canonical IF-attack rows = the implicit attack_k=5 reference."""
    from experiments.loaders import load_canonical_tau1
    rows = load_canonical_tau1()
    return [r for r in rows if r.get("attack") == ATTACK]


def _attack_k_of(r):
    """attack_k field; canonical rows lack the field and are treated as k=5.

    Accepts a dict (row) OR a scalar (pandas .apply on a Series passes scalars).
    """
    if isinstance(r, dict):
        v = r.get("attack_k", None)
    else:
        v = r
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return CANON_IF_K
    return int(v)


def _expected_total():
    """3 ds x 5 alpha x 6 seeds x 2 methods x 2 attack_k = 360."""
    return 3 * 5 * 6 * 2 * 2


def _wilcoxon_greater(d):
    """One-sided Wilcoxon H1: d > 0. Returns 1.0 if degenerate."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0.0):
        return 1.0
    try:
        _, p = wilcoxon(d, alternative="greater", zero_method="wilcox")
        return float(p)
    except ValueError:
        return 1.0


def _coverage(rows):
    """Per (dataset, alpha, attack_k) seed counts for naive/dro."""
    cov = defaultdict(lambda: defaultdict(set))
    for r in rows:
        cov[(r["dataset"], float(r["alpha"]), _attack_k_of(r))][r["method"]].add(r["seed"])
    return cov


def _mean(rows, key):
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def _per_cell_means(rows):
    """Per (dataset, alpha, attack_k, method): mean DP, IF, acc."""
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], float(r["alpha"]), _attack_k_of(r), r["method"])].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = {
            "dp": _mean(rs, "dp_clean"),
            "if": _mean(rs, "if_clean"),
            "acc": _mean(rs, "acc_clean"),
            "n": len(rs),
        }
    return out


def _paired_wilcoxon_k5_vs_k15(rows):
    """Seed-paired Wilcoxon on (k=15 - k=5) per (dataset, alpha, method).

    H1: k15 > k5  (larger neighbourhood raises the metric — stronger attack
    on IF, more DP violation). Reports p for both IF and DP.
    """
    df = pd.DataFrame(rows)
    if df.empty or "dataset" not in df.columns:
        return pd.DataFrame()
    # group by (ds, alpha, method, attack_k) -> seed -> metrics
    out = []
    for (ds, alpha, method), g in df.groupby(["dataset", "alpha", "method"], sort=True):
        g5 = g[g["attack_k"].apply(_attack_k_of) == 5].set_index("seed")
        g15 = g[g["attack_k"].apply(_attack_k_of) == 15].set_index("seed")
        if g5.empty or g15.empty:
            continue
        merged = g5[["dp_clean", "if_clean"]].join(
            g15[["dp_clean", "if_clean"]], lsuffix="_k5", rsuffix="_k15", how="inner"
        )
        n = len(merged)
        if n < 2:
            continue
        diff_if = merged["if_clean_k15"] - merged["if_clean_k5"]
        diff_dp = merged["dp_clean_k15"] - merged["dp_clean_k5"]
        p_if = _wilcoxon_greater(diff_if)
        p_dp = _wilcoxon_greater(diff_dp)
        out.append({
            "dataset": ds,
            "alpha": float(alpha),
            "method": method,
            "n_pairs": int(n),
            "if_k5_mean": float(merged["if_clean_k5"].mean()),
            "if_k15_mean": float(merged["if_clean_k15"].mean()),
            "if_diff_mean_k15_minus_k5": float(diff_if.mean()),
            "if_p_value": float(p_if),
            "if_wins_k15": int((diff_if > 0).sum()),
            "dp_k5_mean": float(merged["dp_clean_k5"].mean()),
            "dp_k15_mean": float(merged["dp_clean_k15"].mean()),
            "dp_diff_mean_k15_minus_k5": float(diff_dp.mean()),
            "dp_p_value": float(p_dp),
            "dp_wins_k15": int((diff_dp > 0).sum()),
        })
    return pd.DataFrame(out).sort_values(["dataset", "alpha", "method"]).reset_index(drop=True)


def _verify_canonical_overlap(canonical_if_rows):
    """Return dict reporting canonical attack_k status."""
    has_field = any("attack_k" in r for r in canonical_if_rows)
    n = len(canonical_if_rows)
    # If field is absent, every canonical IF row is implicit k=5
    if not has_field:
        return {
            "has_attack_k_field": False,
            "n_canonical_if_rows": n,
            "implicit_attack_k": CANON_IF_K,
            "note": (
                "canonical_tau1.json has NO attack_k field on IF rows; "
                f"per run_fairness_pgd.run_single_experiment default attack_k={CANON_IF_K}, "
                "the canonical IF-attack grid IS the k=5 reference. The A1 driver's "
                "missing_configs() should skip attack_k=5 rows that overlap canonical."
            ),
        }
    aks = sorted({_attack_k_of(r) for r in canonical_if_rows})
    return {
        "has_attack_k_field": True,
        "n_canonical_if_rows": n,
        "attack_k_values_present": aks,
        "note": "canonical_tau1.json DOES carry attack_k; overlap check proceeds by value.",
    }


def write_md(rows, source, canonical_info, wilc, means, coverage):
    complete_cells = sum(
        1 for k, m in coverage.items()
        if len(m.get("naive", set())) >= 6 and len(m.get("dro", set())) >= 6
    )
    expected_cells = 3 * 5 * 2  # 3 ds x 5 alpha x 2 attack_k = 30 (dataset,alpha,k) cells
    n_expected = _expected_total()

    lines = []
    lines.append("# Agent A1 — kNN ablation summary (attack_k ∈ {5,15}, IF attack)")
    lines.append("")
    lines.append("Analysis-only. No new training. Source: `results/knn_ablation.json` ")
    lines.append(f"({len(rows)}/{n_expected} rows) + canonical IF-attack rows as the k=5 reference.")
    lines.append("")
    lines.append("## Canonical attack_k question (resolved)")
    lines.append("")
    ci = canonical_info
    lines.append(f"- `canonical_tau1.json` has an `attack_k` field on IF rows: **{ci.get('has_attack_k_field')}**")
    if ci.get("has_attack_k_field") and "attack_k_values_present" in ci:
        lines.append(f"- attack_k values present on canonical IF rows: {ci['attack_k_values_present']}")
    else:
        lines.append(
            f"- implicit canonical attack_k = **{ci.get('implicit_attack_k', CANON_IF_K)}** "
            f"(per `run_fairness_pgd.run_single_experiment` default)"
        )
    lines.append(f"- {ci.get('note', '')}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Ablation rows present: **{len(rows)}/{n_expected}** "
                 f"({100.0*len(rows)/n_expected:.1f}%)" if n_expected else "- no rows")
    lines.append(f"- Complete (dataset,α,attack_k) cells with n≥6 both methods: "
                 f"**{complete_cells}/{expected_cells}**")
    if len(rows) < n_expected:
        lines.append(f"- **INCOMPLETE** — table below reflects partial data; "
                     f"re-run as more rows land (idempotent).")
    lines.append("")
    lines.append("## Per-cell means (DP / IF / acc)")
    lines.append("")
    lines.append("| dataset | α | k | method | n | DP | IF | acc |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (ds, alpha, ak, method), m in sorted(means.items()):
        lines.append(
            f"| {ds} | {alpha:.1f} | {ak} | {method} | {m['n']} "
            f"| {m['dp']:.4f} | {m['if']:.4f} | {m['acc']:.4f} |"
        )
    lines.append("")
    lines.append("## Paired Wilcoxon: k=15 vs k=5 (seed-paired, one-sided H1: k15 > k5)")
    lines.append("")
    lines.append("Positive ΔIF / ΔDP ⇒ larger k raises the metric (stronger attack). "
                 f"* marks p<{ALPHA_LEVEL}.")
    lines.append("")
    if wilc.empty:
        lines.append("_(no paired cells yet — need both k=5 and k=15 rows for the same seed)_")
    else:
        lines.append("| dataset | α | method | n | IF_k5 | IF_k15 | ΔIF | wins_k15 | p_IF | | DP_k5 | DP_k15 | ΔDP | wins_k15 | p_DP |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in wilc.iterrows():
            sig_if = "*" if r["if_p_value"] < ALPHA_LEVEL else ""
            sig_dp = "*" if r["dp_p_value"] < ALPHA_LEVEL else ""
            lines.append(
                f"| {r['dataset']} | {r['alpha']:.1f} | {r['method']} | {int(r['n_pairs'])} "
                f"| {r['if_k5_mean']:.4f} | {r['if_k15_mean']:.4f} "
                f"| {r['if_diff_mean_k15_minus_k5']:+.4f} "
                f"| {int(r['if_wins_k15'])}/{int(r['n_pairs'])} "
                f"| {r['if_p_value']:.4f} {sig_if} | "
                f"| {r['dp_k5_mean']:.4f} | {r['dp_k15_mean']:.4f} "
                f"| {r['dp_diff_mean_k15_minus_k5']:+.4f} "
                f"| {int(r['dp_wins_k15'])}/{int(r['n_pairs'])} "
                f"| {r['dp_p_value']:.4f} {sig_dp} |"
            )
    lines.append("")

    # One-sentence answer.
    if wilc.empty:
        answer = ("Not yet answerable — need seed-paired k=5 and k=15 rows in "
                  "results/knn_ablation.json (currently INCOMPLETE).")
    else:
        sig_if = (wilc["if_p_value"] < ALPHA_LEVEL).sum()
        sig_dp = (wilc["dp_p_value"] < ALPHA_LEVEL).sum()
        n_cells = len(wilc)
        mean_dif = wilc["if_diff_mean_k15_minus_k5"].mean()
        mean_ddp = wilc["dp_diff_mean_k15_minus_k5"].mean()
        if sig_if == 0 and sig_dp == 0 and abs(mean_dif) < 1e-4 and abs(mean_ddp) < 1e-4:
            answer = ("No: the IF attack's measured strength does NOT depend on k "
                      f"within {{5,15}} — neither IF-violation nor DP moves significantly "
                      f"with the neighbourhood (mean ΔIF={mean_dif:+.4f}, mean ΔDP={mean_ddp:+.4f}, "
                      f"0/{n_cells} cells p<0.05).")
        else:
            direction_if = "rises" if mean_dif > 0 else "falls"
            direction_dp = "rises" if mean_ddp > 0 else "falls"
            answer = (
                f"Partial: across {n_cells} cells, larger k {direction_if} IF-violation "
                f"(mean ΔIF={mean_dif:+.4f}, {sig_if}/{n_cells} p<0.05) and {direction_dp} "
                f"DP (mean ΔDP={mean_ddp:+.4f}, {sig_dp}/{n_cells} p<0.05) — "
                + ("the attack's strength DOES depend on k."
                   if (sig_if + sig_dp) > 0 else
                   "the effect is small and not significant; treat the attack as k-robust.")
            )
    lines.append("## One-sentence answer — does the IF attack's strength depend on k?")
    lines.append("")
    lines.append(answer)
    lines.append("")
    return "\n".join(lines), answer


def main():
    print("AGENT A1: kNN ablation summary (analysis-only)")
    print("=" * 78)

    rows, source = _load_ablation()
    n_expected = _expected_total()
    print(f"Loaded: {source}")
    if len(rows) < n_expected:
        print(f"  INCOMPLETE: {len(rows)}/{n_expected} rows "
              f"({100.0*len(rows)/n_expected:.1f}%) — partial-data mode.")

    # Canonical IF-attack rows (k=5 reference).
    try:
        canonical_if = _load_canonical_if()
        canonical_info = _verify_canonical_overlap(canonical_if)
        print(f"Canonical IF-attack rows: {len(canonical_if)} "
              f"(has attack_k field: {canonical_info['has_attack_k_field']})")
    except Exception as e:
        canonical_if = []
        canonical_info = {"has_attack_k_field": "UNKNOWN", "n_canonical_if_rows": 0,
                          "note": f"canonical load failed: {e}"}
        print(f"  WARN: canonical load failed: {e}")

    # Filter ablation to attack=='if' (defensive — file should only hold IF rows).
    if_rows = [r for r in rows if r.get("attack") == ATTACK]
    if rows and not if_rows:
        print(f"  WARN: 0 IF rows in {source} (all rows attack="
              f"{sorted(set(r.get('attack') for r in rows))})")

    coverage = _coverage(if_rows)
    means = _per_cell_means(if_rows)
    wilc = _paired_wilcoxon_k5_vs_k15(if_rows)
    print(f"Computed {len(wilc)} seed-paired (k=5 vs k=15) Wilcoxon rows.")

    md_text, answer = write_md(rows, source, canonical_info, wilc, means, coverage)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print(f"  saved {OUT_MD}")

    print("\nOne-sentence answer:")
    print("  " + answer)

    if not wilc.empty:
        print("\nPer-cell (k=15 vs k=5) Wilcoxon:")
        print(f"{'dataset':<8} {'α':<5} {'method':<6} {'n':<3} "
              f"{'ΔIF':<10} {'p_IF':<8} {'ΔDP':<10} {'p_DP':<8}")
        for _, r in wilc.iterrows():
            print(f"{r['dataset']:<8} {r['alpha']:<5.1f} {r['method']:<6} "
                  f"{int(r['n_pairs']):<3d} "
                  f"{r['if_diff_mean_k15_minus_k5']:<+10.4f} {r['if_p_value']:<8.4f} "
                  f"{r['dp_diff_mean_k15_minus_k5']:<+10.4f} {r['dp_p_value']:<8.4f}")

    print("\n" + "=" * 78)
    print("AGENT A1 SUMMARY MILESTONE complete (analysis-only).")
    print(f"  output: {OUT_MD}")
    print("=" * 78)


if __name__ == "__main__":
    main()