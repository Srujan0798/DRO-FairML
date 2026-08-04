#!/usr/bin/env python3
"""Agent A2 — τ ablation summary (analysis-only).

Reads results/tau_ablation.json (τ ∈ {10,100}, attack='dp') and
results/canonical_tau1.json (the τ=1 reference via experiments.loaders).
Per (dataset, α): DP for naive and DRO at τ=1, τ=10, τ=100. Seed-paired
Wilcoxon DRO vs Naive at each τ (H1: naive_dp > dro_dp).

This is the motivating ablation: the τ=100 artifact — the flip from "DRO
loses at τ=100" to "DRO wins at τ=1" — is the central evidence that the
earlier "DRO is fragile" story was a high-temperature artifact, not a real
failure of DRO-FAIR.

The paper (results.tex Table~tab:tau-comparison) already carries τ=10/100
numbers from a HISTORICAL stepped-schedule pilot (n=3). This script flags
whether the new n=6 fixed-τ data CONFIRMS or CONTRADICTS those historical
numbers, per (α, τ) cell on Adult/DP.

Outputs:
    results/tau_ablation_summary.md

Run:
    PYTHONPATH=. python3 experiments/summarize_a2_tau.py
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

ABLATION_PATH = os.path.join(RESULTS_DIR, "tau_ablation.json")
OUT_MD = os.path.join(RESULTS_DIR, "tau_ablation_summary.md")

ALPHA_LEVEL = 0.05
ATTACK = "dp"
DATASETS = ["adult", "credit", "lsac"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
TAUS_NEW = [10.0, 100.0]
TAU_REF = 1.0

# Historical pilot numbers (n=3, stepped-schedule) lifted verbatim from
# paper/sections/results.tex Table~tab:tau-comparison (Adult, DP attack).
# (alpha, tau) -> (naive_dp, dro_dp, dro_wins_str, verdict)
HISTORICAL_PILOT = {
    (0.1, 1.0):   (0.2026, 0.1999, "5/6", "DRO"),
    (0.1, 10.0):  (0.1850, 0.2231, "0/3", "Tie"),
    (0.1, 100.0): (0.1801, 0.2033, "0/3", "Tie"),
    (0.2, 1.0):   (0.2452, 0.2334, "6/6", "DRO"),
    (0.2, 10.0):  (0.3382, 0.4634, "0/3", "Naive"),
    (0.2, 100.0): (0.3271, 0.5030, "0/3", "Naive"),
    (0.3, 1.0):   (0.2848, 0.2614, "6/6", "DRO"),
    (0.3, 10.0):  (0.5253, 0.5532, "0/3", "Naive"),
    (0.3, 100.0): (0.5313, 0.5622, "0/3", "Naive"),
    (0.4, 1.0):   (0.3140, 0.2855, "6/6", "DRO"),
    (0.4, 10.0):  (0.5158, 0.5231, "0/3", "Naive"),
    (0.4, 100.0): (0.5129, 0.5260, "0/3", "Naive"),
}


def _expected_total():
    """3 ds x 5 alpha x 6 seeds x 2 methods x 2 tau = 360."""
    return 3 * 5 * 6 * 2 * 2


def _load_ablation():
    if not os.path.exists(ABLATION_PATH):
        return [], f"MISSING: {ABLATION_PATH}"
    with open(ABLATION_PATH) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{ABLATION_PATH} is not a JSON list")
    return rows, f"results/tau_ablation.json ({len(rows)} rows)"


def _load_canonical_dp():
    from experiments.loaders import load_canonical_tau1
    rows = load_canonical_tau1()
    return [r for r in rows if r.get("attack") == ATTACK]


def _tau_of(r):
    return float(r.get("tau", TAU_REF))


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


def _per_cell_means(rows):
    """(ds, alpha, tau, method) -> mean DP, n."""
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], float(r["alpha"]), _tau_of(r), r["method"])].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = {"dp": float(np.mean([r["dp_clean"] for r in rs])), "n": len(rs)}
    return out


def _paired_wilcoxon_per_tau(rows, tau):
    """DRO vs Naive at a given τ, seed-paired, H1: naive_dp > dro_dp."""
    sub = [r for r in rows if abs(_tau_of(r) - tau) < 1e-9]
    df = pd.DataFrame(sub)
    out = []
    if df.empty:
        return pd.DataFrame(out)
    for (ds, alpha), g in df.groupby(["dataset", "alpha"], sort=True):
        naive = g[g["method"] == "naive"][["seed", "dp_clean"]]
        dro = g[g["method"] == "dro"][["seed", "dp_clean"]]
        merged = naive.merge(dro, on="seed", suffixes=("_naive", "_dro"))
        n = len(merged)
        if n < 2:
            continue
        diff = merged["dp_clean_naive"] - merged["dp_clean_dro"]
        p = _wilcoxon_greater(diff)
        out.append({
            "dataset": ds,
            "alpha": float(alpha),
            "tau": float(tau),
            "n_pairs": int(n),
            "dp_naive_mean": float(merged["dp_clean_naive"].mean()),
            "dp_dro_mean": float(merged["dp_clean_dro"].mean()),
            "dp_diff_mean": float(diff.mean()),
            "p_value": float(p),
            "wins_dro": int((diff > 0).sum()),
            "sig": "*" if p < ALPHA_LEVEL else "",
        })
    return pd.DataFrame(out).sort_values(["dataset", "alpha"]).reset_index(drop=True)


def _compare_to_historical(new_means):
    """For Adult/DP, flag CONFIRM/CONTRADICT against historical pilot numbers."""
    out = []
    for (alpha, tau), (hist_n, hist_d, hist_w, hist_v) in HISTORICAL_PILOT.items():
        new = new_means.get(("adult", alpha, tau, "naive"))
        new_dro = new_means.get(("adult", alpha, tau, "dro"))
        if new is None or new_dro is None:
            out.append({
                "alpha": alpha, "tau": tau,
                "hist_naive": hist_n, "hist_dro": hist_d,
                "hist_wins": hist_w, "hist_verdict": hist_v,
                "new_naive": None, "new_dro": None, "new_n": 0,
                "verdict": "NO NEW DATA",
            })
            continue
        new_n_dp = new["dp"]
        new_d_dp = new_dro["dp"]
        new_n = new["n"]
        # CONFIRM if sign matches AND magnitudes within 0.10 (loose — pilot vs full grid)
        naive_close = abs(new_n_dp - hist_n) <= 0.10
        dro_close = abs(new_d_dp - hist_d) <= 0.10
        new_dro_better = new_d_dp < new_n_dp
        hist_dro_better = hist_d < hist_n
        sign_match = (new_dro_better == hist_dro_better)
        if naive_close and dro_close and sign_match:
            verdict = "CONFIRM"
        elif sign_match and not (naive_close and dro_close):
            verdict = "CONFIRM (sign, magnitude drift)"
        elif not sign_match:
            verdict = "CONTRADICT (sign flip)"
        else:
            verdict = "PARTIAL"
        out.append({
            "alpha": alpha, "tau": tau,
            "hist_naive": hist_n, "hist_dro": hist_d,
            "hist_wins": hist_w, "hist_verdict": hist_v,
            "new_naive": new_n_dp, "new_dro": new_d_dp, "new_n": new_n,
            "verdict": verdict,
        })
    return out


def write_md(rows, source, n_expected, canonical_rows, all_means, wilc_by_tau, hist_rows):
    lines = []
    lines.append("# Agent A2 — τ ablation summary (τ ∈ {1, 10, 100}, DP attack)")
    lines.append("")
    lines.append("Analysis-only. No new training. Source: `results/tau_ablation.json` ")
    lines.append(f"({len(rows)}/{n_expected} rows, τ∈{{10,100}}) + canonical τ=1 IF-attack rows ")
    lines.append(f"({len(canonical_rows)} rows) as the reference via `experiments.loaders`.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- τ-ablation rows present: **{len(rows)}/{n_expected}** "
                 f"({100.0*len(rows)/n_expected:.1f}%)" if n_expected else "- no rows")
    if len(rows) < n_expected:
        lines.append("- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).")
    lines.append("")
    lines.append("## The τ=100 artifact — DP for Naive vs DRO at each τ")
    lines.append("")
    lines.append("Per (dataset, α), DP at τ=1 (canonical), τ=10, τ=100. Bold = lower (better).")
    lines.append("")
    lines.append("| dataset | α | n@τ1 | Naive τ=1 | DRO τ=1 | n@τ10 | Naive τ=10 | DRO τ=10 | n@τ100 | Naive τ=100 | DRO τ=100 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for alpha in ALPHAS:
            row = [ds, f"{alpha:.1f}"]
            for tau in [1.0, 10.0, 100.0]:
                n_m = all_means.get((ds, alpha, tau, "naive"))
                d_m = all_means.get((ds, alpha, tau, "dro"))
                if n_m and d_m:
                    row.append(str(n_m["n"]))
                    nd, dd = n_m["dp"], d_m["dp"]
                    row.append(f"**{nd:.4f}**" if nd < dd else f"{nd:.4f}")
                    row.append(f"**{dd:.4f}**" if dd < nd else f"{dd:.4f}")
                else:
                    row += ["—", "—", "—"]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Seed-paired Wilcoxon (DRO vs Naive) at each τ")
    lines.append("")
    lines.append("H1: naive_dp > dro_dp (DRO strictly lower DP). * marks p<0.05.")
    lines.append("")
    for tau in [1.0, 10.0, 100.0]:
        w = wilc_by_tau.get(tau, pd.DataFrame())
        lines.append(f"### τ = {tau:g}")
        lines.append("")
        if w.empty:
            lines.append("_(no paired rows yet)_")
            lines.append("")
            continue
        lines.append("| dataset | α | n | Naive DP | DRO DP | ΔDP(naive-dro) | wins_dro | p | sig |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in w.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['alpha']:.1f} | {int(r['n_pairs'])} "
                f"| {r['dp_naive_mean']:.4f} | {r['dp_dro_mean']:.4f} "
                f"| {r['dp_diff_mean']:+.4f} "
                f"| {int(r['wins_dro'])}/{int(r['n_pairs'])} "
                f"| {r['p_value']:.4f} | {r['sig']} |"
            )
        lines.append("")
    # τ=100 artifact demonstration.
    lines.append("## The flip: τ=100 → DRO loses; τ=1 → DRO wins")
    lines.append("")
    # Use Adult if available, else first dataset with data at both ends.
    flip_cell = None
    for ds in ["adult", "credit", "lsac"]:
        for alpha in [0.1, 0.2, 0.3]:
            t1 = wilc_by_tau.get(1.0, pd.DataFrame())
            t100 = wilc_by_tau.get(100.0, pd.DataFrame())
            if t1.empty or t100.empty:
                continue
            r1 = t1[(t1["dataset"] == ds) & (abs(t1["alpha"] - alpha) < 1e-9)]
            r100 = t100[(t100["dataset"] == ds) & (abs(t100["alpha"] - alpha) < 1e-9)]
            if r1.empty or r100.empty:
                continue
            flip_cell = (ds, alpha, r1.iloc[0], r100.iloc[0])
            break
        if flip_cell:
            break
    if flip_cell:
        ds, alpha, r1, r100 = flip_cell
        dro_wins_1 = r1["dp_diff_mean"] > 0 and r1["p_value"] < ALPHA_LEVEL
        dro_loses_100 = r100["dp_diff_mean"] < 0
        lines.append(
            f"Example cell **{ds} α={alpha:.1f}**: "
            f"τ=1 → Naive {r1['dp_naive_mean']:.4f} vs DRO {r1['dp_dro_mean']:.4f} "
            f"(Δ={r1['dp_diff_mean']:+.4f}, p={r1['p_value']:.4f}, "
            f"{'DRO WINS' if dro_wins_1 else 'no sig'}); "
            f"τ=100 → Naive {r100['dp_naive_mean']:.4f} vs DRO {r100['dp_dro_mean']:.4f} "
            f"(Δ={r100['dp_diff_mean']:+.4f}, p={r100['p_value']:.4f}, "
            f"{'DRO LOSES' if dro_loses_100 else 'no sig'}). "
            + ("The flip is reproduced cleanly in the new n=6 data."
               if (dro_wins_1 and dro_loses_100) else
               "The flip is only partially reproduced — see table above.")
        )
    else:
        lines.append("_(need both τ=1 and τ=100 paired rows to demonstrate the flip — currently incomplete)_")
    lines.append("")
    # Historical comparison.
    lines.append("## Comparison to historical pilot (paper Table~tab:tau-comparison)")
    lines.append("")
    lines.append("Paper Table~tab:tau-comparison lists τ=10/100 numbers from a HISTORICAL "
                 "stepped-schedule pilot (n=3). The new n=6 fixed-τ runs are a fresh, "
                 "provenance-clean re-test. CONFIRM = sign + magnitude match; "
                 "CONTRADICT = sign flip.")
    lines.append("")
    lines.append("| α | τ | hist Naive | hist DRO | hist wins | new Naive | new DRO | new n | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for h in hist_rows:
        new_n_s = f"{h['new_naive']:.4f}" if h["new_naive"] is not None else "—"
        new_d_s = f"{h['new_dro']:.4f}" if h["new_dro"] is not None else "—"
        lines.append(
            f"| {h['alpha']:.1f} | {h['tau']:g} "
            f"| {h['hist_naive']:.4f} | {h['hist_dro']:.4f} | {h['hist_wins']} "
            f"| {new_n_s} | {new_d_s} | {h['new_n']} | {h['verdict']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    print("AGENT A2: τ ablation summary (analysis-only)")
    print("=" * 78)

    rows, source = _load_ablation()
    n_expected = _expected_total()
    print(f"Loaded: {source}")
    if len(rows) < n_expected:
        print(f"  INCOMPLETE: {len(rows)}/{n_expected} rows "
              f"({100.0*len(rows)/n_expected:.1f}%) — partial-data mode.")

    try:
        canonical_rows = _load_canonical_dp()
        print(f"Canonical τ=1 DP-attack rows: {len(canonical_rows)}")
    except Exception as e:
        canonical_rows = []
        print(f"  WARN: canonical load failed: {e}")

    # Combine: ablation holds τ∈{10,100}; canonical holds τ=1.
    # (Do NOT mutate canonical rows; only read.)
    combined = list(rows) + list(canonical_rows)
    all_means = _per_cell_means(combined)
    wilc_by_tau = {tau: _paired_wilcoxon_per_tau(combined, tau) for tau in [1.0, 10.0, 100.0]}
    for tau, w in wilc_by_tau.items():
        print(f"  τ={tau:g}: {len(w)} paired Wilcoxon rows")

    hist_rows = _compare_to_historical(all_means)
    n_confirm = sum(1 for h in hist_rows if h["verdict"].startswith("CONFIRM"))
    n_contradict = sum(1 for h in hist_rows if h["verdict"].startswith("CONTRADICT"))
    n_no_data = sum(1 for h in hist_rows if h["verdict"] == "NO NEW DATA")
    print(f"Historical pilot comparison: {n_confirm} CONFIRM, {n_contradict} CONTRADICT, "
          f"{n_no_data} NO NEW DATA.")

    md_text = write_md(rows, source, n_expected, canonical_rows, all_means, wilc_by_tau, hist_rows)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print(f"  saved {OUT_MD}")

    print("\n" + "=" * 78)
    print("AGENT A2 SUMMARY MILESTONE complete (analysis-only).")
    print(f"  output: {OUT_MD}")
    print("=" * 78)


if __name__ == "__main__":
    main()