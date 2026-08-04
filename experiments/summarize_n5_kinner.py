#!/usr/bin/env python3
"""Agent N5 — K_inner ablation summary (analysis-only).

Reads results/kinner_ablation.json (K_inner ∈ {5, 20}, DRO only, attack='dp',
3 datasets × 5 α × 6 seeds × 2 K = 180 configs) and results/canonical_tau1.json
(K_inner=10, the reference). Per (dataset, α): DP/IF/acc at K_inner=5, 10, 20
(DRO only). Paired Wilcoxon K=5 vs K=10 and K=20 vs K=10, H1: K_alt > K=10
(i.e. changing K_inner strictly raises the DP metric — DRO worse). Wall-clock
per config is reported.

Outputs:
    results/kinner_ablation_summary.md

Run:
    PYTHONPATH=. python3 experiments/summarize_n5_kinner.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS_DIR = os.path.join(ROOT, "results")

ABLATION_PATH = os.path.join(RESULTS_DIR, "kinner_ablation.json")
OUT_MD = os.path.join(RESULTS_DIR, "kinner_ablation_summary.md")

ALPHA_LEVEL = 0.05
ATTACK = "dp"
METHOD = "dro"
DATASETS = ["adult", "credit", "lsac"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
K_INNERS_NEW = [5, 20]
K_INNER_REF = 10


def _expected_total():
    """3 ds x 5 alpha x 6 seeds x 2 K_inner = 180 (DRO only)."""
    return 3 * 5 * 6 * 2


def _load_ablation():
    if not os.path.exists(ABLATION_PATH):
        return [], f"MISSING: {ABLATION_PATH}"
    with open(ABLATION_PATH) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{ABLATION_PATH} is not a JSON list")
    return rows, f"results/kinner_ablation.json ({len(rows)} rows)"


def _load_canonical_dro_dp():
    """Canonical DRO/DP rows = the K_inner=10 reference."""
    from experiments.loaders import load_canonical_tau1
    rows = load_canonical_tau1()
    out = []
    for r in rows:
        if r.get("attack") != ATTACK:
            continue
        if r.get("method") != METHOD:
            continue
        # Canonical rows carry k_inner=10 by loader invariant. Defensive:
        if int(r.get("k_inner", K_INNER_REF)) != K_INNER_REF:
            continue
        out.append(r)
    return out


def _kinner_of(r):
    """k_inner field. Accepts a dict (row) OR a scalar (pandas .apply)."""
    if isinstance(r, dict):
        v = r.get("k_inner", K_INNER_REF)
    else:
        v = r
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return K_INNER_REF
    return int(v)


def _mean(rows, key):
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def _per_cell_means(rows):
    """(ds, alpha, k_inner) -> {dp, if, acc, n, wall_clock_per_config}."""
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], float(r["alpha"]), _kinner_of(r))].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = {
            "dp": _mean(rs, "dp_clean"),
            "if": _mean(rs, "if_clean"),
            "acc": _mean(rs, "acc_clean"),
            "n": len(rs),
            "wall_clock_per_config": float(np.mean([r.get("total_time", 0.0) for r in rs])),
        }
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


def _paired_compare(rows, k_alt, k_ref):
    """Seed-paired Wilcoxon (k_alt - k_ref) on DP/IF for DRO at each (ds,α).
    H1: k_alt > k_ref (alt K strictly raises the metric — DRO worse)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    out = []
    alt = df[df["k_inner"].apply(_kinner_of) == k_alt]
    ref = df[df["k_inner"].apply(_kinner_of) == k_ref]
    if alt.empty or ref.empty:
        return pd.DataFrame()
    for (ds, alpha), ag in alt.groupby(["dataset", "alpha"], sort=True):
        rg = ref[(ref["dataset"] == ds) & (ref["alpha"].astype(float) == float(alpha))]
        if rg.empty:
            continue
        merged = ag.set_index("seed")[["dp_clean", "if_clean"]].join(
            rg.set_index("seed")[["dp_clean", "if_clean"]], lsuffix=f"_k{k_alt}", rsuffix=f"_k{k_ref}", how="inner"
        )
        n = len(merged)
        if n < 2:
            continue
        diff_dp = merged[f"dp_clean_k{k_alt}"] - merged[f"dp_clean_k{k_ref}"]
        diff_if = merged[f"if_clean_k{k_alt}"] - merged[f"if_clean_k{k_ref}"]
        p_dp = _wilcoxon_greater(diff_dp)
        p_if = _wilcoxon_greater(diff_if)
        out.append({
            "dataset": ds,
            "alpha": float(alpha),
            "k_alt": int(k_alt),
            "k_ref": int(k_ref),
            "n_pairs": int(n),
            "dp_k_alt": float(merged[f"dp_clean_k{k_alt}"].mean()),
            "dp_k_ref": float(merged[f"dp_clean_k{k_ref}"].mean()),
            "dp_diff": float(diff_dp.mean()),
            "dp_p_value": float(p_dp),
            "dp_wins_alt": int((diff_dp > 0).sum()),
            "if_k_alt": float(merged[f"if_clean_k{k_alt}"].mean()),
            "if_k_ref": float(merged[f"if_clean_k{k_ref}"].mean()),
            "if_diff": float(diff_if.mean()),
            "if_p_value": float(p_if),
        })
    return pd.DataFrame(out).sort_values(["dataset", "alpha"]).reset_index(drop=True)


def write_md(rows, source, n_expected, canonical_rows, means, w_5_vs_10, w_20_vs_10):
    lines = []
    lines.append("# Agent N5 — K_inner ablation summary (K ∈ {5, 10, 20}, DRO only, DP attack)")
    lines.append("")
    lines.append("Analysis-only. No new training. Source: `results/kinner_ablation.json` ")
    lines.append(f"({len(rows)}/{n_expected} rows, K∈{{5,20}}) + canonical K=10 DRO/DP rows ")
    lines.append(f"({len(canonical_rows)} rows) as the reference via `experiments.loaders`.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- K_inner ablation rows present: **{len(rows)}/{n_expected}** "
                 f"({100.0*len(rows)/n_expected:.1f}%)" if n_expected else "- no rows")
    if len(rows) < n_expected:
        lines.append("- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).")
    lines.append("")
    lines.append("## Per-cell means for DRO (dataset, α, K_inner)")
    lines.append("")
    lines.append("| dataset | α | K | n | DP | IF | acc | wall/config (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for alpha in ALPHAS:
            for k in [5, 10, 20]:
                m = means.get((ds, alpha, k))
                if m is None:
                    lines.append(f"| {ds} | {alpha:.1f} | {k} | 0 | — | — | — | — |")
                    continue
                lines.append(
                    f"| {ds} | {alpha:.1f} | {k} | {m['n']} "
                    f"| {m['dp']:.4f} | {m['if']:.4f} | {m['acc']:.4f} "
                    f"| {m['wall_clock_per_config']:.1f} |"
                )
    lines.append("")
    # Wilcoxon tables.
    for label, w in [("K=5 vs K=10", w_5_vs_10), ("K=20 vs K=10", w_20_vs_10)]:
        lines.append(f"## Paired Wilcoxon — {label} (DRO, DP attack)")
        lines.append("")
        lines.append(f"H1: k_alt > k_ref (alt K strictly raises the metric — DRO worse). "
                     f"* marks p<{ALPHA_LEVEL}.")
        lines.append("")
        if w.empty:
            lines.append("_(no paired rows yet — need both k_alt and k_ref rows for the same seed)_")
            lines.append("")
            continue
        lines.append("| dataset | α | n | DP_k_alt | DP_k_ref | ΔDP | wins_alt | p_DP | sig | ΔIF | p_IF |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in w.iterrows():
            sig = "*" if r["dp_p_value"] < ALPHA_LEVEL else ""
            lines.append(
                f"| {r['dataset']} | {r['alpha']:.1f} | {int(r['n_pairs'])} "
                f"| {r['dp_k_alt']:.4f} | {r['dp_k_ref']:.4f} "
                f"| {r['dp_diff']:+.4f} "
                f"| {int(r['dp_wins_alt'])}/{int(r['n_pairs'])} "
                f"| {r['dp_p_value']:.4f} {sig} "
                f"| {r['if_diff']:+.4f} | {r['if_p_value']:.4f} |"
            )
        lines.append("")
    # Verdict.
    both = [w_5_vs_10, w_20_vs_10]
    if all(w.empty for w in both):
        verdict = ("Not yet answerable — need seed-paired K_alt and K=10 rows "
                   "(currently INCOMPLETE).")
    else:
        max_dp_diff = 0.0
        max_p = 1.0
        n_sig = 0
        n_cells = 0
        for w in both:
            if w.empty:
                continue
            n_cells += len(w)
            n_sig += int((w["dp_p_value"] < ALPHA_LEVEL).sum())
            if not w.empty:
                max_dp_diff = max(max_dp_diff, float(w["dp_diff"].abs().max()))
                # use min p so 'max_p' label stays meaningful as 'strongest sig'
                max_p = min(max_p, float(w["dp_p_value"].min()))
        if n_sig == 0 and max_dp_diff < 5e-3:
            verdict = (f"No: K_inner beyond 5 does NOT change anything materially "
                       f"(max |ΔDP|={max_dp_diff:.4f}, 0/{n_cells} cells p<0.05). "
                       "DRO is K_inner-robust within {5,10,20}.")
        elif n_sig == 0:
            verdict = (f"Mostly no: K_inner has a small directional effect "
                       f"(max |ΔDP|={max_dp_diff:.4f}) but 0/{n_cells} cells reach p<0.05. "
                       "Treat DRO as K_inner-robust within {5,10,20}.")
        else:
            verdict = (f"Partial: {n_sig}/{n_cells} cells show a significant K_inner effect "
                       f"(max |ΔDP|={max_dp_diff:.4f}, min p={max_p:.4f}). "
                       "K_inner matters in some cells — report the sensitivity.")
    lines.append("## Verdict — does K_inner beyond 5 change anything materially?")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    # Wall-clock summary.
    lines.append("## Wall-clock per config (DRO, mean over seeds)")
    lines.append("")
    lines.append("| K_inner | mean wall/config (s) | n rows |")
    lines.append("|---|---|---|")
    by_k = defaultdict(list)
    for (ds, alpha, k), m in means.items():
        for _ in range(m["n"]):
            by_k[k].append(m["wall_clock_per_config"])
    for k in sorted(by_k.keys()):
        wc = by_k[k]
        lines.append(f"| {k} | {np.mean(wc):.1f} | {len(wc)} |")
    lines.append("")
    return "\n".join(lines), verdict


def main():
    print("AGENT N5: K_inner ablation summary (analysis-only)")
    print("=" * 78)

    rows, source = _load_ablation()
    n_expected = _expected_total()
    print(f"Loaded: {source}")
    if len(rows) < n_expected:
        print(f"  INCOMPLETE: {len(rows)}/{n_expected} rows "
              f"({100.0*len(rows)/n_expected:.1f}%) — partial-data mode.")

    try:
        canonical_rows = _load_canonical_dro_dp()
        print(f"Canonical K=10 DRO/DP rows: {len(canonical_rows)}")
    except Exception as e:
        canonical_rows = []
        print(f"  WARN: canonical load failed: {e}")

    # Combine ablation (K∈{5,20}) + canonical (K=10) for DRO/DP. Defensive filter.
    ablation_dro_dp = [r for r in rows
                       if r.get("attack") == ATTACK and r.get("method") == METHOD]
    combined = list(ablation_dro_dp) + list(canonical_rows)
    means = _per_cell_means(combined)
    w_5_vs_10 = _paired_compare(combined, 5, 10)
    w_20_vs_10 = _paired_compare(combined, 20, 10)
    print(f"  K=5 vs K=10: {len(w_5_vs_10)} paired rows")
    print(f"  K=20 vs K=10: {len(w_20_vs_10)} paired rows")

    md_text, verdict = write_md(rows, source, n_expected, canonical_rows, means, w_5_vs_10, w_20_vs_10)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print(f"  saved {OUT_MD}")

    print("\nVerdict:")
    print("  " + verdict)

    print("\n" + "=" * 78)
    print("AGENT N5 SUMMARY MILESTONE complete (analysis-only).")
    print(f"  output: {OUT_MD}")
    print("=" * 78)


if __name__ == "__main__":
    main()