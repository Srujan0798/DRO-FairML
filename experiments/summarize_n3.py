#!/usr/bin/env python3
"""Agent N3 — COMPAS + German Credit extended-datasets summary (analysis-only).

Reads results/extended_datasets.json (2 datasets × 3 attacks × 5 α × 6 seeds
× 2 methods = 360 configs) and writes results/extended_datasets_summary.md.

Per (dataset, attack, alpha): Acc/DP/IF means for naive and dro, plus a
seed-paired Wilcoxon (one-sided, H1: naive DP > dro DP) — i.e. DRO improves
fairness. * marks p<0.05.

THE REPLICATION VERDICT: the Adult/Credit pattern (DRO better on DP at α≤0.2)
is the paper's main claim. Does it REPLICATE on COMPAS and German?
  - If yes → the claim generalizes (major strengthening).
  - If no, especially on German (small, 1000 rows, noisy) → report the scope
    limit plainly. A stated scope beats an overclaim.

Constant-predictor baselines are computed FROM DATA (the loader's get_dataset
returns y_train; majority class on TEST) — never hardcoded. The
α≥0.3 defensible-regime discussion needs these (Adult 0.7521, Credit 0.7788,
LSAC 0.9016, COMPAS ~0.5334, German 0.7000).

Outputs:
    results/extended_datasets_summary.md

Run:
    PYTHONPATH=. python3 experiments/summarize_n3.py
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

ABLATION_PATH = os.path.join(RESULTS_DIR, "extended_datasets.json")
OUT_MD = os.path.join(RESULTS_DIR, "extended_datasets_summary.md")

ALPHA_LEVEL = 0.05
DATASETS = ["compas", "german"]
ATTACKS = ["dp", "if", "combined"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
METHODS = ["naive", "dro"]


def _expected_total():
    """2 ds × 3 attacks × 5 α × 6 seeds × 2 methods = 360."""
    return len(DATASETS) * len(ATTACKS) * len(ALPHAS) * 6 * len(METHODS)


def _load_ablation():
    if not os.path.exists(ABLATION_PATH):
        return [], f"MISSING: {ABLATION_PATH}"
    with open(ABLATION_PATH) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{ABLATION_PATH} is not a JSON list")
    return rows, f"results/extended_datasets.json ({len(rows)} rows)"


def _wilcoxon_greater(d):
    """One-sided Wilcoxon signed-rank, H1: d > 0. Returns 1.0 if not computable."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0.0):
        return 1.0
    try:
        _, p = wilcoxon(d, alternative="greater", zero_method="wilcox")
        return float(p)
    except ValueError:
        return 1.0


def _mean(rows, key):
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def _per_cell_means(rows):
    """(ds, attack, alpha, method) -> {acc, dp, if, n}."""
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["attack"], float(r["alpha"]), r["method"])].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = {
            "acc": _mean(rs, "acc_clean"),
            "dp": _mean(rs, "dp_clean"),
            "if": _mean(rs, "if_clean"),
            "n": len(rs),
        }
    return out


def _paired_wilcoxon(rows):
    """Per (ds, attack, alpha): seed-paired Wilcoxon H1: naive_dp > dro_dp.

    ΔDP = naive_dp - dro_dp; positive ΔDP ⇒ DRO fairer. wins_dro counts seeds
    where ΔDP > 0. * marks p<0.05. Also returns acc means + the IF metric means
    so the per-cell table is complete.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    out = []
    for (ds, attack, alpha), g in df.groupby(["dataset", "attack", "alpha"], sort=True):
        naive = g[g["method"] == "naive"][["seed", "dp_clean", "acc_clean", "if_clean"]]
        dro = g[g["method"] == "dro"][["seed", "dp_clean", "acc_clean", "if_clean"]]
        merged = naive.merge(dro, on="seed", suffixes=("_naive", "_dro"))
        n = len(merged)
        if n < 1:
            continue
        diff_dp = merged["dp_clean_naive"] - merged["dp_clean_dro"]
        p = _wilcoxon_greater(diff_dp) if n >= 2 else 1.0
        out.append({
            "dataset": ds,
            "attack": attack,
            "alpha": float(alpha),
            "n_pairs": int(n),
            "acc_naive": float(merged["acc_clean_naive"].mean()),
            "acc_dro": float(merged["acc_clean_dro"].mean()),
            "dp_naive": float(merged["dp_clean_naive"].mean()),
            "dp_dro": float(merged["dp_clean_dro"].mean()),
            "if_naive": float(merged["if_clean_naive"].mean()),
            "if_dro": float(merged["if_clean_dro"].mean()),
            "delta_dp": float(diff_dp.mean()),
            "p_value": float(p),
            "wins_dro": int((diff_dp > 0).sum()),
            "sig": "*" if p < ALPHA_LEVEL else "",
        })
    return pd.DataFrame(out).sort_values(
        ["dataset", "attack", "alpha"]).reset_index(drop=True)


def _constant_predictor_baseline(name):
    """Majority-class accuracy on TEST, computed from data via get_dataset.

    Never hardcoded. The α≥0.3 defensible-regime comparison needs this.
    Returns (acc, majority_class) or (nan, nan) on failure.
    """
    try:
        from src.data.datasets import get_dataset
        Xtr, ytr, atr, Xv, yv, av, Xte, yte, ate, dname = get_dataset(name)
        maj = int(np.bincount(ytr.astype(int)).argmax())
        acc = float((yte.astype(int) == maj).mean())
        return acc, maj
    except Exception as e:
        print(f"  WARN: constant-predictor baseline for {name} failed: {e}")
        return float("nan"), -1


def _verdict_replicate(delta_df):
    """Replication verdict: does the Adult/Credit pattern (DRO better on DP at
    α≤0.2) replicate on COMPAS and German?

    The Adult/Credit pattern: at α≤0.2, DRO's DP is lower than naive's (ΔDP>0)
    and the Wilcoxon is significant (p<0.05) in DRO's favour, without an accuracy
    collapse. We check the DP-attack cells at α∈{0.1, 0.2} as the headline, and
    report the full α∈{0.0..0.4} picture per dataset.
    """
    if delta_df.empty:
        return ("INCOMPLETE", "Not yet answerable — no paired rows landed yet.")

    per_ds = {}
    for ds in DATASETS:
        # Headline cells: dp attack, α∈{0.1,0.2}.
        headline = delta_df[
            (delta_df["dataset"] == ds)
            & (delta_df["attack"] == "dp")
            & (delta_df["alpha"].isin([0.1, 0.2]))
        ]
        n_cells = len(headline)
        sig_cells = int((headline["p_value"] < ALPHA_LEVEL).sum())
        wins = int(headline["wins_dro"].sum())
        n_pairs = int(headline["n_pairs"].sum())
        mean_ddp = float(headline["delta_dp"].mean()) if n_cells else float("nan")
        # Accuracy: DRO not collapsing vs naive
        acc_drop = float((headline["acc_dro"] - headline["acc_naive"]).mean()) if n_cells else float("nan")
        per_ds[ds] = {
            "n_cells": n_cells, "sig_cells": sig_cells, "wins": wins,
            "n_pairs": n_pairs, "mean_ddp": mean_ddp, "acc_drop": acc_drop,
        }

    def _verdict_one(ds):
        d = per_ds[ds]
        if d["n_cells"] == 0:
            return "no data"
        # Replicate: both α≤0.2 DP cells significant in DRO's favour AND mean ΔDP>0.
        if d["sig_cells"] == d["n_cells"] and d["mean_ddp"] > 0:
            return f"REPLICATES ({d['sig_cells']}/{d['n_cells']} DP cells at α≤0.2 sig; mean ΔDP={d['mean_ddp']:+.4f}; Δacc={d['acc_drop']:+.4f})"
        if d["mean_ddp"] > 0 and d["wins"] > d["n_pairs"] / 2:
            return (f"PARTIAL (mean ΔDP={d['mean_ddp']:+.4f} > 0, DRO wins {d['wins']}/{d['n_pairs']} "
                    f"seeds, but only {d['sig_cells']}/{d['n_cells']} cells p<0.05)")
        if d["mean_ddp"] <= 0:
            return (f"DOES NOT REPLICATE (mean ΔDP={d['mean_ddp']:+.4f} ≤ 0 at α≤0.2 — DRO is not "
                    f"fairer than naive on DP for {ds.upper()})")
        return f"AMBIGUOUS (mean ΔDP={d['mean_ddp']:+.4f}, {d['sig_cells']}/{d['n_cells']} sig)"

    v_compas = _verdict_one("compas")
    v_german = _verdict_one("german")

    compas_rep = per_ds["compas"]["sig_cells"] == per_ds["compas"]["n_cells"] and per_ds["compas"]["mean_ddp"] > 0
    german_rep = per_ds["german"]["sig_cells"] == per_ds["german"]["n_cells"] and per_ds["german"]["mean_ddp"] > 0
    german_partial = per_ds["german"]["mean_ddp"] > 0 and per_ds["german"]["wins"] > per_ds["german"]["n_pairs"] / 2

    if compas_rep and german_rep:
        headline = "REPLICATES on both COMPAS and German — the Adult/Credit claim generalizes."
    elif compas_rep and not german_rep and not german_partial:
        headline = ("PARTIAL replication: COMPAS replicates but German does NOT. "
                    "German Credit is small (1000 rows) and noisy — the DRO advantage "
                    "does not reach significance at α≤0.2 there. SCOPE LIMIT: the claim "
                    "generalizes to COMPAS but not to German at this n.")
    elif compas_rep and german_partial:
        headline = ("MOSTLY REPLICATES: COMPAS replicates; German is PARTIAL "
                    "(direction right, DRO wins the majority of seeds, but not p<0.05). "
                    "Likely underpowered at n=6 on 1000 rows.")
    elif not compas_rep and not german_rep:
        headline = ("DOES NOT REPLICATE on either dataset at α≤0.2. The Adult/Credit "
                    "pattern does not generalize to COMPAS or German under this protocol. "
                    "Report as a scope limit of the DRO claim.")
    else:
        headline = (f"MIXED — COMPAS: {v_compas}; German: {v_german}. See per-cell table "
                    f"for the honest picture.")

    detail = (f"COMPAS: {v_compas}. German: {v_german}. "
              f"(Headline = DP attack, α∈{{0.1,0.2}}, paired Wilcoxon H1: naive DP > dro DP, "
              f"p<{ALPHA_LEVEL}.)")
    return (headline, detail)


def write_md(rows, source, n_expected, baselines, means, delta_df, verdict_head, verdict_detail):
    lines = []
    lines.append("# Agent N3 — COMPAS + German Credit extended-datasets summary")
    lines.append("")
    lines.append("Analysis-only. No new training. Source: `results/extended_datasets.json` ")
    lines.append(f"({len(rows)}/{n_expected} rows). Canonical protocol: τ=1.0, k_inner=10, ")
    lines.append("epochs=60, pgd_steps=20, λ_init=0.0, radii_mode='uniform', coordinated=False, ")
    lines.append("corruptor_type='adversarial', attack_k=5.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    if n_expected:
        pct = 100.0 * len(rows) / n_expected
        lines.append(f"- Extended-datasets rows present: **{len(rows)}/{n_expected}** ({pct:.1f}%)")
    if len(rows) < n_expected:
        lines.append("- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).")
    lines.append("")
    # Constant-predictor baselines (computed from data, never hardcoded).
    lines.append("## Constant-predictor baselines (computed from data)")
    lines.append("")
    lines.append("Majority-class accuracy on TEST, computed via `get_dataset(name)` ")
    lines.append("(the loader returns y_train; majority class is fit on train, scored on test). ")
    lines.append("These are the per-dataset baselines for the α≥0.3 defensible-regime discussion. ")
    lines.append("Reference (locked): Adult 0.7521, Credit 0.7788, LSAC 0.9016.")
    lines.append("")
    lines.append("| dataset | majority class | const-predictor TEST acc |")
    lines.append("|---|---|---|")
    for ds in DATASETS:
        acc, maj = baselines.get(ds, (float("nan"), -1))
        lines.append(f"| {ds} | {maj} | {acc:.4f} |")
    lines.append("")
    # Per-cell means.
    lines.append("## Per-cell means — (dataset, attack, α) × method")
    lines.append("")
    lines.append("| dataset | attack | α | n | acc_naive | acc_dro | DP_naive | DP_dro | IF_naive | IF_dro |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for attack in ATTACKS:
            for alpha in ALPHAS:
                mn = means.get((ds, attack, alpha, "naive"))
                md = means.get((ds, attack, alpha, "dro"))
                if mn is None or md is None:
                    lines.append(f"| {ds} | {attack} | {alpha:.1f} | 0 | — | — | — | — | — | — |")
                    continue
                lines.append(
                    f"| {ds} | {attack} | {alpha:.1f} | {mn['n']} "
                    f"| {mn['acc']:.4f} | {md['acc']:.4f} "
                    f"| {mn['dp']:.4f} | {md['dp']:.4f} "
                    f"| {mn['if']:.4f} | {md['if']:.4f} |"
                )
    lines.append("")
    # Paired Wilcoxon.
    lines.append("## Paired Wilcoxon — H1: naive DP > dro DP (DRO improves fairness)")
    lines.append("")
    lines.append(f"ΔDP = DP_naive − DP_dro (positive ⇒ DRO fairer). * marks p<{ALPHA_LEVEL}. ")
    lines.append("`wins_dro` = seeds where ΔDP > 0.")
    lines.append("")
    if delta_df.empty:
        lines.append("_(no paired rows yet)_")
    else:
        lines.append("| dataset | attack | α | n | acc_naive | acc_dro | DP_naive | DP_dro | ΔDP | wins_dro | p | sig | IF_naive | IF_dro |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in delta_df.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['attack']} | {r['alpha']:.1f} "
                f"| {int(r['n_pairs'])} "
                f"| {r['acc_naive']:.4f} | {r['acc_dro']:.4f} "
                f"| {r['dp_naive']:.4f} | {r['dp_dro']:.4f} "
                f"| {r['delta_dp']:+.4f} "
                f"| {int(r['wins_dro'])}/{int(r['n_pairs'])} "
                f"| {r['p_value']:.4f} | {r['sig']} "
                f"| {r['if_naive']:.4f} | {r['if_dro']:.4f} |"
            )
    lines.append("")
    # Verdict.
    lines.append("## REPLICATION VERDICT — does the Adult/Credit pattern (DRO better on DP at α≤0.2) replicate?")
    lines.append("")
    lines.append(f"**{verdict_head}**")
    lines.append("")
    lines.append(verdict_detail)
    lines.append("")
    return "\n".join(lines)


def main():
    print("AGENT N3: COMPAS + German extended-datasets summary (analysis-only)")
    print("=" * 78)

    rows, source = _load_ablation()
    n_expected = _expected_total()
    print(f"Loaded: {source}")
    if len(rows) < n_expected:
        print(f"  INCOMPLETE: {len(rows)}/{n_expected} rows "
              f"({100.0*len(rows)/n_expected:.1f}%) — partial-data mode.")

    print("Computing constant-predictor baselines from data (get_dataset)…")
    baselines = {}
    for ds in DATASETS:
        acc, maj = _constant_predictor_baseline(ds)
        baselines[ds] = (acc, maj)
        print(f"  {ds}: majority={maj} const-predictor TEST acc={acc:.4f}")

    means = _per_cell_means(rows)
    delta_df = _paired_wilcoxon(rows)
    print(f"Computed {len(delta_df)} paired Wilcoxon cells.")

    verdict_head, verdict_detail = _verdict_replicate(delta_df)

    md_text = write_md(rows, source, n_expected, baselines, means, delta_df,
                       verdict_head, verdict_detail)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print(f"  saved {OUT_MD}")

    print("\nREPLICATION VERDICT:")
    print("  " + verdict_head)
    print("  " + verdict_detail)

    print("\n" + "=" * 78)
    print("AGENT N3 SUMMARY MILESTONE complete (analysis-only).")
    print(f"  output: {OUT_MD}")
    print("=" * 78)


if __name__ == "__main__":
    main()