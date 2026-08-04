#!/usr/bin/env python3
"""
Agent C — Meeting summary.

Prints the key numbers for the meeting in a copy-paste-friendly
format. Pulls all headline numbers from results/canonical_tau1.json
via load_canonical_tau1() — no silent fallbacks to tau_ablation or
K_inner=5 backups.

Usage
-----
    python3 experiments/meeting_summary.py
    python3 experiments/meeting_summary.py --dataset adult
    python3 experiments/meeting_summary.py --no-banner
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS = os.path.join(ROOT, "results")

sys.path.insert(0, ROOT)


def _load_optional(name):
    """Load a non-headline optional JSON; empty list if missing (loud note)."""
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    # Never read archived / contaminated trees.
    if "stale_archived" in path.replace("\\", "/"):
        raise RuntimeError(f"Refusing to load stale path: {path}")
    with open(path) as f:
        return json.load(f)


def _load_tau1():
    """Headline numbers: canonical_tau1 only. Fail loud if missing/polluted."""
    from experiments.loaders import load_canonical_tau1

    rows = load_canonical_tau1()
    return rows, f"canonical_tau1.json ({len(rows)} rows, k_inner=10, tau=1)"


def _group(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def _stats(vals):
    n = len(vals)
    if n == 0:
        return (float("nan"), 0.0, 0)
    m = float(np.mean(vals))
    if n == 1:
        return (m, 0.0, 1)
    se = float(np.std(vals, ddof=1) / np.sqrt(n))
    return (m, se, n)


def _fmt(m, se, n, width=6, digits=3):
    if np.isnan(m):
        return f"{'—':>{width}}"
    return f"{m:>{width}.{digits}f} ± {se:<{4}.{digits}f} (n={n})"


def _wins(naive_vals, dro_vals):
    """Number of paired seeds where naive > dro on the metric."""
    n = min(len(naive_vals), len(dro_vals))
    return sum(1 for a, b in zip(naive_vals[:n], dro_vals[:n]) if a > b), n


def header(text, char="="):
    print()
    print(char * 78)
    print(f" {text}")
    print(char * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult", choices=["adult", "credit", "lsac"])
    ap.add_argument("--no-banner", action="store_true")
    args = ap.parse_args()

    if not args.no_banner:
        header("DRO-FAIR — meeting summary", char="#")
        print(f"Dataset focus: {args.dataset.upper()}")
        print("Source: results/canonical_tau1.json (via load_canonical_tau1); "
              "optional knn / random_vs_adv if present in results/")

    tau1, tau1_src = _load_tau1()
    knn = (
        _load_optional("knn_ablation_k5.json")
        + _load_optional("knn_ablation_k10.json")
        + _load_optional("knn_ablation_k15.json")
    )
    rvsa = _load_optional("random_vs_adversarial_new.json")

    print(f"\nTau=1 data source: {tau1_src}")

    # ---- 1. HEADLINE: DP attack, fixed tau=1 --------------------------------
    header("1. HEADLINE — DP attack, fixed tau=1 (canonical)")
    print(f"{'alpha':>6} | {'Naive DP':>22} | {'DRO DP':>22} | {'winner':>8} | {'DRO wins / n'}")
    print("-" * 78)

    sub = [r for r in tau1 if r["dataset"] == args.dataset and r["attack"] == "dp"]
    if not sub:
        print(f"  no DP-attack rows for dataset={args.dataset} in canonical_tau1.json")
    else:
        print("\n  tau=1 (canonical):")
        grouped = _group(sub, "alpha", "method")
        alphas = sorted({r["alpha"] for r in sub})
        for a in alphas:
            n_vals = [r["dp_clean"] for r in grouped.get((a, "naive"), [])]
            d_vals = [r["dp_clean"] for r in grouped.get((a, "dro"), [])]
            n_m, n_se, n_n = _stats(n_vals)
            d_m, d_se, d_n = _stats(d_vals)
            winner = "**DRO**" if (d_m < n_m and not np.isnan(d_m)) else "Naive"
            wins, total = _wins(n_vals, d_vals)
            print(f"  α={a:<4.1f} | {_fmt(n_m, n_se, n_n, 6, 3)} | "
                  f"{_fmt(d_m, d_se, d_n, 6, 3)} | {winner:>8} | {wins}/{total}")

    # ---- 2. IF / Combined at tau=1 -----------------------------------------
    header("2. IF and Combined attacks, fixed tau=1 (canonical)")
    for attack in ["if", "combined"]:
        sub = [r for r in tau1
               if r["dataset"] == args.dataset and r["attack"] == attack]
        if not sub:
            print(f"\n  attack = {attack}: no rows yet in canonical_tau1.json")
            continue
        print(f"\n  attack = {attack}:")
        print(f"  {'alpha':>6} | {'Naive DP':>22} | {'DRO DP':>22} | {'winner':>8} | {'DRO wins / n'}")
        print("  " + "-" * 76)
        grouped = _group(sub, "alpha", "method")
        for a in sorted({r["alpha"] for r in sub}):
            n_vals = [r["dp_clean"] for r in grouped.get((a, "naive"), [])]
            d_vals = [r["dp_clean"] for r in grouped.get((a, "dro"), [])]
            n_m, n_se, n_n = _stats(n_vals)
            d_m, d_se, d_n = _stats(d_vals)
            winner = "**DRO**" if (d_m < n_m and not np.isnan(d_m)) else "Naive"
            wins, total = _wins(n_vals, d_vals)
            print(f"  α={a:<4.1f} | {_fmt(n_m, n_se, n_n, 6, 3)} | "
                  f"{_fmt(d_m, d_se, d_n, 6, 3)} | {winner:>8} | {wins}/{total}")

    # ---- 3. IF k-NN ablation ------------------------------------------------
    header("3. IF k-NN ablation (insensitivity check; optional live files only)")
    if knn:
        print(f"  Attack = if, dataset = {args.dataset}")
        print(f"  {'alpha':>6} | {'k':>3} | {'Naive DP':>22} | {'DRO DP':>22} | "
              f"{'IF metric (Naive)':>16} | {'IF metric (DRO)':>16}")
        print("  " + "-" * 100)
        for a in sorted({r["alpha"] for r in knn if r["dataset"] == args.dataset}):
            for k in [5, 10, 15]:
                n = [r for r in knn if r["dataset"] == args.dataset
                     and np.isclose(r["alpha"], a) and r["k_nn"] == k and r["method"] == "naive"]
                d = [r for r in knn if r["dataset"] == args.dataset
                     and np.isclose(r["alpha"], a) and r["k_nn"] == k and r["method"] == "dro"]
                if not n or not d:
                    continue
                n_m, n_se, n_n = _stats([r["dp_clean"] for r in n])
                d_m, d_se, d_n = _stats([r["dp_clean"] for r in d])
                if_m, if_se, _ = _stats([r["if_clean"] for r in n])
                ifd_m, ifd_se, _ = _stats([r["if_clean"] for r in d])
                print(f"  α={a:<4.1f} | {k:>3} | {_fmt(n_m, n_se, n_n, 6, 4)} | "
                      f"{_fmt(d_m, d_se, d_n, 6, 4)} | "
                      f"{if_m:>8.4f} ± {if_se:.4f}    | "
                      f"{ifd_m:>8.4f} ± {ifd_se:.4f}")
            print()
    else:
        print("  no live k-NN ablation JSONs in results/ (not falling back to stale_archived)")

    # ---- 4. Random vs adversarial ------------------------------------------
    header("4. Random vs adversarial — absolute DP (optional live file)")
    if rvsa:
        for ds in ["adult", "credit", "lsac"]:
            sub = [r for r in rvsa if r["dataset"] == ds]
            if not sub:
                continue
            print(f"\n  {ds.upper()}:")
            print(f"  {'alpha':>6} | {'clean DP':>12} | {'random DP':>22} | "
                  f"{'adversarial DP':>22} | {'adv/random ratio'}")
            print("  " + "-" * 90)
            for a in sorted({r["alpha"] for r in sub}):
                rows = [r for r in sub if r["alpha"] == a]
                clean_v = [r["clean"]["dp"] for r in rows]
                rand_v = [r["random"]["dp"] for r in rows]
                adv_v = [r["adversarial"]["dp"] for r in rows]
                c_m, c_se, c_n = _stats(clean_v)
                r_m, r_se, r_n = _stats(rand_v)
                a_m, a_se, a_n = _stats(adv_v)
                dc = r_m - c_m
                da = a_m - c_m
                if dc > 1e-4 and da > 0:
                    ratio = f"{da / dc:.1f}×"
                else:
                    ratio = "—"
                print(f"  α={a:<4.1f} | {c_m:>5.4f}      | "
                      f"{_fmt(r_m, r_se, r_n, 6, 4)} | "
                      f"{_fmt(a_m, a_se, a_n, 6, 4)} | {ratio}")
    else:
        print("  no live random_vs_adversarial_new.json (not falling back to stale_archived)")

    # ---- 5. Lambda grid (optional) -----------------------------------------
    header("5. Lambda init × lr_lambda grid (optional live files)")
    grid = _load_optional("lambda_lr_grid.json")
    if not grid:
        grid = _load_optional("lambda_grid_comprehensive.json")
    if grid:
        print(f"  {len(grid)} cells present")
        r = grid[0]
        print(f"  Sample cell keys: {sorted(r.keys())}")
        if "lambda_init" in r and "lr_lambda" in r:
            print(f"  Latest-ish cell: alpha={r.get('alpha')} seed={r.get('seed')} "
                  f"lambda_init={r.get('lambda_init')} lr_lambda={r.get('lr_lambda')} "
                  f"-> acc={r.get('acc')} dp={r.get('dp')}")
    else:
        print("  no live lambda grid JSON in results/")

    # ---- 6. Headline one-liner ---------------------------------------------
    header("6. One-liner")
    sub = [r for r in tau1 if r["dataset"] == args.dataset and r["attack"] == "dp"]
    if sub:
        grouped = _group(sub, "alpha", "method")
        alphas = sorted({r["alpha"] for r in sub})
        wins = total = 0
        for a in alphas:
            n_vals = [r["dp_clean"] for r in grouped.get((a, "naive"), [])]
            d_vals = [r["dp_clean"] for r in grouped.get((a, "dro"), [])]
            w, t = _wins(n_vals, d_vals)
            wins += w
            total += t
        print(f"  At fixed tau=1, {args.dataset.upper()}, DP attack: DRO wins {wins}/{total} α-seeds")
        print(f"  (out of {len(alphas)} alpha values × {total // max(len(alphas),1)} seeds each)")
        print(f"  Source: {tau1_src}")
    print()


if __name__ == "__main__":
    main()
