#!/usr/bin/env python3
"""
Generate results/high_alpha_summary.csv from all tau ablation JSONs.

One row per (tau, alpha, method) with mean±std acc and DP across seeds.

Run:
    cd /Users/srujansai/Desktop/DRO-FairML && python experiments/generate_high_alpha_summary.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

TAU_FILES = {
    1:   os.path.join(RESULTS_DIR, "tau_ablation_tau1.json"),
    5:   os.path.join(RESULTS_DIR, "tau_ablation_tau5.json"),
    10:  os.path.join(RESULTS_DIR, "tau_ablation_tau10.json"),
    100: os.path.join(RESULTS_DIR, "tau_ablation_tau100.json"),
}

OUTPUT_CSV = os.path.join(RESULTS_DIR, "high_alpha_summary.csv")


def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    all_rows = []
    for tau_val, path in sorted(TAU_FILES.items()):
        rows = load_json(path)
        df = pd.DataFrame(rows)
        sub = df[(df["dataset"] == "adult") & (df["attack"] == "dp")].copy()
        for alpha in sorted(sub["alpha"].unique()):
            g = sub[sub["alpha"] == alpha]
            for meth in ("naive", "dro"):
                mg = g[g["method"] == meth]
                if mg.empty:
                    continue
                acc_vals = mg["acc_clean"].values
                dp_vals = mg["dp_clean"].values
                if_vals = mg["if_clean"].values
                all_rows.append({
                    "tau": tau_val,
                    "alpha": float(alpha),
                    "method": meth,
                    "n_seeds": len(mg),
                    "acc_mean": np.mean(acc_vals),
                    "acc_std": np.std(acc_vals, ddof=1) if len(acc_vals) > 1 else 0.0,
                    "dp_mean": np.mean(dp_vals),
                    "dp_std": np.std(dp_vals, ddof=1) if len(dp_vals) > 1 else 0.0,
                    "if_mean": np.mean(if_vals),
                    "if_std": np.std(if_vals, ddof=1) if len(if_vals) > 1 else 0.0,
                })

    summary = pd.DataFrame(all_rows)
    summary = summary.sort_values(["alpha", "tau", "method"]).reset_index(drop=True)

    # Format mean±std columns
    summary["acc_fmt"] = summary.apply(
        lambda r: f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}", axis=1)
    summary["dp_fmt"] = summary.apply(
        lambda r: f"{r['dp_mean']:.4f}±{r['dp_std']:.4f}", axis=1)

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(summary)} rows to {OUTPUT_CSV}")
    print(summary[["tau", "alpha", "method", "n_seeds", "acc_fmt", "dp_fmt"]].to_string(index=False))


if __name__ == "__main__":
    main()
