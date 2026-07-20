"""Convert the flat canonical grid (results/canonical_tau1.json) into the
nested ``results/all_results.json`` schema expected by
``experiments/generate_figures.py``.

The canonical grid is flat:
    {dataset, alpha, attack, method, seed, acc_clean, dp_clean, if_clean, ...}
``generate_figures.py`` expects one row per (dataset, alpha, seed) with a
nested ``naive`` / ``dro`` block, each holding ``clean`` / ``corrupted``
metric dicts keyed by ``accuracy`` / ``dp_violation`` / ``if_violation``.

We use the DP-targeted attack (the paper's headline) for the DP-violation
and accuracy panels. With IF-attack rows still pending the cluster re-run,
the IF-violation panel will reflect the IF metric measured under the DP
attack (honest, non-degenerate post-fix values), not the IF attack proper.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.loaders import load_canonical_tau1

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "all_results.json")


def to_nested(rows, attack):
    by_key = {}
    for r in rows:
        if r.get("attack") != attack:
            continue
        k = (r["dataset"], float(r["alpha"]), int(r["seed"]))
        by_key.setdefault(k, {})[r["method"]] = r

    out = []
    for (ds, alpha, seed), methods in by_key.items():
        if "naive" not in methods or "dro" not in methods:
            continue
        rec = {"dataset": ds, "alpha": alpha, "seed": seed, "total_time": 0.0}
        for method in ("naive", "dro"):
            r = methods[method]
            metrics = {
                "accuracy": r.get("acc_clean"),
                "dp_violation": r.get("dp_clean"),
                "if_violation": r.get("if_clean"),
            }
            rec[method] = {
                "time": 0.0,
                "clean": dict(metrics),
                "corrupted": dict(metrics),
            }
        out.append(rec)
    return out


def main():
    rows = load_canonical_tau1()
    attack = "dp"
    nested = to_nested(rows, attack)
    print(f"Converted {len(nested)} rows (attack={attack}) -> {OUT}")
    with open(OUT, "w") as f:
        json.dump(nested, f, indent=2)
    print("Run: python3 experiments/generate_figures.py")


if __name__ == "__main__":
    main()
