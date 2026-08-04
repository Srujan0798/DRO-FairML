"""Convert the flat canonical grid (results/canonical_tau1.json) into the
nested ``results/all_results.json`` schema used by
``experiments/generate_results.py`` (``make results``).

The canonical grid is flat:
    {dataset, alpha, attack, method, seed, acc_clean, dp_clean, if_clean, ...}
The nested schema is one row per (dataset, alpha, seed) with a
``naive`` / ``dro`` block, each holding ``clean`` / ``corrupted``
metric dicts keyed by ``accuracy`` / ``dp_violation`` / ``if_violation``.

We use the DP-targeted attack (the paper's headline) for the DP-violation
and accuracy panels. IF-attack cells live in ``canonical_tau1.json``; this
bridge still defaults to ``attack=dp`` for the nested table/plot path.
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
    print("Run: python3 main.py --generate-results  # or: make results")


if __name__ == "__main__":
    main()
