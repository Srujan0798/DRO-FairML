#!/usr/bin/env python3
"""Summarize results/random_vs_adversarial.json (partial OK). Analysis only.

Compares FairnessTargetedPGD (adversarial) vs RandomCorruptor under the
canonical τ=1 protocol. Prints mean DP / acc and adversarial/random ratio
on clean DP for each (dataset, α, method) cell.

Usage:
  python3 experiments/summarize_rva.py
  python3 experiments/summarize_rva.py --write results/random_vs_adversarial_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results" / "random_vs_adversarial.json"


def load(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def _mean(rs, key):
    if not rs:
        return float("nan")
    return float(np.mean([x[key] for x in rs]))


def summarize(rows, source_label: str) -> str:
    g = defaultdict(list)
    for r in rows:
        key = (
            r["dataset"],
            float(r["alpha"]),
            r["method"],
            r.get("corruptor_type", "adversarial"),
        )
        g[key].append(r)

    lines = [
        "# Random vs adversarial (Wave-1 A4) — live summary",
        "",
        f"Source: `{source_label}` — **{len(rows)}** rows so far (target 144).",
        "Protocol: τ=1, K_inner=10, pgd_steps=20, n_seeds=6, attack=dp.",
        "Ratio = mean(DP_adversarial) / mean(DP_random) on clean test "
        "(higher ⇒ adversarial raises DP more than random).",
        "",
        "| dataset | α | method | n_adv | n_rnd | DP_adv | DP_rnd | ratio | acc_adv | acc_rnd |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    datasets = sorted({r["dataset"] for r in rows})
    alphas = sorted({float(r["alpha"]) for r in rows})
    methods = ["naive", "dro"]
    for ds in datasets:
        for a in alphas:
            for m in methods:
                adv = g.get((ds, a, m, "adversarial"), [])
                rnd = g.get((ds, a, m, "random"), [])
                if not adv and not rnd:
                    continue
                da, dr = _mean(adv, "dp_clean"), _mean(rnd, "dp_clean")
                if rnd and dr > 1e-12:
                    ratio_s = f"{da / dr:.2f}"
                else:
                    ratio_s = "—"
                lines.append(
                    f"| {ds} | {a:.1f} | {m} | {len(adv)} | {len(rnd)} | "
                    f"{da:.4f} | {dr:.4f} | {ratio_s} | "
                    f"{_mean(adv, 'acc_clean'):.3f} | {_mean(rnd, 'acc_clean'):.3f} |"
                )

    complete = 0
    for ds in ["adult", "credit", "lsac"]:
        for a in [0.1, 0.2]:
            for m in methods:
                if (
                    len(g.get((ds, a, m, "adversarial"), [])) >= 6
                    and len(g.get((ds, a, m, "random"), [])) >= 6
                ):
                    complete += 1
    lines += [
        "",
        f"Complete (dataset,α,method) cells with n≥6 both arms: **{complete}/12**.",
        "",
        "Do **not** put incomplete ratios in the paper abstract. Prefer full 144-row file.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT)
    ap.add_argument("--write", type=Path, default=None)
    args = ap.parse_args()
    rows = load(args.path)
    try:
        label = str(args.path.relative_to(ROOT))
    except ValueError:
        label = str(args.path)
    text = summarize(rows, label)
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text)
        print(f"Wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
