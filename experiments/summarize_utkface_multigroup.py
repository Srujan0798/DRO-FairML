#!/usr/bin/env python3
"""Summarize flair2 U2 results/utkface_multigroup.json → utkface_multigroup_summary.md."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "results" / "utkface_multigroup.json"
OUT = ROOT / "results" / "utkface_multigroup_summary.md"


def main():
    if not INP.exists():
        print("missing", INP)
        return
    rows = json.loads(INP.read_text())
    complete = len(rows) >= 30
    title = "UTKFace multi-group (5-race) summary" + ("" if complete else " — PARTIAL")
    lines = [
        f"# {title}",
        "",
        f"rows: **{len(rows)}/30**",
        "",
        "| α | n | DP_bin N | DP_bin D | wins_bin | DP_multi N | DP_multi D | wins_multi | mean Δmulti (N−D) |",
        "|---:|--:|---------:|---------:|---------:|-----------:|-----------:|-----------:|------------------:|",
    ]
    for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
        cell = [r for r in rows if abs(r["alpha"] - a) < 1e-9]
        if not cell:
            continue
        nb = [r["naive"]["dp_binary"] for r in cell]
        db = [r["dro"]["dp_binary"] for r in cell]
        nm = [r["naive"]["dp_multigroup"] for r in cell]
        dm = [r["dro"]["dp_multigroup"] for r in cell]
        lines.append(
            f"| {a} | {len(cell)} | {mean(nb):.4f} | {mean(db):.4f} | "
            f"{sum(n > d for n, d in zip(nb, db))}/{len(cell)} | "
            f"{mean(nm):.4f} | {mean(dm):.4f} | "
            f"{sum(n > d for n, d in zip(nm, dm))}/{len(cell)} | "
            f"{mean(n - d for n, d in zip(nm, dm)):+.4f} |"
        )

    lines += ["", "### DRO group positive rates (mean over seeds)", ""]
    for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
        cell = [r for r in rows if abs(r["alpha"] - a) < 1e-9]
        if not cell:
            continue
        rates = defaultdict(list)
        for r in cell:
            for g, v in r["dro"]["group_pos_rates"].items():
                rates[g].append(v)
        means = {g: mean(v) for g, v in rates.items()}
        gmax = max(means, key=means.get)
        gmin = min(means, key=means.get)
        lines.append(
            f"- α={a} n={len(cell)}: max **{gmax}** {means[gmax]:.3f} / "
            f"min **{gmin}** {means[gmin]:.3f} — "
            f"{ {k: round(v, 3) for k, v in means.items()} }"
        )

    lines += [
        "",
        "Protocol: train DP on binary race (White vs non-White); eval max-min DP on 5 race groups.",
        "REAL ResNet18 features. device=cuda flair2.",
    ]
    if not complete:
        lines.append("**PARTIAL** — not for paper claims until 30/30.")
    else:
        lines.append("Grid complete (30/30). Human review before paper integration.")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT, "rows", len(rows))


if __name__ == "__main__":
    main()
