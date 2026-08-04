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

# |N−D| below this is a numerical tie (not a real win either way).
TIE_EPS = 1e-5


def _winner(n: float, d: float) -> str:
    if abs(n - d) < TIE_EPS:
        return "tie"
    return "DRO" if d < n else "Naive"


def _win_counts(ns, ds):
    """Return (dro_wins, ties, n) for table cells."""
    labels = [_winner(n, d) for n, d in zip(ns, ds)]
    dro = sum(1 for w in labels if w == "DRO")
    ties = sum(1 for w in labels if w == "tie")
    return dro, ties, len(labels)


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
        "wins = DRO strict lower DP; ties if |N−D| < 1e-5 (not counted as wins).",
        "",
        "| α | n | DP_bin N | DP_bin D | wins_bin (D/tie/n) | DP_multi N | DP_multi D | wins_multi (D/tie/n) | mean Δmulti (N−D) |",
        "|---:|--:|---------:|---------:|------------------:|-----------:|-----------:|---------------------:|------------------:|",
    ]
    for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
        cell = [r for r in rows if abs(r["alpha"] - a) < 1e-9]
        if not cell:
            continue
        nb = [r["naive"]["dp_binary"] for r in cell]
        db = [r["dro"]["dp_binary"] for r in cell]
        nm = [r["naive"]["dp_multigroup"] for r in cell]
        dm = [r["dro"]["dp_multigroup"] for r in cell]
        wb, tb, nb_n = _win_counts(nb, db)
        wm, tm, nm_n = _win_counts(nm, dm)
        lines.append(
            f"| {a} | {len(cell)} | {mean(nb):.4f} | {mean(db):.4f} | "
            f"{wb}/{tb}/{nb_n} | "
            f"{mean(nm):.4f} | {mean(dm):.4f} | "
            f"{wm}/{tm}/{nm_n} | "
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

    # Per-seed multi wins for the highest incomplete/complete α (honest partial readout)
    for a in [0.4, 0.3, 0.2, 0.1, 0.0]:
        cell = sorted(
            [r for r in rows if abs(r["alpha"] - a) < 1e-9],
            key=lambda r: int(r["seed"]),
        )
        if not cell:
            continue
        lines += ["", f"### Per-seed multi @ α={a} (n={len(cell)})", ""]
        agree = disagree = 0
        for r in cell:
            n, d = r["naive"]["dp_multigroup"], r["dro"]["dp_multigroup"]
            win = _winner(n, d)
            bn, bd = r["naive"]["dp_binary"], r["dro"]["dp_binary"]
            bwin = _winner(bn, bd)
            if win == bwin:
                agree += 1
            else:
                disagree += 1
            lines.append(
                f"- s{int(r['seed'])}: multi N={n:.4f} D={d:.4f} → **{win}** "
                f"(bin N={bn:.4f} D={bd:.4f} → {bwin})"
            )
        lines.append(
            f"- bin↔multi winner agreement: **{agree}/{len(cell)}** "
            f"(disagreements={disagree}; multi max-min can flip vs binary White/non-White)"
        )
        break  # only the highest α present

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
