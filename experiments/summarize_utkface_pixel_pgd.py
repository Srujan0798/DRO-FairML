#!/usr/bin/env python3
"""Summarize U3 pixel-space PGD results + optional feature-space (U1) contrast.

U3 trains on train-set pixel PGD corruptions, then reports **clean test** metrics.
U1 feature-space rows use FairnessTargetedPGD on cached ResNet features and report
clean + corrupted test. Do not treat clean-test DP as the same attack-strength
number across the two protocols.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "results" / "utkface_pixel_pgd.json"
U1 = ROOT / "results" / "utkface_flair2.json"
OUT = ROOT / "results" / "pixel_pgd_summary.md"
TARGET = 12  # 6 seeds × 2 alphas (each row has both Naive and DRO)
TIE_EPS = 1e-5


def _winner(n: float, d: float) -> str:
    if abs(n - d) < TIE_EPS:
        return "tie"
    return "DRO" if d < n else "Naive"


def main() -> None:
    if not INP.exists():
        print("missing", INP)
        return
    rows = json.loads(INP.read_text())
    complete = len(rows) >= TARGET
    title = "UTKFace pixel-space PGD (U3)" + ("" if complete else " — PARTIAL")
    lines = [
        f"# {title}",
        "",
        f"rows: **{len(rows)}/{TARGET}**",
        "",
        "Protocol: train on **pixel** PGD (ε=4/255, steps=10) over raw UTKFace JPEGs; "
        "eval clean test DP/IF/acc. τ=1, k_inner=10, epochs=60. "
        "data_provenance=REAL_PIXELS.",
        "",
        "wins = DRO strict lower DP (ties if |N−D| < 1e-5).",
        "",
        "| α | n | DP N | DP D | wins_DP (D/tie/n) | mean ΔDP | wins_IF (D/tie/n) | IF N | IF D | acc N | acc D |",
        "|---:|--:|-----:|-----:|-----------------:|---------:|-----------------:|-----:|-----:|------:|------:|",
    ]
    by_a: dict[float, list] = defaultdict(list)
    for r in rows:
        by_a[float(r["alpha"])].append(r)

    for a in sorted(by_a):
        cell = by_a[a]
        nd = [r["naive"]["dp_violation"] for r in cell]
        dd = [r["dro"]["dp_violation"] for r in cell]
        na = [r["naive"]["accuracy"] for r in cell]
        da = [r["dro"]["accuracy"] for r in cell]
        ni = [r["naive"]["if_violation"] for r in cell]
        di = [r["dro"]["if_violation"] for r in cell]
        wins = sum(1 for n, d in zip(nd, dd) if _winner(n, d) == "DRO")
        ties = sum(1 for n, d in zip(nd, dd) if _winner(n, d) == "tie")
        iw = sum(1 for n, d in zip(ni, di) if _winner(n, d) == "DRO")
        it = sum(1 for n, d in zip(ni, di) if _winner(n, d) == "tie")
        lines.append(
            f"| {a} | {len(cell)} | {mean(nd):.4f} | {mean(dd):.4f} | "
            f"{wins}/{ties}/{len(cell)} | {mean(n - d for n, d in zip(nd, dd)):+.4f} | "
            f"{iw}/{it}/{len(cell)} | {mean(ni):.4f} | {mean(di):.4f} | "
            f"{mean(na):.4f} | {mean(da):.4f} |"
        )

    # Per-seed for highest α present
    if by_a:
        a_hi = max(by_a)
        cell = sorted(by_a[a_hi], key=lambda r: int(r["seed"]))
        lines += ["", f"### Per-seed @ α={a_hi} (n={len(cell)})", ""]
        for r in cell:
            n, d = r["naive"]["dp_violation"], r["dro"]["dp_violation"]
            lines.append(
                f"- s{int(r['seed'])}: DP N={n:.4f} D={d:.4f} → **{_winner(n, d)}** "
                f"(acc N={r['naive']['accuracy']:.3f} D={r['dro']['accuracy']:.3f}; "
                f"t={r.get('total_time', 0):.0f}s)"
            )

    # Honest contrast to U1 feature-space clean DP (not same attack)
    lines += [
        "",
        "### Contrast to U1 feature-space (clean test DP means)",
        "",
        "Not apples-to-apples: U1 corrupts **cached 512-d features** via FairnessTargetedPGD; "
        "U3 corrupts **pixels** then re-extracts features. Both report clean-test DP after train-time attack.",
        "",
        "| α | U3 n | U3 DP N/D | U1-dp n | U1 clean DP N/D | U1 corr DP N/D |",
        "|---:|-----:|----------:|--------:|----------------:|---------------:|",
    ]
    u1 = json.loads(U1.read_text()) if U1.exists() else []
    u1_dp = [
        r
        for r in u1
        if r.get("attack") == "dp"
    ]
    for a in sorted(set(by_a) | {0.1, 0.2}):
        c3 = by_a.get(a, [])
        c1 = [r for r in u1_dp if abs(float(r["alpha"]) - a) < 1e-9]
        if not c3 and not c1:
            continue
        u3s = (
            f"{mean(r['naive']['dp_violation'] for r in c3):.4f}/"
            f"{mean(r['dro']['dp_violation'] for r in c3):.4f}"
            if c3
            else "—"
        )
        u1c = (
            f"{mean(r['naive']['clean']['dp_violation'] for r in c1):.4f}/"
            f"{mean(r['dro']['clean']['dp_violation'] for r in c1):.4f}"
            if c1
            else "—"
        )
        u1k = (
            f"{mean(r['naive']['corrupted']['dp_violation'] for r in c1):.4f}/"
            f"{mean(r['dro']['corrupted']['dp_violation'] for r in c1):.4f}"
            if c1
            else "—"
        )
        lines.append(
            f"| {a} | {len(c3)} | {u3s} | {len(c1)} | {u1c} | {u1k} |"
        )

    lines += ["", f"device=cuda flair2. target {TARGET} cells (6 seeds × α∈{{0.1,0.2}} — each row has Naive+DRO)."]
    if not complete:
        lines.append(f"**PARTIAL** — not for paper claims until {TARGET}/{TARGET}.")
    else:
        lines.append("Grid complete. Human review before paper integration.")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT, "rows", len(rows))


if __name__ == "__main__":
    main()
