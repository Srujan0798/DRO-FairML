#!/usr/bin/env python3
"""Summarize REAL UTKFace grid (nested naive/dro schema); write results/utkface_summary.md."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from statistics import mean

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

path = Path("results/utkface_canonical.json")
out_md = Path("results/utkface_summary.md")
if not path.exists():
    sys.exit("missing results/utkface_canonical.json")

rows = json.loads(path.read_text())
atk_c = collections.Counter(r.get("attack") for r in rows)
prov = collections.Counter(r.get("data_provenance") for r in rows)
print(f"rows={len(rows)} target=90")
print("attacks", dict(atk_c))
print("provenance", dict(prov))

non_real = [r for r in rows if r.get("data_provenance") != "REAL"]
if non_real:
    print(f"WARNING: {len(non_real)} non-REAL rows")


def metrics(block: dict) -> tuple[float, float, float]:
    """Return (acc, dp, if) from nested clean block."""
    clean = block.get("clean") or block
    acc = float(clean.get("accuracy", clean.get("acc", float("nan"))))
    dp = float(clean.get("dp_violation", clean.get("dp_clean", clean.get("dp", float("nan")))))
    iff = float(clean.get("if_violation", clean.get("if_clean", clean.get("if", float("nan")))))
    return acc, dp, iff


lines = [
    "# UTKFace REAL summary",
    "",
    f"- rows: **{len(rows)}/90**",
    f"- attacks: {dict(atk_c)}",
    f"- provenance: {dict(prov)}",
    f"- all REAL: **{len(non_real) == 0}**",
    f"- protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, n_seeds=6",
    f"- task: gender prediction; protected = race White/non-White; N=23705 features",
    "",
    "| attack | α | n | DP naive | DP dro | wins DRO (↓DP) | p | acc N | acc D | IF n | IF d |",
    "|--------|---:|--:|---------:|-------:|---------------:|---:|------:|------:|------:|------:|",
]

for atk in ["dp", "if", "combined"]:
    for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
        cell = [
            r
            for r in rows
            if r.get("attack") == atk and float(r.get("alpha", -1)) == a and "naive" in r and "dro" in r
        ]
        if not cell:
            continue
        # one row per seed with both methods nested
        nv, dv, ni, di, an, ad = [], [], [], [], [], []
        for r in cell:
            acc_n, dp_n, if_n = metrics(r["naive"])
            acc_d, dp_d, if_d = metrics(r["dro"])
            an.append(acc_n)
            ad.append(acc_d)
            nv.append(dp_n)
            dv.append(dp_d)
            ni.append(if_n)
            di.append(if_d)
        wins = sum(1 for n, d in zip(nv, dv) if n > d)
        p = "n/a"
        if wilcoxon and len(cell) >= 6 and any(abs(n - d) > 1e-12 for n, d in zip(nv, dv)):
            try:
                p = f"{wilcoxon(nv, dv, alternative='greater').pvalue:.4f}"
            except Exception:
                p = "err"
        lines.append(
            f"| {atk} | {a} | {len(cell)} | {mean(nv):.4f} | {mean(dv):.4f} | "
            f"{wins}/{len(cell)} | {p} | {mean(an):.3f} | {mean(ad):.3f} | "
            f"{mean(ni):.4f} | {mean(di):.4f} |"
        )
        print(
            f"{atk:8} a={a}: n={len(cell)} DPn={mean(nv):.4f} DPd={mean(dv):.4f} "
            f"wins={wins}/{len(cell)} p={p} acc={mean(an):.3f}/{mean(ad):.3f}"
        )

if len(rows) < 90:
    lines += ["", f"**INCOMPLETE ({len(rows)}/90)** — do not claim as final paper result."]
    print(f"\nINCOMPLETE {len(rows)}/90")
else:
    lines += [
        "",
        "**COMPLETE 90/90 REAL.**",
        "",
        "### Honest read (do not overclaim)",
        "- This is an **image-feature** experiment (ResNet18), not a pixel-space attack.",
        "- Compare to tabular carefully; win pattern may **not** mirror Adult/Credit.",
        "- Use Wilcoxon p only when n=6 and report losses as well as wins.",
        "- Safe for paper only after human review of the table above.",
    ]
    print("\nCOMPLETE 90/90")

out_md.write_text("\n".join(lines) + "\n")
print("wrote", out_md)
