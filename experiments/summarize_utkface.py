#!/usr/bin/env python3
"""Summarize REAL UTKFace rows from results/utkface_canonical.json (no claims if incomplete)."""
import json, collections, sys
from statistics import mean
from pathlib import Path

path = Path("results/utkface_canonical.json")
if not path.exists():
    sys.exit("missing results/utkface_canonical.json")
rows = json.loads(path.read_text())
print(f"rows={len(rows)} target=90")
print("attacks", dict(collections.Counter(r.get("attack") for r in rows)))
print("provenance", dict(collections.Counter(r.get("data_provenance") for r in rows)))
if any(r.get("data_provenance") != "REAL" for r in rows):
    print("WARNING: non-REAL rows present")
# pair by (attack, alpha, seed)
g = collections.defaultdict(dict)
for r in rows:
    g[(r.get("attack"), float(r.get("alpha", -1)), int(r.get("seed", -1)))][r.get("method", "?")] = r
print("\nPer (attack, alpha) mean DP (paired seeds with both methods):")
for atk in ["dp", "if", "combined"]:
    for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
        pairs = [
            (v["naive"], v["dro"])
            for (ak, aa, s), v in g.items()
            if ak == atk and aa == a and "naive" in v and "dro" in v
        ]
        if not pairs:
            continue
        # keys may vary
        def dp(x):
            return x.get("dp_clean", x.get("dp", float("nan")))
        def acc(x):
            return x.get("acc_clean", x.get("acc", float("nan")))
        nv, dv = [dp(p[0]) for p in pairs], [dp(p[1]) for p in pairs]
        wins = sum(1 for n, d in zip(nv, dv) if n > d)
        print(
            f"  {atk:8} a={a}: n={len(pairs)} DPn={mean(nv):.4f} DPd={mean(dv):.4f} "
            f"winsDRO={wins}/{len(pairs)} accN={mean(acc(p[0]) for p in pairs):.3f} "
            f"accD={mean(acc(p[1]) for p in pairs):.3f}"
        )
if len(rows) < 90:
    print(f"\nINCOMPLETE ({len(rows)}/90) — do not put in paper as final.")
else:
    print("\nCOMPLETE 90/90 — ready for honest paper subsection review.")
