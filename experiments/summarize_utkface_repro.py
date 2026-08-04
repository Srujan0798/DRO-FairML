#!/usr/bin/env python3
"""Compare Mac MPS utkface_canonical.json vs flair2 CUDA utkface_flair2.json."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAC = ROOT / "results" / "utkface_canonical.json"
GPU = ROOT / "results" / "utkface_flair2.json"
OUT = ROOT / "results" / "utkface_reproducibility_summary.md"


def _clean_dp(block):
    if not isinstance(block, dict):
        return float("nan")
    c = block.get("clean") or block
    return float(c.get("dp_violation", c.get("dp_clean", float("nan"))))


def _clean_acc(block):
    c = block.get("clean") or block
    return float(c.get("accuracy", c.get("acc", float("nan"))))


def load_mac(rows):
    # nested naive/dro per seed
    g = defaultdict(list)
    for r in rows:
        g[(r["attack"], float(r["alpha"]))].append(r)
    return g


def load_gpu(rows):
    # same nested schema from run_utkface
    g = defaultdict(list)
    for r in rows:
        g[(r.get("attack"), float(r["alpha"]))].append(r)
    return g


def main():
    mac = json.loads(MAC.read_text()) if MAC.exists() else []
    gpu = json.loads(GPU.read_text()) if GPU.exists() else []
    lines = [
        "# UTKFace reproducibility: Mac MPS vs flair2 CUDA",
        "",
        f"- Mac rows: **{len(mac)}/90** (`results/utkface_canonical.json`)",
        f"- flair2 rows: **{len(gpu)}/90** (`results/utkface_flair2.json`)",
        "",
        "Protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, n_seeds=6, REAL features.",
        "Same seeds 0–5. Large gaps are bugs to investigate.",
        "",
        "| attack | α | n_mac | n_gpu | Δ mean DP_dro (gpu−mac) | Δ mean acc_dro | note |",
        "|--------|---:|------:|------:|------------------------:|---------------:|------|",
    ]
    mg, gg = load_mac(mac), load_gpu(gpu)
    keys = sorted(set(mg) | set(gg), key=lambda x: (str(x[0]), x[1]))
    for atk, a in keys:
        mr, gr = mg.get((atk, a), []), gg.get((atk, a), [])
        def mean_dp(rs, method):
            vals = []
            for r in rs:
                if method in r:
                    vals.append(_clean_dp(r[method]))
            return float(np.nanmean(vals)) if vals else float("nan")
        def mean_acc(rs, method):
            vals = []
            for r in rs:
                if method in r:
                    vals.append(_clean_acc(r[method]))
            return float(np.nanmean(vals)) if vals else float("nan")
        ddp = mean_dp(gr, "dro") - mean_dp(mr, "dro")
        dacc = mean_acc(gr, "dro") - mean_acc(mr, "dro")
        note = "OK" if abs(ddp) < 0.02 else ("GAP" if len(gr) >= 6 and len(mr) >= 6 else "partial")
        lines.append(
            f"| {atk} | {a} | {len(mr)} | {len(gr)} | {ddp:+.4f} | {dacc:+.4f} | {note} |"
        )
    lines += [
        "",
        "### Verdict",
        f"- Grid complete on both: **{len(mac)==90 and len(gpu)==90}**",
        "- If any cell is GAP with n=6 both sides, investigate device/nondeterminism before claiming CUDA repro.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print("wrote", OUT, "mac", len(mac), "gpu", len(gpu))


if __name__ == "__main__":
    main()
