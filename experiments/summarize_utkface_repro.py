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
GAP_THR = 0.02


def _metric(block, which: str, name: str) -> float:
    if not isinstance(block, dict):
        return float("nan")
    sub = block.get(which) or block
    if not isinstance(sub, dict):
        return float("nan")
    if name == "dp":
        return float(sub.get("dp_violation", sub.get("dp_clean", float("nan"))))
    if name == "acc":
        return float(sub.get("accuracy", sub.get("acc", float("nan"))))
    if name == "if":
        return float(sub.get("if_violation", sub.get("if_clean", float("nan"))))
    return float("nan")


def _group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r.get("attack"), float(r["alpha"]))].append(r)
    return g


def _mean(rs, method, which, name):
    vals = [_metric(r[method], which, name) for r in rs if method in r]
    return float(np.nanmean(vals)) if vals else float("nan")


def _matched_mac(mr, gr):
    """Keep only Mac rows whose seeds appear on GPU (fair partial-cell means)."""
    seeds = {int(r["seed"]) for r in gr}
    return [r for r in mr if int(r["seed"]) in seeds] or mr


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
        "Cell means use **seed-matched** Mac rows only when GPU is partial.",
        "",
        "## Clean test (primary)",
        "",
        "| attack | α | n_mac | n_gpu | Δ mean DP_dro (gpu−mac) | Δ mean acc_dro | note |",
        "|--------|---:|------:|------:|------------------------:|---------------:|------|",
    ]
    def _fmt_delta(x: float) -> str:
        """Empty-GPU / NaN cells as em-dash (not '+nan')."""
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "—"
        return f"{x:+.4f}"

    mg, gg = _group(mac), _group(gpu)
    keys = sorted(set(mg) | set(gg), key=lambda x: (str(x[0]), x[1]))
    gap_cells = []
    for atk, a in keys:
        mr, gr = mg.get((atk, a), []), gg.get((atk, a), [])
        if not gr:
            lines.append(
                f"| {atk} | {a} | {len(mr)} | 0 | — | — | partial |"
            )
            continue
        mr_m = _matched_mac(mr, gr)
        ddp = _mean(gr, "dro", "clean", "dp") - _mean(mr_m, "dro", "clean", "dp")
        dacc = _mean(gr, "dro", "clean", "acc") - _mean(mr_m, "dro", "clean", "acc")
        if abs(ddp) < GAP_THR and abs(dacc) < GAP_THR:
            note = "OK"
        elif len(gr) >= 6 and len(mr) >= 6:
            note = "GAP"
            gap_cells.append((atk, a, ddp, dacc))
        else:
            note = "partial"
        lines.append(
            f"| {atk} | {a} | {len(mr)} | {len(gr)} | {_fmt_delta(ddp)} | {_fmt_delta(dacc)} | {note} |"
        )

    lines += [
        "",
        "## Corrupted (attacked) test",
        "",
        "| attack | α | n_gpu | Δ mean DP_dro_corr | Δ mean acc_dro_corr | note |",
        "|--------|---:|------:|-------------------:|--------------------:|------|",
    ]
    for atk, a in keys:
        mr, gr = mg.get((atk, a), []), gg.get((atk, a), [])
        if not gr:
            continue
        mr_m = _matched_mac(mr, gr)
        ddp = _mean(gr, "dro", "corrupted", "dp") - _mean(mr_m, "dro", "corrupted", "dp")
        dacc = _mean(gr, "dro", "corrupted", "acc") - _mean(mr_m, "dro", "corrupted", "acc")
        note = "OK" if abs(ddp) < GAP_THR and abs(dacc) < GAP_THR else (
            "GAP" if len(gr) >= 6 and len(mr) >= 6 else "partial"
        )
        lines.append(
            f"| {atk} | {a} | {len(gr)} | {_fmt_delta(ddp)} | {_fmt_delta(dacc)} | {note} |"
        )

    # Seed-wise max abs delta over matched cells
    mac_i = {(r["attack"], float(r["alpha"]), int(r["seed"])): r for r in mac}
    gpu_i = {(r.get("attack"), float(r["alpha"]), int(r["seed"])): r for r in gpu}
    common = sorted(set(mac_i) & set(gpu_i))
    if common:
        d_clean = [
            abs(
                _metric(gpu_i[k]["dro"], "clean", "dp")
                - _metric(mac_i[k]["dro"], "clean", "dp")
            )
            for k in common
        ]
        d_corr = [
            abs(
                _metric(gpu_i[k]["dro"], "corrupted", "dp")
                - _metric(mac_i[k]["dro"], "corrupted", "dp")
            )
            for k in common
        ]
        signed_clean = [
            (
                abs(
                    _metric(gpu_i[k]["dro"], "clean", "dp")
                    - _metric(mac_i[k]["dro"], "clean", "dp")
                ),
                k,
                _metric(gpu_i[k]["dro"], "clean", "dp"),
                _metric(mac_i[k]["dro"], "clean", "dp"),
            )
            for k in common
        ]
        signed_clean.sort(reverse=True)
        top = signed_clean[:5]
        top_lines = [
            f"  - {atk} α={a} s={s}: gpu={g:.4f} mac={m:.4f} |Δ|={d:.4f}"
            for d, (atk, a, s), g, m in top
        ]
        lines += [
            "",
            "## Matched seed-wise (all completed GPU cells)",
            f"- Matched cells: **{len(common)}**",
            f"- max\\|Δ DP_dro clean\\| = **{max(d_clean):.4f}**",
            f"- max\\|Δ DP_dro corrupted\\| = **{max(d_corr):.4f}**",
            f"- mean Δ DP_dro clean = "
            f"{float(np.mean([_metric(gpu_i[k]['dro'],'clean','dp')-_metric(mac_i[k]['dro'],'clean','dp') for k in common])):+.5f}",
            "- Largest clean DP deltas (honest outliers, still OK if < thr):",
            *top_lines,
        ]

    lines += [
        "",
        "### Verdict",
        f"- Grid complete on both: **{len(mac) == 90 and len(gpu) == 90}**",
        f"- GAP cells (clean, n≥6 both sides, thr={GAP_THR}): **{len(gap_cells)}**"
        + (f" — {gap_cells}" if gap_cells else ""),
        "- If any cell is GAP with n=6 both sides, investigate device/nondeterminism before claiming CUDA repro.",
        "",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT, "mac", len(mac), "gpu", len(gpu), "matched", len(common) if common else 0)


if __name__ == "__main__":
    main()
