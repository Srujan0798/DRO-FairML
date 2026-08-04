#!/usr/bin/env python3
"""Agent L2 — LSAC degeneracy fix summary (analysis-only).

Reads:
  - results/lsac_radii_fix.json  (arms b, c, d: 90 DRO + 30 naive = 120 configs)
  - results/canonical_tau1.json  (LSAC/dp rows = arm (a) reference + naive
    baseline for arm (b), which uses the canonical coordinated=False corruption)

Arms (LSAC only, attack='dp', 5 alpha x 6 seeds):
  (a) radii_mode='uniform', radii_clamp=None  = CANONICAL LSAC reference
      (pulled read-only from canonical_tau1.json; NOT re-run by L2).
  (b) radii_mode='uniform', radii_clamp=0.3   -> clamp rho_dp to max 0.3
      (coordinated=False, same corruption as canonical).
  (c) radii_mode='empirical', coordinated=True, radii_clamp=None
  (d) radii_mode='empirical', coordinated=True, radii_clamp=0.3

The naive baseline:
  - For arms (c,d) (coordinated=True): the 30 naive rows in lsac_radii_fix.json
    (run under coordinated=True so naive sees the SAME corrupted data as DRO).
  - For arm (b) (coordinated=False): the canonical LSAC naive rows
    (coordinated=False), pulled read-only from canonical_tau1.json.

VERDICT — does any arm UN-DEGENERATE LSAC?
  Degeneracy signature (diagnosed from canonical, docs/LSAC_DEGENERACY.md):
    - accuracy PINNED at the 0.9016 constant-predictor baseline across alpha.
    - DP FROZEN at ~0.222 for alpha in {0.2,0.3,0.4} (~0.183 at alpha=0).
  An arm UN-DEGENERATES if BOTH:
    (1) accuracy moves OFF the 0.9016 pin (mean abs deviation from 0.9016
        across alpha>=0.1 exceeds 0.01, i.e. >1pp on average — clearly above
        the canonical ~0.001-0.003 noise band), AND
    (2) DP UNFREEZES across alpha (the spread of mean DP over
        alpha in {0.2,0.3,0.4} exceeds 0.02, i.e. DP is no longer flat at 0.222;
        the canonical spread there is ~0.002).
  If any arm satisfies both, the paper UPGRADES LSAC from "degenerate, excluded"
  to "recovered by attack-aware radius calibration".
  If none do, the limitation stands WITH EVIDENCE instead of a hypothesis.

Outputs:
    results/lsac_radii_summary.md

Run:
    PYTHONPATH=. python3 experiments/summarize_l2.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS_DIR = os.path.join(ROOT, "results")

ABLATION_PATH = os.path.join(RESULTS_DIR, "lsac_radii_fix.json")
CANONICAL_PATH = os.path.join(RESULTS_DIR, "canonical_tau1.json")
OUT_MD = os.path.join(RESULTS_DIR, "lsac_radii_summary.md")

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
ALPHAS_DEGEN = [0.2, 0.3, 0.4]  # the alpha band where DP freezes in canonical

# Degeneracy thresholds (from the diagnosed canonical signature).
ACC_PIN = 0.9016  # LSAC constant-predictor baseline (loaders.constant_predictor_acc)
ACC_OFF_TOL = 0.01  # mean |acc - ACC_PIN| over alpha>=0.1 must exceed this to count as "off pin"
DP_FROZEN_SPREAD = 0.002  # canonical DP spread over {0.2,0.3,0.4} is ~0.002
DP_UNFROZEN_TOL = 0.02  # DP spread over {0.2,0.3,0.4} must exceed this to count as "unfrozen"

# Arm labels for display.
ARM_A = "(a) uniform, clamp=None [CANONICAL]"
ARM_B = "(b) uniform, clamp=0.3"
ARM_C = "(c) empirical, coord=True, clamp=None"
ARM_D = "(d) empirical, coord=True, clamp=0.3"


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a JSON list")
    return rows


def _mean(rows, key):
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def _arm_of(r):
    """Map a result row to one of the 4 arm labels.

    Canonical rows (no radii_clamp key, radii_mode='uniform', coordinated=False)
    map to arm (a). L2 rows are distinguished by (radii_mode, coordinated,
    radii_clamp).
    """
    mode = r.get("radii_mode", "uniform")
    coord = bool(r.get("coordinated", False))
    clamp = r.get("radii_clamp", None)
    if clamp is not None:
        clamp = float(clamp)
    if r.get("method") == "naive":
        return "naive"  # naive pooled per corruption setting; separated below
    if mode == "uniform" and not coord and clamp is None:
        return "a"
    if mode == "uniform" and not coord and clamp == 0.3:
        return "b"
    if mode == "empirical" and coord and clamp is None:
        return "c"
    if mode == "empirical" and coord and clamp == 0.3:
        return "d"
    return f"?mode={mode},coord={coord},clamp={clamp}"


def _cell_means(dro_rows):
    """arm -> {alpha -> {dp, acc, if, n}}."""
    g = defaultdict(lambda: defaultdict(list))
    for r in dro_rows:
        arm = _arm_of(r)
        if arm in ("naive", "?") or arm.startswith("?"):
            continue
        g[arm][float(r["alpha"])].append(r)
    out = {}
    for arm, by_alpha in g.items():
        out[arm] = {}
        for a, rs in by_alpha.items():
            out[arm][a] = {
                "dp": _mean(rs, "dp_clean"),
                "acc": _mean(rs, "acc_clean"),
                "if": _mean(rs, "if_clean"),
                "n": len(rs),
            }
    return out


def _naive_means(naive_rows):
    """naive comparator per (corruption setting, alpha).

    Returns {(coordinated, alpha): {dp, acc, if, n}}. The summarize step picks
    the matching corruption per arm:
      - arms (a), (b) use coordinated=False (canonical naive).
      - arms (c), (d) use coordinated=True (L2 naive).
    """
    g = defaultdict(list)
    for r in naive_rows:
        coord = bool(r.get("coordinated", False))
        g[(coord, float(r["alpha"]))].append(r)
    out = {}
    for (coord, a), rs in g.items():
        out[(coord, a)] = {
            "dp": _mean(rs, "dp_clean"),
            "acc": _mean(rs, "acc_clean"),
            "if": _mean(rs, "if_clean"),
            "n": len(rs),
        }
    return out


def _degeneracy_metrics(arm_cells):
    """Compute (acc_off_pin_meanabs, dp_spread_degen_band) for an arm.

    acc_off_pin_meanabs: mean over alpha>=0.1 of |mean_acc - ACC_PIN| (pp).
    dp_spread_degen_band: max-min of mean DP over alpha in {0.2,0.3,0.4}.
    """
    acc_devs = []
    for a in ALPHAS:
        if a < 0.1:
            continue
        cell = arm_cells.get(a)
        if cell is None or cell["n"] == 0 or np.isnan(cell["acc"]):
            continue
        acc_devs.append(abs(cell["acc"] - ACC_PIN))
    dps_degen = []
    for a in ALPHAS_DEGEN:
        cell = arm_cells.get(a)
        if cell is None or cell["n"] == 0 or np.isnan(cell["dp"]):
            continue
        dps_degen.append(cell["dp"])
    acc_off = float(np.mean(acc_devs)) if acc_devs else float("nan")
    dp_spread = (float(max(dps_degen) - min(dps_degen)) if len(dps_degen) >= 2
                 else float("nan"))
    return acc_off, dp_spread


def _verdict_for_arm(arm_cells):
    """Return (un_degenerates, reason) for a single arm."""
    acc_off, dp_spread = _degeneracy_metrics(arm_cells)
    # Need at least the degen-band alphas present to even evaluate.
    present_degen = sum(1 for a in ALPHAS_DEGEN if arm_cells.get(a) and arm_cells[a]["n"] > 0)
    if present_degen < 2:
        return None, ("insufficient data (need >=2 of alpha={0.2,0.3,0.4}; "
                      f"have {present_degen})")
    acc_ok = (not np.isnan(acc_off)) and acc_off > ACC_OFF_TOL
    dp_ok = (not np.isnan(dp_spread)) and dp_spread > DP_UNFROZEN_TOL
    parts = [f"acc_off_pin={acc_off:.4f} (>{ACC_OFF_TOL}? {acc_ok})",
             f"dp_spread_degen_band={dp_spread:.4f} (>{DP_UNFROZEN_TOL}? {dp_ok})"]
    un_deg = bool(acc_ok and dp_ok)
    return un_deg, "; ".join(parts)


def _fmt(x, p=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{p}f}"


def write_md(l2_rows, canon_rows, cells_by_arm, naive_by_corruption,
             verdicts, overall_verdict, coverage):
    L = []
    L.append("# Agent L2 — LSAC degeneracy fix: hypothesis test summary")
    L.append("")
    L.append("HYPOTHESIS: LSAC/DP is degenerate because the DRO radii formula "
             "`rho_dp[j] = alpha / ((1-alpha)*pi_clean[j] + alpha)` blows up on "
             "the ~90/10 imbalanced minority group (pi_clean[minority]=0.1 -> "
             "rho_min = 0.53..0.87, i.e. 2.0..4.8x the majority radius 0.11..0.43). "
             "The minority group is over-weighted, the classifier collapses to "
             "the majority class, accuracy pins at the 0.9016 constant-predictor "
             "baseline, and DP freezes at ~0.222 for alpha in {0.2,0.3,0.4}.")
    L.append("")
    L.append(f"FIX UNDER TEST: `radii_clamp=0.3` (chosen on PRINCIPLE before "
             f"running). 0.3 caps the minority radius at the majority-group "
             f"radius level (majority radius is 0.11..0.43; 0.3 sits near the "
             f"majority radius at alpha=0.3 = 0.32). It is the smallest cap that "
             f"brings the minority radius into the same order of magnitude as "
             f"the majority radius. NOT tuned — derived from the formula on "
             f"the diagnosed imbalance, before any L2 result.")
    L.append("")
    L.append("ARMS (LSAC, attack='dp', 5 alpha x 6 seeds):")
    L.append("- (a) uniform, clamp=None — CANONICAL reference (read-only from canonical_tau1.json)")
    L.append("- (b) uniform, clamp=0.3 (coordinated=False, same corruption as canonical)")
    L.append("- (c) empirical, coordinated=True, clamp=None")
    L.append("- (d) empirical, coordinated=True, clamp=0.3")
    L.append("")
    L.append("Naive baseline: arms (c,d) use L2 naive (coordinated=True, same "
             "corrupted data as DRO); arm (b) uses canonical naive "
             "(coordinated=False). Naive does not use radii.")
    L.append("")
    L.append("## Coverage")
    L.append("")
    L.append(f"- L2 rows present: **{len(l2_rows)}/120** "
             f"({100.0*len(l2_rows)/120:.1f}%)" if True else "")
    L.append(f"- Canonical LSAC/dp rows (arm a + naive-b): **{len(canon_rows)}** "
             f"(read-only; should be 60 = 5 alpha x 6 seeds x 2 methods)")
    if len(l2_rows) < 120:
        L.append("- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).")
    L.append("")

    # Per-arm DRO table.
    L.append("## Per (alpha, arm): DRO accuracy & DP")
    L.append("")
    L.append("Constant-predictor baseline (LSAC): **0.9016**. Canonical DP "
             "freezes at ~0.222 for alpha in {0.2,0.3,0.4}.")
    L.append("")
    L.append("| alpha | (a) canonical acc | (a) canonical DP | (b) clamp=0.3 acc | (b) clamp=0.3 DP | (c) empirical acc | (c) empirical DP | (d) emp+clamp acc | (d) emp+clamp DP |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for a in ALPHAS:
        row = [f"{a:.1f}"]
        for arm in ["a", "b", "c", "d"]:
            c = cells_by_arm.get(arm, {}).get(a)
            if c is None or c["n"] == 0:
                row += ["—", "—"]
            else:
                row += [f"{c['acc']:.4f} (n={c['n']})", f"{c['dp']:.4f}"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # Degeneracy metrics per arm.
    L.append("## Degeneracy metrics per arm (DRO)")
    L.append("")
    L.append(f"Thresholds: accuracy is 'off pin' if mean |acc - 0.9016| over "
             f"alpha>=0.1 > {ACC_OFF_TOL:.3f}; DP is 'unfrozen' if spread over "
             f"alpha={{0.2,0.3,0.4}} > {DP_UNFROZEN_TOL:.3f} (canonical spread "
             f"~{DP_FROZEN_SPREAD:.3f}).")
    L.append("")
    L.append("| arm | acc_off_pin (mean abs pp) | dp_spread {0.2,0.3,0.4} | acc off pin? | DP unfrozen? |")
    L.append("|---|---|---|---|---|")
    for arm in ["a", "b", "c", "d"]:
        cells = cells_by_arm.get(arm, {})
        acc_off, dp_spread = _degeneracy_metrics(cells)
        acc_ok = (not np.isnan(acc_off)) and acc_off > ACC_OFF_TOL
        dp_ok = (not np.isnan(dp_spread)) and dp_spread > DP_UNFROZEN_TOL
        L.append(f"| {arm} | {_fmt(acc_off)} | {_fmt(dp_spread)} | "
                 f"{'YES' if acc_ok else 'no'} | {'YES' if dp_ok else 'no'} |")
    L.append("")

    # Verdict per arm.
    L.append("## Verdict per arm — does this arm un-degenerate LSAC?")
    L.append("")
    L.append("An arm UN-DEGENERATES LSAC if BOTH: accuracy moves off the 0.9016 "
             "pin AND DP unfreezes across alpha in {0.2,0.3,0.4}.")
    L.append("")
    L.append("| arm | un-degenerates? | evidence |")
    L.append("|---|---|---|")
    for arm in ["a", "b", "c", "d"]:
        ud, reason = verdicts.get(arm, (None, "no data"))
        if ud is None:
            mark = "—"
        else:
            mark = "**YES**" if ud else "no"
        L.append(f"| {arm} | {mark} | {reason} |")
    L.append("")

    # Overall verdict.
    L.append("## Overall verdict")
    L.append("")
    L.append(overall_verdict)
    L.append("")

    # Per-arm narrative.
    L.append("## Arm-by-arm reading")
    L.append("")
    arm_names = {"a": ARM_A, "b": ARM_B, "c": ARM_C, "d": ARM_D}
    for arm in ["a", "b", "c", "d"]:
        cells = cells_by_arm.get(arm, {})
        ud, reason = verdicts.get(arm, (None, "no data"))
        L.append(f"### {arm_names[arm]}")
        L.append("")
        if not cells:
            L.append("- No data yet (INCOMPLETE).")
            L.append("")
            continue
        # accuracy at each alpha
        accs = [(a, cells[a]["acc"]) for a in ALPHAS if a in cells and cells[a]["n"] > 0]
        dps = [(a, cells[a]["dp"]) for a in ALPHAS if a in cells and cells[a]["n"] > 0]
        if accs:
            acc_str = ", ".join(f"alpha={a}: {v:.4f}" for a, v in accs)
            L.append(f"- accuracy: {acc_str}")
        if dps:
            dp_str = ", ".join(f"alpha={a}: {v:.4f}" for a, v in dps)
            L.append(f"- DP: {dp_str}")
        if ud is None:
            L.append(f"- verdict: INCOMPLETE ({reason})")
        elif ud:
            L.append(f"- verdict: **UN-DEGENERATES** ({reason})")
        else:
            L.append(f"- verdict: still degenerate ({reason})")
        L.append("")

    # Provenance / clamp justification footer.
    L.append("## Provenance")
    L.append("")
    L.append(f"- Source (L2): `{ABLATION_PATH}` ({len(l2_rows)} rows)")
    L.append(f"- Source (canonical reference): `{CANONICAL_PATH}` ({len(canon_rows)} LSAC/dp rows)")
    L.append("- clamp=0.3 justification: LSAC minority radius blows up to "
             "0.53..0.87 (2.0..4.8x the majority radius 0.11..0.43) because "
             "pi_clean[minority]=0.1 shrinks the denominator. 0.3 caps the "
             "minority radius at the majority-group radius level, preventing "
             "minority over-weighting. Chosen on principle before running; "
             "not tuned-until-it-wins.")
    L.append("- All arms: tau=1.0, k_inner=10, epochs=60, pgd_steps=20, "
             "lambda_init=0.0, lr_lambda=5e-3, attack_k=5, 6 seeds.")
    return "\n".join(L)


def main():
    print("AGENT L2: LSAC degeneracy fix summary (analysis-only)")
    print("=" * 78)

    l2_rows = _load(ABLATION_PATH)
    print(f"Loaded: results/lsac_radii_fix.json ({len(l2_rows)}/120 rows)")
    if len(l2_rows) < 120:
        print(f"  INCOMPLETE: {len(l2_rows)}/120 rows "
              f"({100.0*len(l2_rows)/120:.1f}%) — partial-data mode.")

    try:
        from experiments.loaders import load_canonical_tau1
        canon_all = load_canonical_tau1()
    except Exception as e:
        print(f"  WARNING: could not load canonical: {e}")
        canon_all = []
    canon_rows = [r for r in canon_all
                  if r.get("dataset") == "lsac" and r.get("attack") == "dp"]
    print(f"Loaded: canonical LSAC/dp = {len(canon_rows)} rows (arm a + naive-b)")

    # Split L2 rows into DRO (arms b,c,d) and naive (coordinated=True).
    l2_dro = [r for r in l2_rows if r.get("method") == "dro"]
    l2_naive = [r for r in l2_rows if r.get("method") == "naive"]
    # Canonical DRO = arm (a); canonical naive = naive-b (coordinated=False).
    canon_dro = [r for r in canon_rows if r.get("method") == "dro"]
    canon_naive = [r for r in canon_rows if r.get("method") == "naive"]

    # Build per-arm DRO cells. Arm (a) comes from canonical; arms (b,c,d) from L2.
    cells = {}
    cells["a"] = {a: {"dp": _mean(rs, "dp_clean"), "acc": _mean(rs, "acc_clean"),
                     "if": _mean(rs, "if_clean"), "n": len(rs)}
                  for a, rs in _group_by_alpha(canon_dro).items()}
    l2_cells = _cell_means(l2_dro)
    for arm in ["b", "c", "d"]:
        cells[arm] = l2_cells.get(arm, {})
    # Defensive: flag any unexpected arm keys.
    extra = sorted(set(l2_cells) - {"a", "b", "c", "d"})
    if extra:
        print(f"  WARNING: unexpected arm labels in L2 data: {extra}")

    # Naive comparators: (coordinated=False) for arms a,b from canonical;
    # (coordinated=True) for arms c,d from L2 naive.
    naive_by = {}
    for (coord, a), rs in _group_naive(canon_naive).items():
        naive_by[(coord, a)] = {"dp": _mean(rs, "dp_clean"), "acc": _mean(rs, "acc_clean"),
                               "if": _mean(rs, "if_clean"), "n": len(rs)}
    for (coord, a), rs in _group_naive(l2_naive).items():
        naive_by[(coord, a)] = {"dp": _mean(rs, "dp_clean"), "acc": _mean(rs, "acc_clean"),
                               "if": _mean(rs, "if_clean"), "n": len(rs)}

    # Per-arm verdicts.
    verdicts = {arm: _verdict_for_arm(cells[arm]) for arm in ["a", "b", "c", "d"]}
    for arm in ["a", "b", "c", "d"]:
        ud, reason = verdicts[arm]
        mark = "UN-DEG" if ud else ("INCOMPLETE" if ud is None else "degenerate")
        print(f"  arm ({arm}): {mark} — {reason}")

    # Overall verdict: does ANY of the FIX arms (b, c, d) un-degenerate?
    fix_arms = ["b", "c", "d"]
    any_un_deg = any(verdicts[a][0] for a in fix_arms)
    incomplete = any(verdicts[a][0] is None for a in fix_arms)
    if any_un_deg:
        winners = [a for a in fix_arms if verdicts[a][0]]
        overall_verdict = (
            f"**YES — LSAC UN-DEGENERATES.** Fix arm(s) {', '.join(winners)} move "
            f"accuracy off the 0.9016 pin AND unfreeze DP across alpha. The paper "
            f"UPGRADES LSAC from 'degenerate, excluded' to 'recovered by "
            f"attack-aware radius calibration' — a genuine methodological "
            f"contribution. Report which arm wins (and whether it is the principled "
            f"clamp=0.3, the empirical radii, or both) in the paper text."
        )
    elif incomplete:
        overall_verdict = (
            "**INCOMPLETE.** Not all fix arms have enough data to evaluate yet. "
            "Re-run this summary as more L2 rows land (idempotent). No verdict "
            "can be honestly stated against an empty band."
        )
    else:
        overall_verdict = (
            "**NO — the limitation stands WITH EVIDENCE.** None of the fix arms "
            "(uniform+clamp=0.3, empirical, empirical+clamp=0.3) move accuracy "
            "off the 0.9016 constant-predictor pin AND unfreeze DP across alpha. "
            "The LSAC/DP degeneracy is NOT an artifact of the un-clamped minority "
            "radius that principled radius calibration fixes. The diagnosis "
            "(docs/LSAC_DEGENERACY.md) is confirmed, and the limitation is now "
            "supported by a tested fix rather than an untested hypothesis. "
            "LSAC/DP remains a degenerate/diagnostic result, NOT a DRO win or loss."
        )
    print("\nOverall verdict:")
    print("  " + overall_verdict.replace("\n", "\n  "))

    coverage = {"l2": len(l2_rows), "canon": len(canon_rows)}
    md = write_md(l2_rows, canon_rows, cells, naive_by, verdicts,
                  overall_verdict, coverage)
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(f"\nSaved: {OUT_MD}")
    print("=" * 78)
    print("AGENT L2 SUMMARY complete (analysis-only).")
    print("=" * 78)


def _group_by_alpha(rows):
    g = defaultdict(list)
    for r in rows:
        g[float(r["alpha"])].append(r)
    return g


def _group_naive(rows):
    g = defaultdict(list)
    for r in rows:
        coord = bool(r.get("coordinated", False))
        g[(coord, float(r["alpha"]))].append(r)
    return g


if __name__ == "__main__":
    main()