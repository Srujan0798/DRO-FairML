#!/usr/bin/env python3
"""Agent A5 — Empirical radii summary (analysis-only).

Reads results/empirical_radii.json (Adult, attack='dp', 5 α × 6 seeds × 2
methods × 3 arms = 180 configs). Arms:
  (uniform, uncoordinated) = canonical reference
  (uniform, coordinated)
  (empirical, coordinated)

Per (α, radii_mode, coordinated): DP, IF, accuracy for DRO. The clean
comparison is empirical+coordinated vs uniform+coordinated — SAME corrupted
data (coordinated=True), different radius calibration (empirical vs
uniform). This isolates the value of attack-aware radius calibration.

Also refreshes paper/sections/appendix_q5_empirical.tex with the actual
comparison numbers if data exists.

Outputs:
    results/empirical_radii_summary.md
    paper/sections/appendix_q5_empirical.tex  (refreshed if data exists)

Run:
    PYTHONPATH=. python3 experiments/summarize_a5_empirical.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
PAPER_DIR = os.path.join(ROOT, "paper", "sections")

ABLATION_PATH = os.path.join(RESULTS_DIR, "empirical_radii.json")
OUT_MD = os.path.join(RESULTS_DIR, "empirical_radii_summary.md")
APPENDIX_TEX = os.path.join(PAPER_DIR, "appendix_q5_empirical.tex")

ALPHA_LEVEL = 0.05
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
ARMS = [("uniform", False), ("uniform", True), ("empirical", True)]
# The clean comparison arm pair (same corrupted data; different radius calibration).
COMPARE_ARMS = [("uniform", True), ("empirical", True)]


def _expected_total():
    """5 alpha x 6 seeds x 2 methods x 3 arms = 180."""
    return 5 * 6 * 2 * 3


def _load_ablation():
    if not os.path.exists(ABLATION_PATH):
        return [], f"MISSING: {ABLATION_PATH}"
    with open(ABLATION_PATH) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{ABLATION_PATH} is not a JSON list")
    return rows, f"results/empirical_radii.json ({len(rows)} rows)"


def _mean(rows, key):
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def _per_cell_means(rows):
    """(alpha, radii_mode, coordinated, method) -> {dp, if, acc, n}."""
    g = defaultdict(list)
    for r in rows:
        g[(float(r["alpha"]), r.get("radii_mode", "uniform"),
           bool(r.get("coordinated", False)), r["method"])].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = {
            "dp": _mean(rs, "dp_clean"),
            "if": _mean(rs, "if_clean"),
            "acc": _mean(rs, "acc_clean"),
            "n": len(rs),
        }
    return out


def _wilcoxon_greater(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0.0):
        return 1.0
    try:
        _, p = wilcoxon(d, alternative="greater", zero_method="wilcox")
        return float(p)
    except ValueError:
        return 1.0


def _paired_comparison(rows, alpha):
    """empirical+coordinated vs uniform+coordinated for DRO at a given α.
    H1: dp_uniform > dp_empirical (empirical radii LOWER DP — better)."""
    df = pd.DataFrame(rows)
    sub = df[(df["alpha"].astype(float) == float(alpha)) & (df["method"] == "dro")]
    if sub.empty:
        return None
    uni = sub[(sub["radii_mode"] == "uniform") & (sub["coordinated"] == True)].set_index("seed")
    emp = sub[(sub["radii_mode"] == "empirical") & (sub["coordinated"] == True)].set_index("seed")
    if uni.empty or emp.empty:
        return None
    merged = uni[["dp_clean", "if_clean", "acc_clean"]].join(
        emp[["dp_clean", "if_clean", "acc_clean"]], lsuffix="_uni", rsuffix="_emp", how="inner"
    )
    n = len(merged)
    if n < 1:
        return None
    diff_dp = merged["dp_clean_uni"] - merged["dp_clean_emp"]  # >0 => empirical better
    diff_if = merged["if_clean_uni"] - merged["if_clean_emp"]
    diff_acc = merged["acc_clean_uni"] - merged["acc_clean_emp"]
    p_dp = _wilcoxon_greater(diff_dp) if n >= 2 else 1.0
    return {
        "alpha": float(alpha),
        "n_pairs": int(n),
        "dp_uniform": float(merged["dp_clean_uni"].mean()),
        "dp_empirical": float(merged["dp_clean_emp"].mean()),
        "dp_diff_uniform_minus_empirical": float(diff_dp.mean()),
        "p_value_dp": float(p_dp),
        "wins_empirical_dp": int((diff_dp > 0).sum()),
        "if_uniform": float(merged["if_clean_uni"].mean()),
        "if_empirical": float(merged["if_clean_emp"].mean()),
        "if_diff_uniform_minus_empirical": float(diff_if.mean()),
        "acc_uniform": float(merged["acc_clean_uni"].mean()),
        "acc_empirical": float(merged["acc_clean_emp"].mean()),
        "acc_diff_uniform_minus_empirical": float(diff_acc.mean()),
    }


def write_md(rows, source, n_expected, means, comparisons):
    lines = []
    lines.append("# Agent A5 — Empirical radii summary (Adult, DP attack)")
    lines.append("")
    lines.append("Analysis-only. No new training. Source: `results/empirical_radii.json` ")
    lines.append(f"({len(rows)}/{n_expected} rows). Arms: (uniform,uncoordinated)=canonical, ")
    lines.append("(uniform,coordinated), (empirical,coordinated).")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Empirical-radii rows present: **{len(rows)}/{n_expected}** "
                 f"({100.0*len(rows)/n_expected:.1f}%)" if n_expected else "- no rows")
    if len(rows) < n_expected:
        lines.append("- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).")
    lines.append("")
    lines.append("## Per-cell means for DRO (α, radii_mode, coordinated)")
    lines.append("")
    lines.append("| α | radii_mode | coordinated | n | DP | IF | acc |")
    lines.append("|---|---|---|---|---|---|---|")
    for alpha in ALPHAS:
        for (mode, coord) in ARMS:
            m = means.get((alpha, mode, coord, "dro"))
            if m is None:
                lines.append(f"| {alpha:.1f} | {mode} | {coord} | 0 | — | — | — |")
                continue
            lines.append(
                f"| {alpha:.1f} | {mode} | {str(coord)} | {m['n']} "
                f"| {m['dp']:.4f} | {m['if']:.4f} | {m['acc']:.4f} |"
            )
    lines.append("")
    lines.append("## Clean comparison: empirical+coordinated vs uniform+coordinated (DRO)")
    lines.append("")
    lines.append("Same corrupted data (coordinated=True), different radius calibration. "
                 "H1 (Wilcoxon on DP): dp_uniform > dp_empirical (empirical radii lower DP = better). "
                 "* marks p<0.05.")
    lines.append("")
    if not comparisons:
        lines.append("_(no paired rows yet — need both uniform+coord and empirical+coord "
                     "DRO rows for the same seed)_")
    else:
        lines.append("| α | n | DP_uni | DP_emp | ΔDP(uni-emp) | wins_emp | p_DP | IF_uni | IF_emp | ΔIF | acc_uni | acc_emp | Δacc |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in comparisons:
            if c is None:
                continue
            sig = "*" if c["p_value_dp"] < ALPHA_LEVEL else ""
            lines.append(
                f"| {c['alpha']:.1f} | {c['n_pairs']} "
                f"| {c['dp_uniform']:.4f} | {c['dp_empirical']:.4f} "
                f"| {c['dp_diff_uniform_minus_empirical']:+.4f} "
                f"| {c['wins_empirical_dp']}/{c['n_pairs']} "
                f"| {c['p_value_dp']:.4f} {sig} "
                f"| {c['if_uniform']:.4f} | {c['if_empirical']:.4f} "
                f"| {c['if_diff_uniform_minus_empirical']:+.4f} "
                f"| {c['acc_uniform']:.4f} | {c['acc_empirical']:.4f} "
                f"| {c['acc_diff_uniform_minus_empirical']:+.4f} |"
            )
    lines.append("")
    # Verdict.
    valid = [c for c in comparisons if c is not None and c["n_pairs"] >= 2]
    if not valid:
        verdict = ("Not yet answerable — need seed-paired uniform+coord and "
                   "empirical+coord DRO rows (currently INCOMPLETE).")
    else:
        sig_cells = [c for c in valid if c["p_value_dp"] < ALPHA_LEVEL]
        better_mean = [c for c in valid if c["dp_diff_uniform_minus_empirical"] > 0]
        n = len(valid)
        if sig_cells and len(better_mean) == n:
            verdict = (f"Yes: empirical+coordinated gives significantly lower DP than "
                       f"uniform+coordinated in {len(sig_cells)}/{n} cells (mean ΔDP "
                       f"= {np.mean([c['dp_diff_uniform_minus_empirical'] for c in valid]):+.4f}). "
                       "Attack-aware radius calibration improves DRO.")
        elif len(better_mean) == n and not sig_cells:
            verdict = (f"Directional yes: empirical+coordinated lowers DP in {len(better_mean)}/{n} "
                       "cells but does not reach p<0.05 (small n or small effect). "
                       "Attack-aware calibration helps weakly.")
        elif not len(better_mean):
            verdict = (f"No: empirical+coordinated does NOT lower DP in any cell "
                       f"(0/{n}); attack-aware radius calibration does not help under "
                       "this coordinated attack.")
        else:
            verdict = (f"Mixed: empirical+coordinated lowers DP in {len(better_mean)}/{n} cells "
                       f"({len(sig_cells)} significant). Attack-aware calibration helps "
                       "on some α but not uniformly.")
    lines.append("## Verdict — does attack-aware radius calibration improve DRO?")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines), verdict


def _refresh_appendix(comparisons, verdict):
    """Refresh paper/sections/appendix_q5_empirical.tex with comparison numbers
    if at least one paired cell exists. Otherwise leave the existing file alone."""
    valid = [c for c in comparisons if c is not None and c["n_pairs"] >= 2]
    if not valid:
        return False, "pilot framing retained (no paired cells yet)"

    a_lines = []
    a_lines.append(r"\subsection{Q5: Empirical Radii Under Coordinated Attacks}")
    a_lines.append(r"\label{app:empirical-radii}")
    a_lines.append("")
    a_lines.append(r"This appendix clarifies the calculation of uncertainty radii "
                   r"$\pi_{\mathrm{clean}}$ used in our DRO-FairML objective under different "
                   r"assumption modes, and reports the empirical comparison requested by "
                   r"Kuldeep (Q5).")
    a_lines.append("")
    a_lines.append(r"\paragraph{Uniform vs.\ Empirical Mode}")
    a_lines.append(r"In the \textit{uniform mode}, we assume no prior knowledge of the attack "
                   r"structure beyond the total corruption budget $\alpha$. The cleaned group "
                   r"proportions are calculated as $\pi_{\mathrm{clean}} = (\pi_{\mathrm{obs}} "
                   r"- \alpha)/(1 - 2\alpha)$.")
    a_lines.append("")
    a_lines.append(r"\paragraph{Coordinated Attack Structure}")
    a_lines.append(r"In the \textit{empirical mode}, we exploit the observation that "
                   r"real-world adversarial attacks are often coordinated, targeting specific "
                   r"demographic groups (70\% of the budget directed at the minority group). "
                   r"The empirical clean proportions are adjusted as $\pi_{\mathrm{clean}}[\min] "
                   r"= \pi_{\mathrm{obs}}[\min] + 0.4\alpha$, $\pi_{\mathrm{clean}}[\maj] = "
                   r"\pi_{\mathrm{obs}}[\maj] - 0.4\alpha$, clipped to $[0,1]$ and renormalised.")
    a_lines.append("")
    a_lines.append(r"\paragraph{Comparison (Adult, DP attack, $n{=}6$ seeds, DRO only)}")
    a_lines.append(r"Same corrupted data (\texttt{coordinated=True}); only the radius "
                   r"calibration differs. Source: \texttt{results/empirical\_radii.json}.")
    a_lines.append("")
    a_lines.append(r"\begin{center}")
    a_lines.append(r"\begin{tabular}{cccccccc}")
    a_lines.append(r"\toprule")
    a_lines.append(r"$\alpha$ & $n$ & DP (uniform) & DP (empirical) & $\Delta$DP & wins (emp) & $p$ & sig \\")
    a_lines.append(r"\midrule")
    for c in valid:
        sig = r"$^{*}$" if c["p_value_dp"] < ALPHA_LEVEL else ""
        a_lines.append(
            rf"{c['alpha']:.1f} & {c['n_pairs']} & {c['dp_uniform']:.4f} & "
            rf"{c['dp_empirical']:.4f} & {c['dp_diff_uniform_minus_empirical']:+.4f} & "
            rf"{c['wins_empirical_dp']}/{c['n_pairs']} & {c['p_value_dp']:.4f} & {sig} \\"
        )
    a_lines.append(r"\bottomrule")
    a_lines.append(r"\end{tabular}")
    a_lines.append(r"\end{center}")
    a_lines.append(r"$^{*}$ marks $p<0.05$ (one-sided Wilcoxon, H1: DP$_{\mathrm{uniform}} > $ "
                   r"DP$_{\mathrm{empirical}}$).")
    a_lines.append("")
    a_lines.append(r"\paragraph{Verdict}")
    a_lines.append(verdict.replace("_", r"\_"))
    a_lines.append("")
    a_lines.append(r"\paragraph{Implementation}")
    a_lines.append(r"This logic is implemented in \texttt{src/training/dro\_fair.py} within the "
                   r"\texttt{\_empirical\_pi\_clean()} method. The locked canonical grid uses "
                   r"\texttt{radii\_mode=uniform}; the empirical path is available for known "
                   r"attack structure (this appendix).")

    with open(APPENDIX_TEX, "w") as f:
        f.write("\n".join(a_lines) + "\n")
    return True, f"refreshed with {len(valid)} paired cells"


def main():
    print("AGENT A5: Empirical radii summary (analysis-only)")
    print("=" * 78)

    rows, source = _load_ablation()
    n_expected = _expected_total()
    print(f"Loaded: {source}")
    if len(rows) < n_expected:
        print(f"  INCOMPLETE: {len(rows)}/{n_expected} rows "
              f"({100.0*len(rows)/n_expected:.1f}%) — partial-data mode.")

    means = _per_cell_means(rows)
    comparisons = [_paired_comparison(rows, alpha) for alpha in ALPHAS]
    n_comp = sum(1 for c in comparisons if c is not None)
    print(f"Computed {n_comp} (α) comparison rows (empirical+coord vs uniform+coord, DRO).")

    md_text, verdict = write_md(rows, source, n_expected, means, comparisons)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print(f"  saved {OUT_MD}")

    ok, msg = _refresh_appendix(comparisons, verdict)
    if ok:
        print(f"  refreshed {APPENDIX_TEX} ({msg})")
    else:
        print(f"  {APPENDIX_TEX}: {msg}")

    print("\nVerdict:")
    print("  " + verdict)

    print("\n" + "=" * 78)
    print("AGENT A5 SUMMARY MILESTONE complete (analysis-only).")
    print(f"  output: {OUT_MD}")
    print("=" * 78)


if __name__ == "__main__":
    main()