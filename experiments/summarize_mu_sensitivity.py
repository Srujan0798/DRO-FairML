#!/usr/bin/env python3
"""TASK C summary — 6-point mu-sensitivity curve + degeneracy analysis.

Combines:
  - results/mu_sensitivity.json (new runs: mu in {0.5,1,2,20}, Adult+Credit, alpha=0.2,
    naive+dro, 6 seeds)
  - results/aug_lagrangian.json (existing: mu in {5,10}, Adult+Credit, alpha=0.2, dro)
  - results/canonical_tau1.json (naive reference at mu=0, read-only)

Produces:
  - results/mu_sensitivity_summary.md (tables + verdict)
  - results/mu_sensitivity_curve.png    (DP-vs-mu + accuracy-vs-mu, both datasets)

Rules (pre-registered):
  C1 constant-predictor floor guard: any DP improvement at/below floor = DEGEN.
  C2 recommended mu = largest mu whose mean val accuracy >= floor + 0.02.
  C3 mark the degeneracy threshold: where each dataset's accuracy curve crosses the floor.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np

# Constant-predictor (majority-class) accuracy floor per dataset.
CONST_FLOOR = {'adult': 0.7521, 'credit': 0.7788}
SAFE_MARGIN = 0.02
SEEDS = [0, 1, 2, 3, 4, 5]
MU_POINTS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
DATASETS = ['adult', 'credit']


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else []


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r]
    return float(np.mean(vals)) if vals else float('nan')


def _std(rows, key):
    vals = [r[key] for r in rows if key in r]
    return float(np.std(vals, ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0


def main():
    mu_sens = _load('results/mu_sensitivity.json')
    aug_lag = _load('results/aug_lagrangian.json')
    canon = _load('results/canonical_tau1.json')

    # Naive reference at alpha=0.2: from canonical_tau1 (mu=0 by construction) OR
    # the new mu_sensitivity naive rows (which are mu-independent duplicates of
    # canonical naive). Use the new naive rows when present, else canonical.
    def naive_rows(ds, mu):
        rows = [r for r in mu_sens if r['dataset'] == ds and r['alpha'] == 0.2
                and r['method'] == 'naive' and r['aug_lagrangian_mu'] == mu]
        if rows:
            return rows
        return [r for r in canon if r['dataset'] == ds and r['alpha'] == 0.2
                and r['method'] == 'naive' and r.get('tau') == 1.0
                and r.get('attack') == 'dp']

    def dro_rows(ds, mu):
        # Primary: mu_sensitivity.json
        rows = [r for r in mu_sens if r['dataset'] == ds and r['alpha'] == 0.2
                and r['method'] == 'dro' and r['aug_lagrangian_mu'] == mu]
        if rows:
            return rows
        # Fallback: aug_lagrangian.json holds mu=5,10
        return [r for r in aug_lag if r['dataset'] == ds and r['alpha'] == 0.2
                and r['method'] == 'dro' and r['aug_lagrangian_mu'] == mu]

    L = ["# TASK C — mu sensitivity curve (pre-registered)",
         "",
         f"6-point curve: mu in {MU_POINTS}",
         f"Datasets: {DATASETS} | alpha=0.2 | 6 seeds | methods: naive + dro",
         f"Data: mu_sensitivity.json ({len(mu_sens)} rows) + "
         f"aug_lagrangian.json ({len(aug_lag)} rows, mu=5,10 reused)",
         ""]

    # --- DP vs mu table ---
    L += ["## DP violation vs mu (alpha=0.2, attack=dp)",
          "",
          "| mu | Adult naive | Adult DRO | Credit naive | Credit DRO |",
          "|---|---|---|---|---|"]
    curve = {ds: {'mu': [], 'dp_dro': [], 'dp_naive': [], 'acc_dro': [], 'acc_naive': []}
            for ds in DATASETS}
    for mu in MU_POINTS:
        bits = [f"{mu:g}"]
        for ds in DATASETS:
            nv = naive_rows(ds, mu)
            dr = dro_rows(ds, mu)
            dp_nv = _mean(nv, 'dp_clean')
            dp_dr = _mean(dr, 'dp_clean')
            acc_dr = _mean(dr, 'acc_clean')
            acc_nv = _mean(nv, 'acc_clean')
            bits.append(f"{dp_nv:.4f}")
            bits.append(f"{dp_dr:.4f}")
            curve[ds]['mu'].append(mu)
            curve[ds]['dp_dro'].append(dp_dr)
            curve[ds]['dp_naive'].append(dp_nv)
            curve[ds]['acc_dro'].append(acc_dr)
            curve[ds]['acc_naive'].append(acc_nv)
        L.append(f"| {' | '.join(bits)} |")

    # --- Accuracy vs mu table ---
    L += ["", "## Accuracy vs mu (alpha=0.2, attack=dp)",
          "",
          "| mu | Adult floor+0.02 | Adult naive | Adult DRO | Credit floor+0.02 | Credit naive | Credit DRO |",
          "|---|---|---|---|---|---|---|"]
    for i, mu in enumerate(MU_POINTS):
        bits = [f"{mu:g}"]
        for ds in DATASETS:
            floor_thr = CONST_FLOOR[ds] + SAFE_MARGIN
            acc_nv = curve[ds]['acc_naive'][i]
            acc_dr = curve[ds]['acc_dro'][i]
            bits.append(f"{floor_thr:.4f}")
            bits.append(f"{acc_nv:.4f}")
            bits.append(f"{acc_dr:.4f}")
        L.append(f"| {' | '.join(bits)} |")

    # --- Degeneracy threshold + recommendation ---
    L += ["", "## Degeneracy threshold and recommended mu",
          "",
          f"Rule C2: recommended mu = **largest mu whose mean DRO accuracy stays "
          f">= floor + {SAFE_MARGIN}**. Any DP improvement at/below the floor is "
          "DEGEN (collapse, not fairness).",
          "",
          "| dataset | floor | floor+0.02 | threshold mu* | DRO acc at mu* | verdict |",
          "|---|---|---|---|---|---|"]
    recommendations = {}
    for ds in DATASETS:
        floor = CONST_FLOOR[ds]
        thr = floor + SAFE_MARGIN
        # Scan from largest mu downward for the safe maximum.
        best_mu = None
        best_acc = None
        for mu in reversed(MU_POINTS):
            dr = dro_rows(ds, mu)
            if not dr:
                continue
            acc = _mean(dr, 'acc_clean')
            if acc >= thr:
                best_mu = mu
                best_acc = acc
                break
        if best_mu is None:
            # Even the smallest mu fails the safety margin: report the lowest-tested mu.
            lowest = MU_POINTS[0]
            dr = dro_rows(ds, lowest)
            acc = _mean(dr, 'acc_clean') if dr else float('nan')
            recommendations[ds] = dict(mu=lowest, acc=acc, safe=False)
            L.append(f"| {ds} | {floor:.4f} | {thr:.4f} | {lowest:g} | {acc:.4f} | "
                     f"**no mu reaches floor+{SAFE_MARGIN}; AL unsafe** |")
        else:
            recommendations[ds] = dict(mu=best_mu, acc=best_acc, safe=True)
            # Find threshold: smallest mu where acc crosses below floor (the collapse point).
            collapse_mu = None
            for mu in MU_POINTS:
                dr = dro_rows(ds, mu)
                if not dr:
                    continue
                if _mean(dr, 'acc_clean') <= floor:
                    collapse_mu = mu
                    break
            thr_str = f"mu>={collapse_mu:g}" if collapse_mu else "not reached in [0.5, 20]"
            L.append(f"| {ds} | {floor:.4f} | {thr:.4f} | {best_mu:g} | {best_acc:.4f} | "
                     f"threshold: collapse below floor at {thr_str} |")

    # --- Per-mu degeneracy flags ---
    L += ["", "## Per-mu degeneracy flags (DRO arm)",
          "",
          "Any (mu, dataset) where DRO mean accuracy <= floor is flagged **DEGEN**: the "
          "apparent DP improvement is constant-predictor collapse, not fairness.",
          "",
          "| mu |" + "".join(f" {ds} DRO acc | {ds} dp | {ds} floor | {ds} verdict |" for ds in DATASETS),
          "|---|" + "".join("---|---|---|---|" for _ in DATASETS)]
    for i, mu in enumerate(MU_POINTS):
        cells = [f"{mu:g}"]
        for ds in DATASETS:
            dr = dro_rows(ds, mu)
            if not dr:
                cells += ["—", "—", "—", "no data"]
                continue
            acc = _mean(dr, 'acc_clean')
            dp = _mean(dr, 'dp_clean')
            floor = CONST_FLOOR[ds]
            degen = acc <= floor
            cells.append(f"{acc:.4f}")
            cells.append(f"{dp:.4f}")
            cells.append(f"{floor:.4f}")
            cells.append("**DEGEN**" if degen else "ok")
        L.append(f"| {' | '.join(cells)} |")

    # --- Final stated recommendation ---
    L += ["", "## Recommendation",
          ""]
    for ds in DATASETS:
        rec = recommendations[ds]
        floor = CONST_FLOOR[ds]
        if rec['safe']:
            L.append(f"- **{ds}**: mu={rec['mu']:g} (largest tested mu with mean accuracy "
                     f"{rec['acc']:.4f} >= floor+0.02={floor + SAFE_MARGIN:.4f}).")
        else:
            L.append(f"- **{ds}**: **no safe mu** — even mu={rec['mu']:g} gives mean accuracy "
                     f"{rec['acc']:.4f} < floor+0.02={floor + SAFE_MARGIN:.4f}. Do not use AL on {ds}.")
    L.append("")
    L.append(f"Degeneracy guard enforced: any DP improvement at/below the floor "
             f"({', '.join(f'{ds}={CONST_FLOOR[ds]}' for ds in DATASETS)}) is labelled "
             "**DEGEN**, not a win.")

    # --- Write summary ---
    out = 'results/mu_sensitivity_summary.md'
    with open(out, 'w') as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plot_curve(curve, recommendations, CONST_FLOOR, SAFE_MARGIN)
    except Exception as e:
        print(f"[warn] plot skipped: {e}")


def _plot_curve(curve, recommendations, const_floor, safe_margin):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'naive': '#888888', 'dro': '#1f77b4'}
    mu_x = MU_POINTS

    ax_dp, ax_acc = axes
    for ds in DATASETS:
        # DP vs mu
        ax_dp.plot(mu_x, curve[ds]['dp_dro'], 'o-', color=colors['dro'],
                   label=f'{ds} DRO' if ds == 'adult' else f'{ds} DRO',
                   linewidth=2, markersize=5)
        ax_dp.plot(mu_x, curve[ds]['dp_naive'], 's--', color=colors['naive'],
                   alpha=0.6, label=f'{ds} naive' if ds == 'adult' else f'{ds} naive')
        # Accuracy vs mu
        ax_acc.plot(mu_x, curve[ds]['dro_acc' if False else 'acc_dro'], 'o-',
                    color=colors['dro'], linewidth=2, markersize=5,
                    label=f'{ds} DRO')
        ax_acc.plot(mu_x, curve[ds]['acc_naive'], 's--', color=colors['naive'],
                    alpha=0.6, label=f'{ds} naive')
        # Floor line
        floor = const_floor[ds]
        ax_acc.axhline(floor, color='red', linestyle=':', alpha=0.5,
                       label=f'{ds} floor={floor}' if ds == 'adult' else f'{ds} floor={floor}')
        ax_acc.axhline(floor + safe_margin, color='orange', linestyle=':', alpha=0.4)
        # Recommendation marker
        rec = recommendations.get(ds)
        if rec and rec.get('safe'):
            rmu = rec['mu']
            racc = rec['acc']
            rdp = curve[ds]['dp_dro'][MU_POINTS.index(rmu)]
            ax_dp.plot(rmu, rdp, '*', color='gold', markersize=14, zorder=5)
            ax_acc.plot(rmu, racc, '*', color='gold', markersize=14, zorder=5)

    for ax, title, yl in [
        (ax_dp, 'DP violation vs $\\mu$ (alpha=0.2, attack=dp)', 'mean DP'),
        (ax_acc, 'Accuracy vs $\\mu$ (alpha=0.2, attack=dp)', 'mean accuracy'),
    ]:
        ax.set_xscale('log')
        ax.set_xticks(mu_x)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel('mu')
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    ax_acc.set_ylim(bottom=min(v for ds in DATASETS for v in curve[ds]['acc_dro'] + curve[ds]['acc_naive']) - 0.05)
    fig.tight_layout()
    fig_path = 'results/mu_sensitivity_curve.png'
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == '__main__':
    main()
