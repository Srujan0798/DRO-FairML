#!/usr/bin/env python3
"""TASK C analysis — applies the PRE-REGISTERED rules (Rule C1-C4).

Pre-registration: docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md

Mapped: where AL is SAFE (accuracy > floor + 0.005) and EFFECTIVE (p<0.05 vs
canonical DRO) as a function of mu and alpha on Adult, and whether a smaller
mu rescues Credit from its α=0.2 collapse at mu ∈ {5,10}.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import wilcoxon

CONST_FLOOR = {'adult': 0.7521, 'credit': 0.7788}
FLOOR_MARGIN = 0.005
SEEDS = range(6)


def canonical_ref(ds, atk='dp'):
    """Canonical DRO rows, n=6 slice only."""
    rows = json.load(open('results/canonical_tau1.json'))
    ref = {}
    for r in rows:
        if r.get('tau') != 1.0 or r.get('method') != 'dro' or r.get('attack') != atk:
            continue
        if r.get('corruptor_type', 'adversarial') != 'adversarial':
            continue
        if int(r.get('seed', -1)) > 5 or r['dataset'] != ds:
            continue
        ref.setdefault((r['alpha'], r['seed']), r)
    return ref


def cell_stats(mu_rows, ref, ds, alpha):
    """Per-cell stats: SAFE, EFFECTIVE, DP reduction."""
    dro_dp, mu_dp, mu_acc = [], [], []
    for s in SEEDS:
        rd = ref.get((alpha, s))
        rm = next((r for r in mu_rows if r['dataset'] == ds and r['alpha'] == alpha
                   and r['seed'] == s), None)
        if rd is None or rm is None:
            continue
        dro_dp.append(rd['dp_clean']); mu_dp.append(rm['dp_clean']); mu_acc.append(rm['acc_clean'])
    if len(dro_dp) < 6:
        return None
    d, m = np.array(dro_dp), np.array(mu_dp)
    p = 1.0 if np.allclose(d - m, 0) else wilcoxon(d, m, alternative='greater').pvalue
    floor = CONST_FLOOR[ds] + FLOOR_MARGIN
    safe = float(np.mean(mu_acc)) > floor
    effective = (p < 0.05) and (np.mean(m) < np.mean(d))
    return dict(dp_dro=float(np.mean(d)), dp_mu=float(np.mean(m)),
                R=100*(np.mean(d)-np.mean(m))/np.mean(d) if np.mean(d)>1e-12 else 0,
                p=float(p), acc=float(np.mean(mu_acc)), safe=safe, effective=effective)


def main():
    mu_sens = json.load(open('results/mu_sensitivity.json'))
    L = ["# TASK C — μ sensitivity (pre-registered)", "",
         f"rows: **{len(mu_sens)}/90** · pre-reg in "
         "`docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md`", ""]

    # RULE C1: Safe operating range on Adult
    L += ["## RULE C1 — μ_max_safe(α): where is AL SAFE?", ""]
    L += ["| α | μ=0.5 | μ=1 | μ=2 | μ=20 | μ_max_safe |",
          "|---|---|---|---|---|---|"]
    ref_adult = canonical_ref('adult')
    c1_results = {}
    for alpha in [0.0, 0.2, 0.4]:
        row_cells = []
        mu_max_safe = None
        for mu in [0.5, 1.0, 2.0, 20.0]:
            mu_rows = [r for r in mu_sens if r['dataset']=='adult' and r['alpha']==alpha and r['aug_lagrangian_mu']==mu]
            c = cell_stats(mu_rows, ref_adult, 'adult', alpha)
            if c:
                row_cells.append(f"{'✓' if c['safe'] else '✗'} ({c['acc']:.3f})")
                if c['safe']: mu_max_safe = mu
            else:
                row_cells.append("?")
        c1_results[alpha] = mu_max_safe
        L.append(f"| {alpha} | {' | '.join(row_cells)} | {mu_max_safe} |")

    L += ["", "**Pre-registered expectation:** μ_max_safe DECREASES as α increases. "
          f"**Observed:** α=0.0→{c1_results[0.0]}, α=0.2→{c1_results[0.2]}, "
          f"α=0.4→{c1_results[0.4]}. "
          + ("✓ expectation CONFIRMED" if c1_results[0.0] >= (c1_results[0.2] or 0) >= (c1_results[0.4] or 0)
             else "⚠ expectation NOT confirmed")]

    # RULE C2: Recommended μ per α
    L += ["", "## RULE C2 — recommended μ (largest both SAFE and EFFECTIVE)", ""]
    L += ["| α | μ_recommend | DP reduction | p-value | accuracy | action |",
          "|---|---|---|---|---|---|"]
    for alpha in [0.0, 0.2, 0.4]:
        mu_rows_all = [r for r in mu_sens if r['dataset']=='adult' and r['alpha']==alpha]
        best_mu = None
        best_stats = None
        for mu in [0.5, 1.0, 2.0, 20.0]:
            mu_rows = [r for r in mu_rows_all if r['aug_lagrangian_mu']==mu]
            c = cell_stats(mu_rows, ref_adult, 'adult', alpha)
            if c and c['safe'] and c['effective']:
                if best_mu is None or mu > best_mu:
                    best_mu = mu
                    best_stats = c
        if best_mu:
            L.append(f"| {alpha} | μ={best_mu} | {best_stats['R']:+.1f}% | "
                     f"{best_stats['p']:.4f} | {best_stats['acc']:.4f} | **use μ={best_mu}** |")
        else:
            L.append(f"| {alpha} | none | — | — | — | **do not use AL** |")

    # RULE C3: Credit rescue
    L += ["", "## RULE C3 — Credit rescue: does a smaller μ work?", ""]
    ref_credit = canonical_ref('credit')
    credit_rescued = False
    for mu in [0.5, 1.0, 2.0]:
        mu_rows = [r for r in mu_sens if r['dataset']=='credit' and r['alpha']==0.2 and r['aug_lagrangian_mu']==mu]
        c = cell_stats(mu_rows, ref_credit, 'credit', 0.2)
        if c and c['safe'] and c['effective']:
            L.append(f"**μ={mu} on Credit α=0.2:** SAFE (acc {c['acc']:.4f}) and EFFECTIVE "
                     f"(DP {c['dp_dro']:.4f}→{c['dp_mu']:.4f}, p={c['p']:.4f}). "
                     f"**Credit RESCUED; AL generalises to 2 datasets.**")
            credit_rescued = True
            break
    if not credit_rescued:
        L.append("**Credit NOT rescued:** no μ ∈ {0.5, 1, 2} is both SAFE and EFFECTIVE. "
                 "AL is **Adult-only among tested datasets**.")

    # RULE C4: Monotonicity
    L += ["", "## RULE C4 — monotonicity sanity check", ""]
    all_mono = True
    for alpha in [0.0, 0.2, 0.4]:
        dps = []
        for mu in [0.5, 1.0, 2.0, 20.0]:
            mu_rows = [r for r in mu_sens if r['dataset']=='adult' and r['alpha']==alpha and r['aug_lagrangian_mu']==mu]
            c = cell_stats(mu_rows, ref_adult, 'adult', alpha)
            if c: dps.append(c['dp_mu'])
        is_mono = all(dps[i] >= dps[i+1] for i in range(len(dps)-1))
        if not is_mono: all_mono = False
        L.append(f"α={alpha}: DP curve {'monotone ✓' if is_mono else '**non-monotone ⚠**'} {[f'{d:.4f}' for d in dps]}")

    out = 'results/mu_sensitivity_summary.md'
    with open(out, 'w') as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
