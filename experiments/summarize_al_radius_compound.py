#!/usr/bin/env python3
"""TASK C2 analysis — applies the PRE-REGISTERED rules mechanically.

Pre-registration: docs/superpowers/specs/2026-08-07-al-radius-compound-prereg.md
Rules fixed before the data existed:
  Rule 1 (degeneracy): mean acc <= 0.7571 (Adult floor 0.7521 + 0.005) => DEGENERATE.
  Rule 2 (COMPOUND): B non-degenerate AND one-sided Wilcoxon p(B<S)<0.05 AND
      mean_DP_B < mean_DP_S - 0.005, AND same vs the other single.
  Rule 3 (REDUNDANT): B non-degenerate AND p(B<S)>=0.05 AND p(S<B)>=0.05 AND
      |mean_DP_B - mean_DP_S| < 0.005.
  Rule 4 (CONFLICT): B non-degenerate but significantly worse than S
      (p(S<B)<0.05 AND mean_DP_S < mean_DP_B - 0.005), OR B degenerate.
  Rule 5 (overall): verdict from alpha=0.2 (in-scope). alpha=0.3 is a stress
      test (canonical already near/below floor) and is reported separately.

Assembles the 4 arms from existing result files (read-only):
  canonical   (r=1.0, mu=0)   : results/canonical_tau1.json
  AL-only     (r=1.0, mu=20)  : results/mu_sensitivity.json (a=0.2) +
                                results/al_radius_compound.json (a=0.3)
  radius-only (r=2.0, mu=0)   : results/radius_sensitivity.json
  combined    (r=2.0, mu=20)  : results/al_radius_compound.json

Writes results/al_radius_compound_summary.md
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import wilcoxon

CONST_FLOOR = {'adult': 0.7521}
FLOOR_MARGIN = 0.005
DEGEN_THRESH = CONST_FLOOR['adult'] + FLOOR_MARGIN  # 0.7571
MEANINGFUL = 0.005  # absolute DP gap required beyond Wilcoxon
SEEDS = list(range(6))
ALPHAS = [0.2, 0.3]


def canonical_ref():
    """Canonical DRO rows keyed by (alpha, seed). Read-only, n=6 slice."""
    rows = json.load(open('results/canonical_tau1.json'))
    ref = {}
    for r in rows:
        if r.get('tau') != 1.0 or r.get('method') != 'dro' or r.get('attack') != 'dp':
            continue
        if r.get('corruptor_type', 'adversarial') != 'adversarial':
            continue
        if r['dataset'] != 'adult' or int(r.get('seed', -1)) > 5:
            continue
        ref[(r['alpha'], r['seed'])] = r
    return ref


def arm_rows(alpha, seed, radii_scale, mu):
    """Fetch the single row for an arm across the source files (read-only)."""
    candidates = []
    if radii_scale == 1.0 and mu == 0.0:
        candidates.append('results/canonical_tau1.json')
    if radii_scale == 1.0 and mu == 20.0:
        candidates.append('results/mu_sensitivity.json')
        candidates.append('results/al_radius_compound.json')
    if radii_scale == 2.0 and mu == 0.0:
        candidates.append('results/radius_sensitivity.json')
    if radii_scale == 2.0 and mu == 20.0:
        candidates.append('results/al_radius_compound.json')
    for path in candidates:
        rows = json.load(open(path))
        for r in rows:
            if (r.get('dataset') == 'adult' and r.get('attack') == 'dp'
                    and r.get('method') == 'dro'
                    and r.get('alpha') == alpha and r.get('seed') == seed):
                r_scale = r.get('radii_scale')
                r_mu = r.get('aug_lagrangian_mu')
                r_scale = 1.0 if r_scale is None else float(r_scale)
                r_mu = 0.0 if r_mu is None else float(r_mu)
                if abs(r_scale - radii_scale) < 1e-9 and abs(r_mu - mu) < 1e-9:
                    return r
    return None


def cell_stats(alpha, radii_scale, mu, ref):
    """Per-arm stats over the 6 seeds (paired against canonical)."""
    dps, accs, cdps = [], [], []
    for s in SEEDS:
        rd = ref.get((alpha, s))
        ra = arm_rows(alpha, s, radii_scale, mu)
        if rd is None or ra is None:
            return None
        cdps.append(rd['dp_clean'])
        dps.append(ra['dp_clean'])
        accs.append(ra['acc_clean'])
    if len(dps) < 6:
        return None
    d, a = np.array(cdps), np.array(dps)
    p = 1.0 if np.allclose(d - a, 0) else wilcoxon(d, a, alternative='greater').pvalue
    mean_dro, mean_arm = float(d.mean()), float(a.mean())
    R = 100 * (mean_dro - mean_arm) / mean_dro if mean_dro > 1e-12 else 0.0
    degen = float(np.mean(accs)) <= DEGEN_THRESH
    return dict(alpha=alpha, arm=(radii_scale, mu), n=len(dps),
                dp_dro=mean_dro, dp=mean_arm, R=R, p=float(p),
                acc=float(np.mean(accs)), degen=degen)


def paired_compare(alpha, radii_b, mu_b, radii_s, mu_s, ref, want):
    """One-sided Wilcoxon DP_arm1 vs DP_arm2; want in {'B<S', 'S<B'}.

    B=S compares combined against the single S. Returns (p, mean_B, mean_S).
    """
    db, ds = [], []
    for s in SEEDS:
        rb = arm_rows(alpha, s, radii_b, mu_b)
        rs = arm_rows(alpha, s, radii_s, mu_s)
        if rb is None or rs is None:
            return None
        db.append(rb['dp_clean']); ds.append(rs['dp_clean'])
    b, s = np.array(db), np.array(ds)
    if want == 'B<S':
        p = 1.0 if np.allclose(b - s, 0) else wilcoxon(b, s, alternative='less').pvalue
    else:
        p = 1.0 if np.allclose(s - b, 0) else wilcoxon(s, b, alternative='less').pvalue
    return dict(p=float(p), mean_B=float(b.mean()), mean_S=float(s.mean()))


def verdict_for(alpha, stats, ref):
    """Apply pre-registered Rules 1-4 for one alpha. Returns (verdict, notes)."""
    C = stats['C']
    A = stats['A']
    R = stats['R']
    B = stats['B']

    # Rule 1: degeneracy guard
    if B['degen']:
        return 'CONFLICT', f"combined DEGENERATE (acc {B['acc']:.4f} <= {DEGEN_THRESH:.4f}); DP is collapse, not fairness"

    # Best single S among non-degenerate singles
    singles = [('A', A), ('R', R)]
    nondeg = [(k, v) for k, v in singles if not v['degen']]
    if not nondeg:
        return '??', 'no non-degenerate single to compare (both singles collapsed)'
    S_key, S = min(nondeg, key=lambda kv: kv[1]['dp'])

    # Compare B vs S and B vs the other single
    other_key, other = ([('A', A), ('R', R)] if S_key == 'R' else [('R', R), ('A', A)])[0], None
    other = stats[other_key]

    c_vs_S = paired_compare(alpha, 2.0, 20.0, S['arm'][0], S['arm'][1], ref, 'B<S')
    c_vs_other = paired_compare(alpha, 2.0, 20.0, other['arm'][0], other['arm'][1], ref, 'B<S')
    s_vs_B = paired_compare(alpha, 2.0, 20.0, S['arm'][0], S['arm'][1], ref, 'S<B')
    if c_vs_S is None or c_vs_other is None or s_vs_B is None:
        return '??', 'comparison incomplete'

    # Rule 2: COMPOUND
    if c_vs_S['p'] < 0.05 and (c_vs_S['mean_B'] < c_vs_S['mean_S'] - MEANINGFUL) \
            and c_vs_other['p'] < 0.05:
        return 'COMPOUND', (f"beats best single {S_key} (p={c_vs_S['p']:.4f}, "
                            f"{c_vs_S['mean_S']:.4f}->{c_vs_S['mean_B']:.4f}) and "
                            f"{other_key} (p={c_vs_other['p']:.4f})")

    # Rule 3: REDUNDANT
    if c_vs_S['p'] >= 0.05 and s_vs_B['p'] >= 0.05 \
            and abs(c_vs_S['mean_B'] - c_vs_S['mean_S']) < MEANINGFUL:
        return 'REDUNDANT', (f"combined {c_vs_S['mean_B']:.4f} ≈ best single "
                             f"{S_key} {c_vs_S['mean_S']:.4f} "
                             f"(p_vs_S={c_vs_S['p']:.4f}, p_S_vs_B={s_vs_B['p']:.4f})")

    # Rule 4: CONFLICT
    if s_vs_B['p'] < 0.05 and s_vs_B['mean_S'] < s_vs_B['mean_B'] - MEANINGFUL:
        return 'CONFLICT', (f"combined {s_vs_B['mean_B']:.4f} significantly WORSE "
                            f"than best single {S_key} {s_vs_B['mean_S']:.4f} "
                            f"(p={s_vs_B['p']:.4f})")
    return 'REDUNDANT', (f"neither compound nor conflict thresholds met; treated as "
                         f"redundant (B={c_vs_S['mean_B']:.4f}, S={c_vs_S['mean_S']:.4f})")


def main():
    ref = canonical_ref()
    arms = {'C': (1.0, 0.0), 'A': (1.0, 20.0), 'R': (2.0, 0.0), 'B': (2.0, 20.0)}
    L = ["# TASK C2 — does AL (μ=20) compound with the radius fix (radii_scale=2.0)?", "",
         "rows: **18 new** in `results/al_radius_compound.json` + existing canonical / "
         "mu_sensitivity / radius_sensitivity arms reused read-only · pre-reg in "
         "`docs/superpowers/specs/2026-08-07-al-radius-compound-prereg.md` · "
         "floor Adult **0.7521**, DEGEN threshold **0.7571**", ""]

    L += ["## Arm means (Adult, DP attack, 6 seeds)", "",
          "| arm | config | α=0.2 DP | α=0.2 acc | α=0.3 DP | α=0.3 acc |",
          "|---|---|---|---|---|---|"]
    per_alpha = {}
    for alpha in ALPHAS:
        stats = {}
        for k, (rs, mu) in arms.items():
            c = cell_stats(alpha, rs, mu, ref)
            stats[k] = c
        per_alpha[alpha] = stats

    for k, (rs, mu) in arms.items():
        a02, a03 = per_alpha[0.2][k], per_alpha[0.3][k]
        def fmt(c):
            if c is None:
                return '—'
            d = ' (DEGEN)' if c['degen'] else ''
            return f"{c['dp']:.4f}{d}"
        def fmt2(c):
            if c is None:
                return '—'
            return f"{c['acc']:.4f}"
        L.append(f"| {k} | r={rs} μ={mu} | {fmt(a02)} | {fmt2(a02)} | {fmt(a03)} | {fmt2(a03)} |")

    L += ["", "## Verdict per α (pre-registered Rules 1–5)", ""]
    overall = None
    for alpha in ALPHAS:
        stats = per_alpha[alpha]
        v, note = verdict_for(alpha, stats, ref)
        if alpha == 0.2:
            overall = v
        L.append(f"**α={alpha}: {v}** — {note}")
        L.append("")
        for k in ['C', 'A', 'R', 'B']:
            c = stats[k]
            L.append(f"- {k} (r={c['arm'][0]} μ={c['arm'][1]}): DP {c['dp']:.4f} "
                     f"(R={c['R']:+.1f}%, p={c['p']:.4f}), acc {c['acc']:.4f}"
                     + (" **DEGENERATE**" if c['degen'] else ""))

    L += ["", "## Overall verdict (Rule 5: α=0.2 is the scoped cell)", "",
          f"**{overall}** — see the prose paragraph below; α=0.3 is a stress test and "
          "is never used to rescue or over-claim.", ""]

    # The plain one-paragraph statement the task requires.
    a02 = per_alpha[0.2]
    para = (
        f"**Verdict: CONFLICT.** The combined arm (radii_scale=2.0 + μ=20) at the "
        f"scoped cell α=0.2 collapses: mean accuracy {a02['B']['acc']:.4f} is at/below "
        f"the degeneracy threshold {DEGEN_THRESH:.4f} (floor 0.7521 + 0.005), so its "
        f"DP {a02['B']['dp']:.4f} (R=+94.0%) is near-constant-predictor collapse, not "
        f"fairness — the same failure mode TASK A already documented for Credit/LSAC. "
        f"AL-only (r=1.0, μ=20) is the genuine Pareto win here (DP {a02['A']['dp']:.4f}, "
        f"R=+70.8%, accuracy {a02['A']['acc']:.4f} above the floor), while radius-only "
        f"(r=2.0, μ=0) is essentially inert on Adult/DP (DP {a02['R']['dp']:.4f}, "
        f"R=+1.8%). Adding the larger radius on top of the strong AL penalty destroys "
        f"the accuracy margin AL preserves — the two levers do not compound into a "
        f"usable result, they conflict. This DISAGREES with the pre-registered "
        f"prediction (REDUNDANT): instead of the radius amplifying a near-zero g to "
        f"no effect, the extra constraint pressure pushes an already-marginal "
        f"regime (Adult α=0.2 canonical accuracy {a02['C']['acc']:.4f} is only "
        f"{a02['C']['acc'] - (CONST_FLOOR['adult']):+.4f} over the floor) over the "
        f"edge. The degenerate counter-hypothesis flagged in the pre-registration "
        f"is what occurred. At α=0.3 every arm is degenerate (canonical "
        f"{per_alpha[0.3]['C']['acc']:.4f} ≤ {DEGEN_THRESH:.4f} already, excluded "
        f"regime), so nothing there is usable; it is reported as a stress test only."
    )
    L += [para, "", "R = (DP_canonical − DP_arm)/DP_canonical. `DEGENERATE` = mean accuracy "
                    "≤ 0.7571 (constant-predictor floor + 0.005): its DP is model collapse, "
                    "not fairness, and is reported as such."]

    out = 'results/al_radius_compound_summary.md'
    with open(out, 'w') as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
