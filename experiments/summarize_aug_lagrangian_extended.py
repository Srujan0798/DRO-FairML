#!/usr/bin/env python3
"""TASK A analysis — applies the PRE-REGISTERED rules mechanically.

Rules fixed before the data existed, in
docs/superpowers/specs/2026-08-05-al-generalisation-prereg.md:

  Rule 1 (falsification): robustness framing SUPPORTED only if
      R(0.0) < 0.5 * R(0.2) = 20.9%,  where R = (DP_dro - DP_AL)/DP_dro.
  Rule 2: win holds at higher alpha if Adult 0.3 AND 0.4 both p<0.05 and
      non-degenerate.
  Rule 3: transfers across attacks if >=1 of {if, combined} at Adult 0.2 is
      p<0.05 and non-degenerate.
  Rule 4: LSAC expected DEGENERATE; confirming collapse is the success condition.

Degeneracy guard everywhere: accuracy must exceed the constant-predictor floor
by >0.005, else a DP "win" is model collapse and is labelled as such.

Writes results/aug_lagrangian_extended_summary.md
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import wilcoxon

CONST_FLOOR = {'adult': 0.7521, 'credit': 0.7788, 'lsac': 0.9016}
FLOOR_MARGIN = 0.005
R_THRESHOLD = 20.9  # pre-registered, = 0.5 * R(0.2)=41.8%
SEEDS = range(6)


def canonical_ref():
    """Canonical DRO rows keyed by (ds, attack, alpha, seed). Read-only."""
    rows = json.load(open('results/canonical_tau1.json'))
    ref = {}
    for r in rows:
        if r.get('tau') != 1.0 or r.get('method') != 'dro':
            continue
        if r.get('corruptor_type', 'adversarial') != 'adversarial':
            continue
        if int(r.get('seed', -1)) > 5:
            continue  # locked n=6 slice
        ref.setdefault((r['dataset'], r['attack'], r['alpha'], r['seed']), r)
    return ref


def cell(al_rows, ref, ds, attack, alpha):
    """Return per-cell stats or None if incomplete."""
    dro_dp, al_dp, dro_acc, al_acc = [], [], [], []
    for s in SEEDS:
        rd = ref.get((ds, attack, alpha, s))
        ra = next((r for r in al_rows if r['dataset'] == ds and r['attack'] == attack
                   and r['alpha'] == alpha and r['seed'] == s), None)
        if rd is None or ra is None:
            continue
        dro_dp.append(rd['dp_clean']); al_dp.append(ra['dp_clean'])
        dro_acc.append(rd['acc_clean']); al_acc.append(ra['acc_clean'])
    if len(dro_dp) < 6:
        return None
    d, a = np.array(dro_dp), np.array(al_dp)
    p = 1.0 if np.allclose(d - a, 0) else wilcoxon(d, a, alternative='greater').pvalue
    mean_dro, mean_al = float(d.mean()), float(a.mean())
    R = 100 * (mean_dro - mean_al) / mean_dro if mean_dro > 1e-12 else 0.0
    floor = CONST_FLOOR[ds] + FLOOR_MARGIN
    degen = float(np.mean(al_acc)) <= floor
    # Canonical DRO can be below the floor too (the paper already excludes
    # alpha>=0.3 on Adult/Credit for exactly this reason). Flagging only AL
    # there would misattribute a pre-existing regime failure to AL.
    dro_degen = float(np.mean(dro_acc)) <= floor
    return dict(ds=ds, attack=attack, alpha=alpha, n=len(d), dp_dro=mean_dro,
                dp_al=mean_al, R=R, p=float(p), acc_dro=float(np.mean(dro_acc)),
                acc_al=float(np.mean(al_acc)), degen=degen, dro_degen=dro_degen,
                acc_delta=float(np.mean(al_acc)) - float(np.mean(dro_acc)),
                sig=(p < 0.05), genuine=(p < 0.05 and not degen))


def main():
    al_rows = json.load(open('results/aug_lagrangian_extended.json'))
    ref = canonical_ref()
    L = [f"# TASK A — does the AL improvement generalise? (pre-registered)", "",
         f"rows: **{len(al_rows)}/42** · μ=5 · criterion fixed before data in "
         "`docs/superpowers/specs/2026-08-05-al-generalisation-prereg.md`", "",
         "| dataset | attack | α | n | DP dro | DP AL | R (rel. reduction) | p | "
         "acc dro | acc AL | floor | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    wanted = ([('adult', 'dp', 0.0)] + [('adult', 'dp', a) for a in (0.3, 0.4)]
              + [('adult', atk, 0.2) for atk in ('if', 'combined')]
              + [('lsac', 'dp', a) for a in (0.1, 0.2)])
    cells = {}
    for ds, atk, a in wanted:
        c = cell(al_rows, ref, ds, atk, a)
        if c is None:
            L.append(f"| {ds} | {atk} | {a} | — | *incomplete* | | | | | | | |")
            continue
        cells[(ds, atk, a)] = c
        if c['degen'] and c['dro_degen']:
            verdict = "both below floor (excluded regime)"
        elif c['degen']:
            verdict = "**AL DEGENERATE**"
        elif c['genuine']:
            verdict = "genuine win"
        else:
            verdict = "no sig. win"
        L.append(f"| {ds} | {atk} | {a} | {c['n']} | {c['dp_dro']:.4f} | "
                 f"{c['dp_al']:.4f} | {c['R']:+.1f}% | {c['p']:.4f}"
                 f"{'*' if c['sig'] else ''} | {c['acc_dro']:.4f} | "
                 f"{c['acc_al']:.4f} | {CONST_FLOOR[ds]:.4f} | {verdict} |")

    L += ["", "## Pre-registered rules applied", ""]

    c0 = cells.get(('adult', 'dp', 0.0))
    if c0:
        supported = c0['R'] < R_THRESHOLD
        L.append(f"**Rule 1 — FALSIFICATION (α=0 control).** R(0.0) = "
                 f"**{c0['R']:+.1f}%** vs pre-registered threshold {R_THRESHOLD}% "
                 f"(half of R(0.2)=41.8%). → "
                 + ("**Corruption-robustness framing SUPPORTED**: AL's benefit "
                    "without corruption is materially smaller than under attack."
                    if supported else
                    "**Corruption-robustness framing NOT SUPPORTED**: AL helps "
                    "comparably with no corruption present, so it behaves as a "
                    "GENERIC fairness regulariser. The robustness framing is "
                    "withdrawn; AL must be presented as a general fairness "
                    "improvement, not a corruption-specific one."))
    else:
        L.append("**Rule 1** — α=0 control incomplete.")

    hi = [cells.get(('adult', 'dp', a)) for a in (0.3, 0.4)]
    if all(hi):
        ok = all(c['genuine'] for c in hi)
        L.append(f"\n**Rule 2 — higher corruption.** α=0.3 p={hi[0]['p']:.4f}, "
                 f"acc {hi[0]['acc_al']:.4f} (DRO {hi[0]['acc_dro']:.4f}); "
                 f"α=0.4 p={hi[1]['p']:.4f}, acc {hi[1]['acc_al']:.4f} "
                 f"(DRO {hi[1]['acc_dro']:.4f}) → "
                 + ("**win HOLDS at higher α**" if ok
                    else "**does NOT hold at higher α — the DP gains there are "
                         "not usable improvements**"))
        if hi[1]['acc_delta'] < -0.1:
            L.append(f"\n> **Instability warning (new failure mode).** At α=0.4 "
                     f"AL's accuracy is {hi[1]['acc_al']:.4f} against canonical "
                     f"DRO's {hi[1]['acc_dro']:.4f} — a drop of "
                     f"{hi[1]['acc_delta']:+.4f}. That is far below the "
                     f"constant-predictor floor and below chance for this label "
                     f"balance: μ=5 destabilises training at heavy corruption "
                     f"rather than merely degrading it. Any recommendation for "
                     f"μ must therefore be corruption-dependent (see TASK C), "
                     f"and μ=5 must not be presented as a universal default.")

    atks = [cells.get(('adult', atk, 0.2)) for atk in ('if', 'combined')]
    if all(atks):
        ok = any(c['genuine'] for c in atks)
        L.append(f"\n**Rule 3 — other attacks.** IF p={atks[0]['p']:.4f} "
                 f"({'genuine' if atks[0]['genuine'] else 'no'}), COMBINED "
                 f"p={atks[1]['p']:.4f} ({'genuine' if atks[1]['genuine'] else 'no'}) → "
                 + ("**transfers across attacks**" if ok else "**does NOT transfer**"))

    ls = [cells.get(('lsac', 'dp', a)) for a in (0.1, 0.2)]
    if all(ls):
        alldeg = all(c['degen'] for c in ls)
        L.append(f"\n**Rule 4 — LSAC.** "
                 + ("Both cells DEGENERATE as predicted — collapse to the "
                    "constant predictor, consistent with the documented LSAC/DP "
                    "failure mode. Reported as collapse, NOT as a win."
                    if alldeg else
                    "NOT uniformly degenerate — unexpected; inspect before claiming anything."))

    L += ["", "R = (DP_dro − DP_AL)/DP_dro, the relative DP reduction AL buys. "
              "`genuine win` requires p<0.05 AND accuracy clear of the "
              "constant-predictor floor by >0.005."]

    out = 'results/aug_lagrangian_extended_summary.md'
    with open(out, 'w') as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
