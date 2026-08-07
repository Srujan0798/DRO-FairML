#!/usr/bin/env python3
"""TASK E: independent recomputation of the headline statistics.

Steps 3, 4, 5, 8 of work/wave-al-e/01-independent-review.md. Recomputes from
raw JSON + datasets only. Writes results/al_review_stats.json.
"""
import json
import numpy as np
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    with open(path) as f:
        return json.load(f)


canonical = load('results/canonical_tau1.json')
mu_sens = load('results/mu_sensitivity.json')
radius = load('results/al_radius_compound.json')

out = {}

# ---------- STEP 3: Wilcoxon ----------
can = sorted(
    [r for r in canonical
     if r.get('dataset') == 'adult' and r.get('alpha') == 0.2
     and r.get('method') == 'dro' and r.get('attack') == 'dp'
     and r.get('tau') == 1.0 and r.get('seed') < 6
     and r.get('corruptor_type', 'adversarial') == 'adversarial'],
    key=lambda r: r['seed'])
al = sorted(
    [r for r in mu_sens
     if r.get('dataset') == 'adult' and r.get('alpha') == 0.2
     and r.get('method') == 'dro' and r.get('attack') == 'dp'
     and abs(r.get('aug_lagrangian_mu', 0.0) - 20.0) < 1e-9],
    key=lambda r: r['seed'])

print(f"canonical rows: {len(can)}, seeds: {[r['seed'] for r in can]}")
print(f"AL rows: {len(al)}, seeds: {[r['seed'] for r in al]}")

dro_vals = [r['dp_clean'] for r in can]
al_vals = [r['dp_clean'] for r in al]
out['canonical_dp'] = dro_vals
out['al_dp'] = al_vals
out['canonical_seeds'] = [r['seed'] for r in can]
out['al_seeds'] = [r['seed'] for r in al]
out['canonical_acc'] = [r['acc_clean'] for r in can]
out['al_acc'] = [r['acc_clean'] for r in al]

res = stats.wilcoxon(dro_vals, al_vals, alternative='greater')
out['wilcoxon_stat'] = res.statistic
out['wilcoxon_p_one_sided_greater'] = res.pvalue
out['wilcoxon_p_two_sided'] = stats.wilcoxon(dro_vals, al_vals).pvalue
out['diffs'] = [d - a for d, a in zip(dro_vals, al_vals)]
out['mean_canonical_dp'] = float(np.mean(dro_vals))
out['mean_al_dp'] = float(np.mean(al_vals))
out['mean_dp_reduction_pct'] = float(100 * (np.mean(dro_vals) - np.mean(al_vals)) / np.mean(dro_vals))
per_seed_reduction = [100 * (d - a) / d for d, a in zip(dro_vals, al_vals)]
out['per_seed_reduction_pct'] = per_seed_reduction
out['per_seed_reduction_min'] = min(per_seed_reduction)
out['per_seed_reduction_max'] = max(per_seed_reduction)

# alpha=0.0 cell reduction (claim 81.7%)
can0 = [r for r in canonical
        if r.get('dataset') == 'adult' and r.get('alpha') == 0.0
        and r.get('method') == 'dro' and r.get('attack') == 'dp'
        and r.get('tau') == 1.0 and r.get('seed') < 6]
al0 = sorted([r for r in mu_sens
              if r.get('dataset') == 'adult' and r.get('alpha') == 0.0
              and r.get('method') == 'dro' and r.get('attack') == 'dp'
              and abs(r.get('aug_lagrangian_mu', 0.0) - 20.0) < 1e-9],
             key=lambda r: r['seed'])
print(f"alpha=0.0: canonical rows {len(can0)} seeds {[r['seed'] for r in can0]}, AL rows {len(al0)} seeds {[r['seed'] for r in al0]}")
if len(can0) == 6 and len(al0) == 6:
    d0 = [r['dp_clean'] for r in sorted(can0, key=lambda r: r['seed'])]
    a0 = [r['dp_clean'] for r in al0]
    out['alpha0_canonical_dp'] = d0
    out['alpha0_al_dp'] = a0
    out['alpha0_mean_dp_reduction_pct'] = float(100 * (np.mean(d0) - np.mean(a0)) / np.mean(d0))
    out['alpha0_wilcoxon_p'] = stats.wilcoxon(d0, a0, alternative='greater').pvalue

# ---------- STEP 4: constant-predictor floors ----------
from src.data.datasets import get_dataset
floors = {}
for ds in ['adult', 'credit', 'lsac']:
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(ds, random_state=42)
    ytr = np.asarray(y_train)
    pos = ytr.mean()
    maj_frac = max(pos, 1 - pos)
    # also test-set majority fraction
    yte = np.asarray(y_test)
    pos_te = yte.mean()
    maj_frac_te = max(pos_te, 1 - pos_te)
    floors[ds] = dict(train_majority_frac=float(maj_frac), test_majority_frac=float(maj_frac_te),
                      train_pos_rate=float(pos), test_pos_rate=float(pos_te))
    print(f"{ds}: train majority {maj_frac:.6f} (claim) test majority {maj_frac_te:.6f}")
out['constant_predictor_floors'] = floors

# ---------- STEP 5: al_radius_compound degeneracy ----------
compound = sorted(
    [r for r in radius
     if r.get('dataset') == 'adult' and r.get('alpha') == 0.2
     and abs(r.get('radii_scale', 1.0) - 2.0) < 1e-9
     and abs(r.get('aug_lagrangian_mu', 0.0) - 20.0) < 1e-9
     and r.get('method') == 'dro'],
    key=lambda r: r['seed'])
print(f"compound rows: {len(compound)} seeds {[r['seed'] for r in compound]}")
compound_accs = [r['acc_clean'] for r in compound]
compound_dps = [r['dp_clean'] for r in compound]
out['compound_accs'] = compound_accs
out['compound_dps'] = compound_dps
out['compound_mean_acc'] = float(np.mean(compound_accs))
out['compound_mean_dp'] = float(np.mean(compound_dps))

floor_adult = floors['adult']['train_majority_frac']
out['floor_adult_exact'] = floor_adult
for margin in [0.0, 0.005, 0.01]:
    thr = floor_adult + margin
    n_at_below = sum(1 for a in compound_accs if a <= thr)
    out[f'compound_margin_{margin}_n_at_or_below'] = n_at_below
    out[f'compound_margin_{margin}_mean_below_thr'] = bool(out['compound_mean_acc'] < thr)
    print(f"margin={margin}: threshold={thr:.6f} mean_below={out['compound_mean_acc'] < thr} "
          f"seeds_at_or_below={n_at_below}/6")

# ---------- STEP 8: canonical DRO near-floor check ----------
for r in can:
    r['gap_to_floor'] = r['acc_clean'] - floor_adult
out['canonical_acc_gap_to_floor'] = [r['acc_clean'] - floor_adult for r in can]
out['canonical_acc_floor'] = floor_adult
print("canonical acc gaps to floor:", [f"{x:.6f}" for x in out['canonical_acc_gap_to_floor']])

with open('results/al_review_stats.json', 'w') as f:
    json.dump(out, f, indent=2, default=float)
print("\nwrote results/al_review_stats.json")
