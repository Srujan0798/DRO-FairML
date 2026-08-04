"""
Diagnostic: Why DRO underperforms Naive under adversarial attack.

Hypothesis: DRO's _compute_radii() assumes uniform random corruption,
but our greedy label attack creates a NON-UNIFORM corrupt set.
This misaligns DRO's uncertainty set.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.datasets import get_dataset
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.dro_fair import DroFairTrainer
from src.models.classifier import MLPClassifier

def compute_dp(y, a):
    m0, m1 = (a == 0), (a == 1)
    p0 = np.mean(y[m0]) if m0.sum() > 0 else 0
    p1 = np.mean(y[m1]) if m1.sum() > 0 else 0
    return abs(p0 - p1)

# Load Adult
X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = get_dataset('adult', random_state=0)

print("=" * 70)
print("DRO RADII DIAGNOSTIC")
print("=" * 70)
print(f"\nClean data: n={len(y_train)}, groups={np.bincount(a_train.astype(int))}")
print(f"Clean DP: {compute_dp(y_train, a_train):.4f}")

for alpha in [0.1, 0.2, 0.3]:
    print(f"\n{'='*70}")
    print(f"ALPHA = {alpha}")
    print(f"{'='*70}")

    # Attack with coordinated=False
    atk = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=20,
                               epsilon=0.3, pgd_step_size=0.02,
                               coordinated=False, random_state=0)
    X_c, y_c, a_c, mask = atk.corrupt(X_train, y_train, a_train)

    n_corrupt = mask.sum()
    print(f"Corrupted: n={n_corrupt} ({100*n_corrupt/len(y_train):.1f}%)")
    print(f"Corrupted DP: {compute_dp(y_c, a_c):.4f}")

    # Group proportions
    clean_counts = np.bincount(a_train.astype(int))
    corrupt_counts = np.bincount(a_c.astype(int))
    print(f"Clean groups:   {clean_counts}  ({clean_counts/clean_counts.sum()*100})")
    print(f"Corrupt groups: {corrupt_counts}  ({corrupt_counts/corrupt_counts.sum()*100})")

    # What DRO's formula computes
    pi_obs = np.array([np.mean(a_c == j) for j in [0, 1]])
    if alpha != 0.5:
        pi_clean_est = (pi_obs - alpha) / (1 - 2 * alpha)
    else:
        pi_clean_est = pi_obs
    pi_clean_est = np.clip(pi_clean_est, 0.0, 1.0)

    print(f"\nDRO's _compute_radii():")
    print(f"  pi_obs (corrupted):     {pi_obs}")
    print(f"  pi_clean (estimated):   {pi_clean_est}")
    print(f"  pi_clean (TRUE):        {clean_counts/clean_counts.sum()}")
    print(f"  ESTIMATION ERROR:       {np.abs(pi_clean_est - clean_counts/clean_counts.sum())}")

    rho_dp = []
    for j in [0, 1]:
        denom = (1 - alpha) * pi_clean_est[j] + alpha
        rho_dp.append(alpha / denom if denom > 0 else 1.0)
    print(f"  rho_dp: {rho_dp}")

    # The corrupt set composition
    corrupt_idx = np.where(mask)[0]
    minority_group = int(np.argmin(clean_counts))
    n_minority_corrupt = np.sum(a_train[corrupt_idx] == minority_group)
    print(f"\nCorrupt set composition:")
    print(f"  Minority group ({minority_group}) in corrupt set: {n_minority_corrupt}/{n_corrupt} = {100*n_minority_corrupt/n_corrupt:.1f}%")
    print(f"  Majority group in corrupt set: {n_corrupt - n_minority_corrupt}/{n_corrupt} = {100*(n_corrupt-n_minority_corrupt)/n_corrupt:.1f}%")
    print(f"  Uniform would be: ~{100*clean_counts[minority_group]/clean_counts.sum():.1f}% minority")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("""
If the corrupt set is NOT uniformly random (e.g., minority group is
OVER-REPRESENTED due to greedy label attack), then pi_obs is biased.
DRO's formula inverts this biased pi_obs and gets a WRONG pi_clean_est.
This pushes DRO's uncertainty set in the wrong direction.
""")
