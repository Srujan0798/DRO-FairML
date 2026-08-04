#!/usr/bin/env python3
"""
Demonstrate the bugs in FairnessTargetedPGD's DP attack and verify the fix.

Bugs found:
1. GRADIENT MAGNITUDE: ignores group sizes. A flip in a small group has
   larger impact on DP than a flip in a large group, but current code
   assigns uniform magnitude (+1/-1) regardless of group size.

2. BATCHED PGD ON DISCRETE FLIPS: flips the SAME samples multiple times
   (pgd_steps iterations). With pgd_steps=5, each selected sample is
   flipped 5 times → ends up back at original label if started at 0!
   The gradient is recomputed after each batch flip on a corrupted state,
   making the selection nonsensical.

3. CORRECT ALGORITHM: greedy — flip ONE sample at a time, recompute ALL
   marginal gains, repeat.

Run: python3 experiments/test_dp_attack_bug.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.corruption.adversarial import FairnessTargetedPGD


def dp_violation(y, a):
    """Compute DP violation."""
    p0 = np.mean(y[a == 0]) if np.any(a == 0) else 0
    p1 = np.mean(y[a == 1]) if np.any(a == 1) else 0
    return abs(p0 - p1)


def greedy_dp_attack(y, a, alpha, coordinated=False):
    """
    CORRECT greedy DP attack.
    At each step, compute the marginal gain in DP from flipping each
    unflipped sample, then flip the sample with highest positive gain.
    """
    n = len(y)
    n_corrupt = int(alpha * n)
    y_adv = y.copy()
    flipped = np.zeros(n, dtype=bool)
    group_counts = np.bincount(a.astype(int))
    minority_group = int(np.argmin(group_counts))
    
    for _ in range(n_corrupt):
        p0 = np.mean(y_adv[a == 0]) if np.any(a == 0) else 0
        p1 = np.mean(y_adv[a == 1]) if np.any(a == 1) else 0
        n0 = np.sum(a == 0)
        n1 = np.sum(a == 1)
        
        gains = np.full(n, -np.inf)
        
        for i in range(n):
            if flipped[i]:
                continue
            g = a[i]
            ng = n0 if g == 0 else n1
            
            if p0 >= p1:
                if g == 0:
                    gain = (1.0 / ng) if y_adv[i] == 0 else (-1.0 / ng)
                else:
                    gain = (1.0 / ng) if y_adv[i] == 1 else (-1.0 / ng)
            else:
                if g == 1:
                    gain = (1.0 / ng) if y_adv[i] == 0 else (-1.0 / ng)
                else:
                    gain = (1.0 / ng) if y_adv[i] == 1 else (-1.0 / ng)
            
            gains[i] = gain
        
        if coordinated:
            minority_mask = (a == minority_group) & ~flipped
            majority_mask = (a != minority_group) & ~flipped
            n_minority_target = int(0.7 * n_corrupt)
            n_minority_done = np.sum(flipped & (a == minority_group))
            
            best = np.argmax(gains)
            if n_minority_done < n_minority_target and np.any(minority_mask):
                best_minority = np.argmax(gains * minority_mask)
                if gains[best_minority] > 0:
                    best = best_minority
            elif np.any(majority_mask):
                best_majority = np.argmax(gains * majority_mask)
                if gains[best_majority] > 0:
                    best = best_majority
        else:
            best = np.argmax(gains)
        
        if gains[best] <= 0:
            break
        
        y_adv[best] = 1 - y_adv[best]
        flipped[best] = True
    
    return y_adv, flipped


def main():
    print("=" * 70)
    print("DP ATTACK BUG DEMONSTRATION")
    print("=" * 70)
    
    np.random.seed(42)
    n = 1000
    a = np.zeros(n, dtype=np.int64)
    a[800:] = 1
    y = np.random.binomial(1, 0.3, n).astype(np.float32)
    X_dummy = np.random.randn(n, 10).astype(np.float32)
    
    dp_before = dp_violation(y, a)
    print(f"\nDataset: n={n}, group0={np.sum(a==0)}, group1={np.sum(a==1)}")
    print(f"DP before attack: {dp_before:.4f}")
    
    alpha = 0.2
    n_corrupt = int(alpha * n)
    print(f"Corruption budget: {n_corrupt} samples (alpha={alpha})")
    
    print("\n--- Current FairnessTargetedPGD (BUGGY) ---")
    for pgd_steps in [1, 3, 5]:
        atk = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=pgd_steps,
                                   coordinated=False, random_state=42)
        _, y_buggy, _, mask_buggy = atk.corrupt(X_dummy.copy(), y.copy(), a)
        dp_buggy = dp_violation(y_buggy, a)
        n_actually_flipped = np.sum(mask_buggy)
        print(f"  pgd_steps={pgd_steps}: DP after={dp_buggy:.4f} (delta={dp_buggy-dp_before:+.4f}), "
              f"actually flipped={n_actually_flipped}/{n_corrupt}")
    
    print("\n--- Correct Greedy DP Attack ---")
    y_greedy, mask_greedy = greedy_dp_attack(y, a, alpha, coordinated=False)
    dp_greedy = dp_violation(y_greedy, a)
    print(f"  DP after={dp_greedy:.4f} (delta={dp_greedy-dp_before:+.4f}), "
          f"flipped={np.sum(mask_greedy)}/{n_corrupt}")
    
    print("\n--- With Coordinated Targeting (minority group) ---")
    atk_coord = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=1,
                                     coordinated=True, random_state=42)
    _, y_coord_buggy, _, _ = atk_coord.corrupt(X_dummy.copy(), y.copy(), a)
    dp_coord_buggy = dp_violation(y_coord_buggy, a)
    
    y_coord_greedy, _ = greedy_dp_attack(y, a, alpha, coordinated=True)
    dp_coord_greedy = dp_violation(y_coord_greedy, a)
    
    print(f"  Buggy  (pgd_steps=1, coord): DP after={dp_coord_buggy:.4f} (delta={dp_coord_buggy-dp_before:+.4f})")
    print(f"  Greedy (coord):              DP after={dp_coord_greedy:.4f} (delta={dp_coord_greedy-dp_before:+.4f})")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    if dp_greedy > dp_buggy:
        print(f"  Greedy attack increases DP by {dp_greedy-dp_before:.4f}")
        print(f"  Buggy attack increases DP by only {dp_buggy-dp_before:.4f}")
        print("  → The buggy attack is SUBOPTIMAL.")
    print("=" * 70)


if __name__ == '__main__':
    main()
