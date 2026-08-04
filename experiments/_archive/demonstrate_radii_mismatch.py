#!/usr/bin/env python3
"""Demonstrate radii mismatch on Adult dataset.

Shows that DRO's _compute_radii formula assumes uniform corruption,
but coordinated targeting produces observed group proportions that
cause the formula to estimate wildly wrong clean proportions.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.datasets import get_dataset


def simulate_coordinated_attack(a, alpha=0.2, minority_frac=0.7, seed=0):
    """Simulate coordinated attribute flip attack.
    
    minority_frac: fraction of corruption budget targeting minority group.
    """
    np.random.seed(seed)
    a = a.flatten()
    n = len(a)
    n_corrupt = int(alpha * n)
    
    # Identify minority group
    g0 = (a == 0).sum()
    g1 = (a == 1).sum()
    minority_g = 0 if g0 < g1 else 1
    majority_g = 1 - minority_g
    
    minority_idx = np.where(a == minority_g)[0]
    majority_idx = np.where(a == majority_g)[0]
    
    # Target minority
    n_target_minority = int(minority_frac * n_corrupt)
    n_target_majority = n_corrupt - n_target_minority
    
    a_corrupt = a.copy()
    
    # Flip minority -> majority
    flipped_m = np.random.choice(minority_idx, size=min(n_target_minority, len(minority_idx)), replace=False)
    a_corrupt[flipped_m] = majority_g
    
    # Flip majority -> minority
    flipped_M = np.random.choice(majority_idx, size=min(n_target_majority, len(majority_idx)), replace=False)
    a_corrupt[flipped_M] = minority_g
    
    return a_corrupt


def dro_formula_estimate(pi_obs, alpha):
    """DRO's formula for estimating clean proportions from observed."""
    pi_clean = np.zeros_like(pi_obs)
    for j in range(len(pi_obs)):
        if alpha != 0.5:
            pi_clean[j] = (pi_obs[j] - alpha) / (1 - 2 * alpha)
        else:
            pi_clean[j] = pi_obs[j]
    return np.clip(pi_clean, 0.01, 0.99)


def main():
    print("=" * 70)
    print("DRO Radii Mismatch Demonstration")
    print("=" * 70)
    
    # Load Adult
    X_train, y_train, a_train, *_ = get_dataset('adult')
    a_train = a_train.flatten()
    n = len(a_train)
    
    # Group names
    g0_name = "Female"  # minority
    g1_name = "Male"    # majority
    
    print(f"\nDataset: Adult (n={n})")
    
    # Clean proportions
    pi_clean_true = np.array([
        (a_train == 0).mean(),
        (a_train == 1).mean()
    ])
    print(f"\nTrue clean group proportions:")
    print(f"  {g0_name}: {pi_clean_true[0]:.3f}")
    print(f"  {g1_name}: {pi_clean_true[1]:.3f}")
    
    for alpha in [0.1, 0.2, 0.3]:
        print(f"\n{'-' * 70}")
        print(f"Attack: α = {alpha}, coordinated (70% minority targeting)")
        print(f"{'-' * 70}")
        
        a_corrupt = simulate_coordinated_attack(a_train, alpha=alpha, minority_frac=0.7)
        
        # Observed after attack
        pi_obs = np.array([
            (a_corrupt == 0).mean(),
            (a_corrupt == 1).mean()
        ])
        print(f"Observed group proportions after attack:")
        print(f"  {g0_name}: {pi_obs[0]:.3f} (was {pi_clean_true[0]:.3f})")
        print(f"  {g1_name}: {pi_obs[1]:.3f} (was {pi_clean_true[1]:.3f})")
        
        # DRO's estimate
        pi_est = dro_formula_estimate(pi_obs, alpha)
        print(f"\nDRO formula ESTIMATES clean as:")
        print(f"  {g0_name}: {pi_est[0]:.3f}  (error: {abs(pi_est[0]-pi_clean_true[0]):.3f})")
        print(f"  {g1_name}: {pi_est[1]:.3f}  (error: {abs(pi_est[1]-pi_clean_true[1]):.3f})")
        
        # Radii computed by DRO
        rho = alpha / ((1 - alpha) * pi_est + alpha)
        print(f"\nDRO computes radii:")
        print(f"  {g0_name}: ρ = {rho[0]:.3f}")
        print(f"  {g1_name}: ρ = {rho[1]:.3f}")
        
        # What radii SHOULD be (using true clean proportions)
        rho_true = alpha / ((1 - alpha) * pi_clean_true + alpha)
        print(f"\nCorrect radii (using true clean proportions):")
        print(f"  {g0_name}: ρ = {rho_true[0]:.3f}")
        print(f"  {g1_name}: ρ = {rho_true[1]:.3f}")
        
        # Impact
        print(f"\n>>> DRO's uncertainty set is centered on the WRONG distribution.")
        if pi_est[0] < pi_clean_true[0]:
            print(f">>> It thinks {g0_name} is rarer than reality -> underestimates their importance.")
        else:
            print(f">>> It thinks {g0_name} is more common than reality -> overestimates their importance.")
    
    print(f"\n{'=' * 70}")
    print("Conclusion: DRO's defense is miscalibrated because the radii formula")
    print("assumes uniform corruption, but the attack uses coordinated targeting.")
    print("This is a RESEARCH DESIGN mismatch, not a code bug.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
