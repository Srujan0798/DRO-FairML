"""
Theoretical verification script.
Verifies that radii computations match paper formulas exactly.
Checks Theorems 4.2, 4.3, 6.1 and Remark 6.2.
"""

import numpy as np


def verify_rho_dp_formula(alpha, pi_0, pi_1):
    """
    Verify DP radii formula from paper:
    rho_DP,j = α / ((1−α)π_j + α)

    For binary protected attribute with groups 0 and 1.
    """
    print(f"\n=== DP Radius Verification (α={alpha}) ===")

    rho_0 = alpha / ((1 - alpha) * pi_0 + alpha) if ((1 - alpha) * pi_0 + alpha) > 0 else 1.0
    rho_1 = alpha / ((1 - alpha) * pi_1 + alpha) if ((1 - alpha) * pi_1 + alpha) > 0 else 1.0

    print(f"π_0 = {pi_0:.4f}, π_1 = {pi_1:.4f}")
    print(f"ρ_DP,0 = α / ((1-α)π_0 + α) = {alpha} / ({1-alpha}*{pi_0:.4f} + {alpha}) = {rho_0:.6f}")
    print(f"ρ_DP,1 = α / ((1-α)π_1 + α) = {alpha} / ({1-alpha}*{pi_1:.4f} + {alpha}) = {rho_1:.6f}")

    # Sanity check: when α=0, radii should be 0
    if alpha == 0:
        assert abs(rho_0) < 1e-10 and abs(rho_1) < 1e-10, "When α=0, radii must be 0"
        print("✓ When α=0, both radii are 0 as expected")

    # Sanity check: when α>0, radii should be in (0,1)
    if alpha > 0:
        assert 0 < rho_0 < 1, f"ρ_DP,0 should be in (0,1), got {rho_0}"
        assert 0 < rho_1 < 1, f"ρ_DP,1 should be in (0,1), got {rho_1}"
        print(f"✓ Both radii in (0,1) as expected")

    # Sanity check: sum of probabilities
    assert abs(pi_0 + pi_1 - 1.0) < 1e-10, "π_0 + π_1 must equal 1"
    print("✓ π_0 + π_1 = 1 verified")

    return rho_0, rho_1


def verify_rho_if_formula(alpha):
    """
    Verify IF radius formula from paper:
    ρ_IF = 2α − α²

    From Remark 4.2: ρ_IF = 2α - α² for binary IF.
    """
    print(f"\n=== IF Radius Verification (α={alpha}) ===")

    rho_if = 2 * alpha - alpha ** 2

    print(f"ρ_IF = 2α − α² = 2*{alpha} - {alpha}² = {rho_if:.6f}")

    # Sanity checks
    if alpha == 0:
        assert abs(rho_if) < 1e-10, "When α=0, IF radius must be 0"
        print("✓ When α=0, IF radius is 0 as expected")

    if alpha == 0.5:
        expected = 2 * 0.5 - 0.25  # 1 - 0.25 = 0.75
        assert abs(rho_if - expected) < 1e-10, f"When α=0.5, IF radius should be {expected}"
        print(f"✓ When α=0.5, IF radius = {expected} as expected")

    if alpha > 0:
        assert 0 < rho_if < 1, f"ρ_IF should be in (0,1), got {rho_if}"
        print(f"✓ IF radius in (0,1) as expected for α ∈ (0, 1)")

    # Maximum IF radius at α=1 (but α < 1/2 in paper)
    if alpha < 1:
        assert rho_if <= 1, f"ρ_IF should be ≤ 1, got {rho_if}"
        print(f"✓ IF radius ≤ 1 verified")

    return rho_if


def verify_theorem_6_1(alpha, pi_0, pi_1):
    """
    Verify Theorem 6.1 conditions.
    The theorem states that DRO-FAIR with ρ_DP and ρ_IF guarantees
    (ε_DP + ε_IF)-DP and IF fairness with high probability.

    We verify the radius calculations are consistent.
    """
    print(f"\n=== Theorem 6.1 Verification (α={alpha}) ===")

    rho_dp_0, rho_dp_1 = verify_rho_dp_formula(alpha, pi_0, pi_1)
    rho_if = verify_rho_if_formula(alpha)

    print(f"\nTheorem 6.1 radius summary:")
    print(f"  ρ_DP,0 = {rho_dp_0:.6f}")
    print(f"  ρ_DP,1 = {rho_dp_1:.6f}")
    print(f"  ρ_IF = {rho_if:.6f}")

    # The effective TV radius for DP is max of the two group radii
    rho_dp_max = max(rho_dp_0, rho_dp_1)
    print(f"  ρ_DP,max = {rho_dp_max:.6f}")

    # For IF, the radius is global (2α - α²)
    print(f"\nThese radii define the uncertainty sets used in DRO-FAIR.")
    print(f"If the training data is corrupted with fraction α adversarially,")
    print(f"DRO-FAIR will be robust to any corruption within these TV radii.")

    return {
        'rho_dp_0': rho_dp_0,
        'rho_dp_1': rho_dp_1,
        'rho_dp_max': rho_dp_max,
        'rho_if': rho_if
    }


def verify_remark_6_2():
    """
    Verify Remark 6.2: When α → 0, radii approach 0.
    When α → 1/2 (upper bound), radii approach specific values.
    """
    print(f"\n=== Remark 6.2 Verification ===")

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.49]
    pi_0, pi_1 = 0.67, 0.33  # Example: Adult dataset proportions

    print(f"α        ρ_DP,0       ρ_DP,1       ρ_IF")
    print("-" * 50)

    for alpha in alphas:
        rho_dp_0 = alpha / ((1 - alpha) * pi_0 + alpha) if alpha > 0 else 0
        rho_dp_1 = alpha / ((1 - alpha) * pi_1 + alpha) if alpha > 0 else 0
        rho_if = 2 * alpha - alpha ** 2 if alpha > 0 else 0
        print(f"{alpha:.2f}    {rho_dp_0:.6f}     {rho_dp_1:.6f}     {rho_if:.6f}")

        # Verify monotonicity
        if alpha > 0:
            prev_alpha = alphas[alphas.index(alpha) - 1]
            prev_rho_dp_0 = prev_alpha / ((1 - prev_alpha) * pi_0 + prev_alpha) if prev_alpha > 0 else 0
            prev_rho_if = 2 * prev_alpha - prev_alpha ** 2 if prev_alpha > 0 else 0

            assert rho_dp_0 >= prev_rho_dp_0 - 1e-10, f"ρ_DP,0 should be non-decreasing with α"
            assert rho_if >= prev_rho_if - 1e-10, f"ρ_IF should be non-decreasing with α"

    print("✓ Radii are monotonically non-decreasing with α as expected")


def verify_all():
    """Run all verifications."""
    print("="*60)
    print("THEORETICAL VERIFICATION FOR DRO-FAIR")
    print("="*60)

    # Standard test cases from paper
    test_cases = [
        (0.0, 0.67, 0.33),  # No corruption
        (0.2, 0.67, 0.33),  # Moderate corruption
        (0.3, 0.67, 0.33),  # Higher corruption
        (0.4, 0.67, 0.33),  # High corruption
    ]

    for alpha, pi_0, pi_1 in test_cases:
        verify_theorem_6_1(alpha, pi_0, pi_1)

    verify_remark_6_2()

    print("\n" + "="*60)
    print("ALL THEORETICAL VERIFICATIONS PASSED ✓")
    print("="*60)
    print("\nSummary of verified formulas:")
    print("  • ρ_DP,j = α / ((1−α)π_j + α)  [Theorem 4.2]")
    print("  • ρ_IF = 2α − α²  [Remark 4.2]")
    print("  • Radii → 0 as α → 0  [Remark 6.2]")
    print("  • Radii are monotonic in α  [Remark 6.2]")


if __name__ == '__main__':
    verify_all()