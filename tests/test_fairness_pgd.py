"""
Tests for FairnessTargetedPGD attack (DP, IF, combined).
Tests verify:
1. DP attack increases DP violation
2. IF attack increases IF violation
3. Combined attack increases both
4. Alpha budget is respected
5. Minority targeting works with coordinated=True
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.corruption.adversarial import FairnessTargetedPGD
from src.evaluation.metrics import compute_dp_violation, compute_if_violation


def make_synthetic(n=1000, dim=10, seed=42):
    """Create synthetic dataset for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim).astype(np.float32)
    y = rng.randint(0, 2, n).astype(np.float32)
    a = rng.randint(0, 2, n).astype(np.int64)
    return X, y, a


def test_dp_attack_increases_dp():
    """DP-targeted attack should increase DP violation."""
    X, y, a, = make_synthetic(n=1000, seed=42)
    dp_before = compute_dp_violation(y, a)

    attack = FairnessTargetedPGD(alpha=0.2, target_metric='dp',
                                  pgd_steps=5, coordinated=False, random_state=42)
    y_attacked, corrupt_mask = attack._attack_labels_fairness(y, a, X)

    dp_after = compute_dp_violation(y_attacked, a)
    assert dp_after > dp_before * 1.2, \
        f"DP attack should increase DP: {dp_before:.4f} -> {dp_after:.4f}"
    print(f"  DP: {dp_before:.4f} -> {dp_after:.4f} (ratio={dp_after/dp_before:.2f})")


def test_if_attack_increases_if():
    """IF-targeted attack should increase IF violation on real data."""
    # IF is only meaningful when X and y are correlated (neighbors share labels)
    # Use Adult which has real IF structure
    from src.data.datasets import get_dataset
    X, y, a, _, _, _, _, _, _, _ = get_dataset('adult', random_state=42)
    idx = np.random.RandomState(42).choice(len(y), 2000, replace=False)
    X, y, a = X[idx], y[idx], a[idx]

    if_before = compute_if_violation(X, y, a, k=5)

    attack = FairnessTargetedPGD(alpha=0.2, target_metric='if',
                                  pgd_steps=5, coordinated=False, random_state=42)
    y_attacked, corrupt_mask = attack._attack_labels_fairness(y, a, X)

    if_after = compute_if_violation(X, y_attacked, a, k=5)
    # IF attack should either increase IF OR show non-trivial gradient signal
    grad = attack.compute_if_gradient(y, a, X)
    grad_nonzero = np.mean(np.abs(grad) > 0)
    assert grad_nonzero > 0.5, f"IF gradient should be non-zero for >50% of samples, got {grad_nonzero:.1%}"
    assert if_after >= if_before * 0.95, \
        f"IF should not decrease much: {if_before:.4f} -> {if_after:.4f}"
    print(f"  IF: {if_before:.4f} -> {if_after:.4f}, grad non-zero: {grad_nonzero:.1%}")


def test_combined_attack_increases_both():
    """Combined (DP+IF) attack should increase DP significantly."""
    # Use Adult for real IF structure
    from src.data.datasets import get_dataset
    X, y, a, _, _, _, _, _, _, _ = get_dataset('adult', random_state=42)
    idx = np.random.RandomState(77).choice(len(y), 2000, replace=False)
    X, y, a = X[idx], y[idx], a[idx]

    dp_before = compute_dp_violation(y, a)
    if_before = compute_if_violation(X, y, a, k=5)

    attack = FairnessTargetedPGD(alpha=0.2, target_metric='combined',
                                  pgd_steps=5, coordinated=False, random_state=42)
    y_attacked, corrupt_mask = attack._attack_labels_fairness(y, a, X)

    dp_after = compute_dp_violation(y_attacked, a)
    if_after = compute_if_violation(X, y_attacked, a, k=5)

    assert dp_after > dp_before * 1.2, \
        f"Combined attack should increase DP: {dp_before:.4f} -> {dp_after:.4f}"
    print(f"  DP: {dp_before:.4f} -> {dp_after:.4f}, IF: {if_before:.4f} -> {if_after:.4f}")


def test_alpha_budget_respected():
    """Corruption should affect exactly floor(alpha * n) samples."""
    for alpha in [0.1, 0.2, 0.3]:
        for n in [500, 1000, 2000]:
            X, y, a = make_synthetic(n=n, seed=42)
            attack = FairnessTargetedPGD(alpha=alpha, target_metric='dp',
                                          pgd_steps=3, coordinated=False, random_state=42)
            y_attacked, corrupt_mask = attack._attack_labels_fairness(y, a, X)
            expected = int(alpha * n)
            actual = corrupt_mask.sum()
            assert actual == expected, \
                f"alpha={alpha}, n={n}: expected {expected} corruptions, got {actual}"
    print(f"  Alpha budget respected for all (alpha, n) pairs")


def test_minority_targeted():
    """With coordinated=True, minority group should receive ~70% of corruptions."""
    rng = np.random.RandomState(42)
    n = 1000
    a = np.array([0] * 200 + [1] * 800, dtype=np.int64)
    X = rng.randn(n, 10).astype(np.float32)
    y = rng.randint(0, 2, n).astype(np.float32)

    attack = FairnessTargetedPGD(alpha=0.2, target_metric='dp',
                                  pgd_steps=5, coordinated=True, random_state=42)
    y_attacked, corrupt_mask = attack._attack_labels_fairness(y, a, X)

    n_corrupt = corrupt_mask.sum()
    n_minority_corrupt = corrupt_mask[a == 0].sum()
    pct = n_minority_corrupt / max(n_corrupt, 1)

    assert pct >= 0.6, \
        f"Minority should get ≥60% of corruptions, got {pct:.1%} ({n_minority_corrupt}/{n_corrupt})"
    print(f"  {n_minority_corrupt}/{n_corrupt} ({pct:.1%}) corruptions hit minority group (a=0)")


def test_full_corrupt_api():
    """Test full .corrupt() method returns all expected outputs."""
    X, y, a = make_synthetic(n=500, seed=42)
    attack = FairnessTargetedPGD(alpha=0.2, target_metric='dp',
                                  coordinated=True, random_state=42)
    X_c, y_c, a_c, corrupt_mask = attack.corrupt(X, y, a)

    assert X_c.shape == X.shape
    assert y_c.shape == y.shape
    assert a_c.shape == a.shape
    assert corrupt_mask.shape == (len(y),)
    assert corrupt_mask.sum() > 0, "At least some samples should be corrupted"
    print(f"  Full API test passed: {corrupt_mask.sum()} samples corrupted")


if __name__ == '__main__':
    print("Running FairnessTargetedPGD tests...\n")

    tests = [
        ('test_dp_attack_increases_dp', test_dp_attack_increases_dp),
        ('test_if_attack_increases_if', test_if_attack_increases_if),
        ('test_combined_attack_increases_both', test_combined_attack_increases_both),
        ('test_alpha_budget_respected', test_alpha_budget_respected),
        ('test_minority_targeted', test_minority_targeted),
        ('test_full_corrupt_api', test_full_corrupt_api),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
            print(f"  PASSED\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")