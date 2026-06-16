"""Unit tests for evaluation metrics."""

import numpy as np
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation


def test_compute_accuracy():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    assert compute_accuracy(y_true, y_pred) == 0.8


def test_compute_dp_violation_zero():
    """DP violation should be 0 when rates are equal."""
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert compute_dp_violation(y_pred, a) == 0.0


def test_compute_dp_violation_nonzero():
    """DP violation should be > 0 when rates differ."""
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    a = np.array([0, 0, 0, 1, 1, 1])
    assert compute_dp_violation(y_pred, a) == 1.0


def test_compute_dp_violation_three_groups():
    """DP for >=3 groups (UTKFace race) should be max_rate - min_rate.

    Rates: g0=1.0, g1=0.5, g2=0.0 → max-min = 1.0.
    """
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0])
    a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert compute_dp_violation(y_pred, a) == 1.0


def test_compute_dp_violation_four_groups():
    """DP for 4 groups: 0.0, 0.25, 0.5, 0.75 → max-min = 0.75."""
    y_pred = np.array([0, 0, 0, 0,
                       1, 0, 0, 0,
                       1, 1, 0, 0,
                       1, 1, 1, 0])
    a = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    assert abs(compute_dp_violation(y_pred, a) - 0.75) < 1e-9


def test_compute_dp_violation_three_groups_equal():
    """DP for 3 equal-rate groups should be 0."""
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    a = np.array([0, 0, 1, 1, 2, 2])
    assert compute_dp_violation(y_pred, a) == 0.0


def test_compute_dp_violation_binary_matches_old_behavior():
    """Binary DP should still be abs(rate_0 - rate_1) — preserves old behavior."""
    rng = np.random.RandomState(0)
    y_pred = rng.binomial(1, 0.6, 100).astype(np.float32)
    a = rng.binomial(1, 0.5, 100)
    expected = abs(y_pred[a == 0].mean() - y_pred[a == 1].mean())
    assert abs(compute_dp_violation(y_pred, a) - expected) < 1e-9


def test_compute_if_violation_zero():
    """IF violation should be 0 when all predictions are the same."""
    X = np.random.randn(10, 3)
    y_pred = np.ones(10)
    assert compute_if_violation(X, y_pred, k=3) == 0.0


def test_compute_if_violation_nonzero():
    """IF violation should be > 0 when similar points have different predictions."""
    X = np.array([[0.0, 0.0], [0.01, 0.01], [10.0, 10.0]])
    y_pred = np.array([0.0, 1.0, 0.0])
    if_viol = compute_if_violation(X, y_pred, k=2)
    assert if_viol > 0.0
