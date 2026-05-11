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
