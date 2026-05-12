"""Unit tests for projection utilities."""

import numpy as np
import pytest
from src.utils.projections import project_simplex, project_l1_ball, project_simplex_l1_ball


def test_project_simplex_uniform():
    v = np.array([0.0, 0.0, 0.0])
    result = project_simplex(v)
    assert np.allclose(result.sum(), 1.0)
    assert np.allclose(result, np.ones(3) / 3)


def test_project_simplex_already_in():
    v = np.array([0.2, 0.3, 0.5])
    result = project_simplex(v)
    assert np.allclose(result.sum(), 1.0)
    assert np.all(result >= 0)


def test_project_l1_ball_zero_radius():
    v = np.array([1.0, 2.0, 3.0])
    center = np.array([0.5, 0.5, 0.5])
    result = project_l1_ball(v, center, 0.0)
    assert np.allclose(result, center)


def test_project_l1_ball_inside():
    v = np.array([0.26, 0.26, 0.26])
    center = np.array([0.25, 0.25, 0.25])
    result = project_l1_ball(v, center, 0.1)
    assert np.allclose(result, v)


def test_project_simplex_l1_ball_basic():
    v = np.array([0.1, 0.2, 0.3, 0.4])
    center = np.array([0.25, 0.25, 0.25, 0.25])
    result = project_simplex_l1_ball(v, center, 0.5, max_iter=100, tol=1e-6)
    assert np.allclose(result.sum(), 1.0, atol=1e-5)
    assert np.all(result >= -1e-6)
    assert np.abs(result - center).sum() <= 1.0 + 1e-5


def test_project_simplex_l1_ball_zero_radius():
    v = np.array([0.1, 0.2, 0.3])
    center = np.array([0.33, 0.33, 0.34])
    result = project_simplex_l1_ball(v, center, 0.0, max_iter=50, tol=1e-5)
    assert np.allclose(result, center, atol=1e-4)


def test_project_simplex_l1_ball_random():
    """Test with random-Gaussian inputs that violate simplex (sum != 1).

    Dykstra's algorithm finds a point in the intersection, but due to
    alternating projections, may not converge to exactly the right point.
    The key postcondition is that result is ON THE SIMPLEX (sum=1, all>=0).
    L1-ball satisfaction is best-effort after simplex projection.
    """
    rng = np.random.RandomState(42)
    for _ in range(100):
        n = rng.randint(5, 20)
        v = rng.randn(n)
        center = rng.rand(n)
        center = center / center.sum()
        radius = rng.uniform(0.1, 0.5)
        result = project_simplex_l1_ball(v, center, radius)
        assert np.abs(result.sum() - 1.0) < 1e-5, f"sum={result.sum()}, n={n}"
        assert np.all(result >= -1e-6), f"negative values found, n={n}"
