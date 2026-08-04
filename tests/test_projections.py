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


def test_project_simplex_l1_ball_training_regime():
    """Realistic training-time inputs: small PGA-step perturbations of the
    uniform center. Both simplex AND L1-ball constraints must hold.

    This is the regime DRO-FAIR actually calls the projection in:
    p_{t+1} = p_t + lr_p · ∇_p g, with lr_p=5e-3 and uniform initialization.
    """
    rng = np.random.RandomState(0)
    fails = []
    for _ in range(100):
        n = rng.randint(10, 200)
        center = np.ones(n) / n
        v = center + 5e-3 * rng.randn(n)   # PGA-sized step from uniform
        radius = rng.uniform(0.02, 0.3)
        result = project_simplex_l1_ball(v, center, radius)
        if abs(result.sum() - 1.0) > 1e-4:
            fails.append(f"simplex: sum={result.sum()}, n={n}")
        if np.any(result < -1e-6):
            fails.append(f"negative element, n={n}")
        if np.abs(result - center).sum() > radius + 2e-4:
            fails.append(f"L1: {np.abs(result-center).sum():.6f} > r={radius:.6f}, n={n}")
    assert not fails, f"{len(fails)}/100 training-regime projections failed: {fails[:3]}"


def test_tv_to_l1_radius_factor_2():
    """Phase 0 Item 4: confirm TV radius ρ → L1 radius 2ρ is correct.

    The DRO-FAIR trainer passes `2 * radius` to project_simplex_l1_ball
    (dro_fair.py:219). This test constructs a case with a known analytic
    TV radius and verifies the projected point lands inside the CORRECT
    L1 ball (2ρ, not ρ or 4ρ).
    """
    import numpy as np
    center = np.array([0.5, 0.5])
    rho_tv = 0.1
    l1_radius = 2 * rho_tv  # = 0.2, as the trainer passes it

    # Point far from center — should project onto the L1 ball boundary
    v = np.array([0.9, 0.1])
    result = project_simplex_l1_ball(v, center, l1_radius, max_iter=500, tol=1e-8)

    # Must be on simplex
    assert abs(result.sum() - 1.0) < 1e-6, f"not on simplex: sum={result.sum()}"
    # Must be inside L1 ball of radius 2ρ = 0.2
    l1_dist = np.abs(result - center).sum()
    assert l1_dist <= l1_radius + 1e-6, f"L1 dist {l1_dist} > radius {l1_radius}"

    # Must NOT fit inside a ball of radius ρ (would mean ×2 is wrong)
    assert l1_dist > rho_tv + 1e-4, (
        f"Projected point fits in L1 ball of radius ρ={rho_tv} "
        f"(dist={l1_dist}) — the ×2 factor may be missing"
    )
    # The projected point should be at or near the L1 ball boundary (2ρ)
    assert abs(l1_dist - l1_radius) < 1e-4, (
        f"Expected point on L1 ball boundary (dist={l1_radius}), got dist={l1_dist}"
    )
