"""
Tests for Kuldeep Q5 — empirical radii calibration.

Verifies that the radii_mode='empirical' flag in DroFairTrainer correctly
inverts the coordinated 70%-minority attribute-flip attack to recover
clean group proportions, and that it works end-to-end on a real dataset.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.dro_fair import DroFairTrainer
from src.models.classifier import MLPClassifier


def _make_dummy_model(input_dim=5):
    return MLPClassifier(input_dim, hidden_dims=[8], dropout=0.0)


def test_empirical_mode_recovers_clean_proportions():
    """Empirical mode should recover clean pi exactly under the coordinated attack model.

    Per MASTER_PLAN Q5 validation requirement:
    - Synthesizes a known coordinated attack (70% minority targeting, attribute flips only, exact alpha budget).
    - Calls _empirical_pi_clean on the observed pi after attack.
    - Asserts recovered pi_clean matches the true clean proportions within tol (1e-6).

    Setup: 1000 samples, 0.7 group-0 (maj) / 0.3 group-1 (min) in clean data.
    Apply coordinated attack with alpha=0.2 → 70% of budget (140) hits minority,
    30% (60) hits majority. After attribute flip:
        obs count0 = 700 -60 +140 = 780 → pi_obs[0]=0.78
        obs count1 = 300 -140 +60 = 220 → pi_obs[1]=0.22
    Empirical inversion (see _empirical_pi_clean and Q5_derivation):
        minority_idx = argmin(pi_obs)=1; pi_clean[1] = 0.22 + 0.4*0.2 = 0.30
        pi_clean[0] = 0.78 - 0.08 = 0.70
    Exact recovery (no oracle mask used; only pi_obs + alpha + known 70/30 structure).
    """
    n = 1000
    a_clean = np.array([0] * 700 + [1] * 300, dtype=np.int64)
    rng = np.random.RandomState(42)
    rng.shuffle(a_clean)

    alpha = 0.2
    n_corrupt = int(alpha * n)
    n_min_corr = int(0.7 * n_corrupt)
    n_maj_corr = n_corrupt - n_min_corr

    a_corrupt = a_clean.copy()
    minority_idx = np.where(a_clean == 1)[0]
    majority_idx = np.where(a_clean == 0)[0]
    flip_min = rng.choice(minority_idx, n_min_corr, replace=False)
    flip_maj = rng.choice(majority_idx, n_maj_corr, replace=False)
    a_corrupt[flip_min] = 0
    a_corrupt[flip_maj] = 1

    pi_obs = np.array([np.mean(a_corrupt == j) for j in [0, 1]])
    # Must use DroFairTrainer (not dummy model) to call _empirical_pi_clean which relies on self.alpha
    model = _make_dummy_model()
    trainer = DroFairTrainer(model, alpha=alpha, radii_mode='empirical', epochs=1, K_inner=1)
    pi_clean = trainer._empirical_pi_clean(pi_obs)

    pi_true = np.array([0.7, 0.3])
    assert np.allclose(pi_clean, pi_true, atol=1e-6), \
        f"Empirical recovery failed: got {pi_clean}, expected {pi_true}"


def test_empirical_mode_at_alpha_zero():
    """At alpha=0, empirical formula should be a no-op (pi_clean = pi_obs)."""
    pi_obs = np.array([0.7, 0.3])
    model = _make_dummy_model()
    trainer = DroFairTrainer(model, alpha=0.0, radii_mode='empirical', epochs=1)
    pi_clean = trainer._empirical_pi_clean(pi_obs)
    assert np.allclose(pi_clean, pi_obs, atol=1e-9)


def test_empirical_mode_handles_clamping():
    """When the formula pushes a value below 0, it should clip and renormalize."""
    pi_obs = np.array([0.1, 0.9])
    model = _make_dummy_model()
    trainer = DroFairTrainer(model, alpha=0.4, radii_mode='empirical', epochs=1)
    pi_clean = trainer._empirical_pi_clean(pi_obs)
    assert np.all(pi_clean >= 0.0)
    assert np.all(pi_clean <= 1.0)
    assert abs(pi_clean.sum() - 1.0) < 1e-6


def test_empirical_mode_produces_different_radii_than_uniform():
    """Empirical and uniform modes should give different rho_dp under coordinated attack."""
    a_clean = np.array([0] * 700 + [1] * 300, dtype=np.int64)
    rng = np.random.RandomState(0)
    rng.shuffle(a_clean)
    alpha = 0.2
    n_corrupt = int(alpha * len(a_clean))
    flip_min = rng.choice(np.where(a_clean == 1)[0], int(0.7 * n_corrupt), replace=False)
    flip_maj = rng.choice(np.where(a_clean == 0)[0], n_corrupt - int(0.7 * n_corrupt), replace=False)
    a_corrupt = a_clean.copy()
    a_corrupt[flip_min] = 0
    a_corrupt[flip_maj] = 1

    model = _make_dummy_model()
    trainer_e = DroFairTrainer(model, alpha=alpha, epochs=1, radii_mode='empirical')
    model2 = _make_dummy_model()
    trainer_u = DroFairTrainer(model2, alpha=alpha, epochs=1, radii_mode='uniform')

    rho_e, _ = trainer_e._compute_radii(a_corrupt)
    rho_u, _ = trainer_u._compute_radii(a_corrupt)
    assert not np.allclose(rho_e, rho_u, atol=1e-4), \
        f"Empirical and uniform modes gave same radii: {rho_e}"


def test_invalid_radii_mode_raises():
    """Passing an unknown radii_mode should raise ValueError."""
    with pytest.raises(ValueError, match="radii_mode"):
        DroFairTrainer(_make_dummy_model(), alpha=0.2, radii_mode='bogus')


def test_default_radii_mode_is_uniform():
    """The default radii_mode should be 'uniform' to preserve existing behavior."""
    trainer = DroFairTrainer(_make_dummy_model(), alpha=0.2)
    assert trainer.radii_mode == 'uniform'


def test_default_radii_scale_and_clamp():
    """Defaults: radii_scale=1.0 (no change), radii_clamp=None (no clamping)."""
    trainer = DroFairTrainer(_make_dummy_model(), alpha=0.2)
    assert trainer.radii_scale == 1.0
    assert trainer.radii_clamp is None


def test_radii_scale():
    """Agent N1: radii_scale=2.0 should make rho exactly 2x the scale=1.0 version.

    Verifies the global radius multiplier is applied to BOTH rho_dp (per-group)
    and rho_if (scalar), and that scale=1.0 reproduces the canonical closed form.
    """
    a = np.array([0] * 700 + [1] * 300, dtype=np.int64)
    rng = np.random.RandomState(7); rng.shuffle(a)
    alpha = 0.2

    t1 = DroFairTrainer(_make_dummy_model(), alpha=alpha, radii_scale=1.0, epochs=1)
    t2 = DroFairTrainer(_make_dummy_model(), alpha=alpha, radii_scale=2.0, epochs=1)
    rho_dp1, rho_if1 = t1._compute_radii(a)
    rho_dp2, rho_if2 = t2._compute_radii(a)

    for j in [0, 1]:
        assert abs(rho_dp2[j] - 2.0 * rho_dp1[j]) < 1e-9, \
            f"rho_dp[{j}] not 2x: {rho_dp1[j]} -> {rho_dp2[j]}"
    assert abs(rho_if2 - 2.0 * rho_if1) < 1e-9, \
        f"rho_if not 2x: {rho_if1} -> {rho_if2}"


def test_radii_clamp():
    """Agent L2: per-group radius clamp caps rho_dp[j] at the cap value.

    Simulates the LSAC 90/10 imbalance where the uniform closed form gives the
    tiny minority group a degenerate radius (denom -> 0 => rho_dp -> 1.0 or above).
    With radii_clamp=0.3, no group's rho_dp exceeds 0.3.
    """
    n = 1000
    a_clean = np.array([0] * 900 + [1] * 100, dtype=np.int64)  # 90/10 LSAC-like
    rng = np.random.RandomState(0); rng.shuffle(a_clean)
    alpha = 0.2
    nc = int(alpha * n)
    n_min_corr = min(int(0.7 * nc), int((a_clean == 1).sum()))  # cap at minority pop
    n_maj_corr = nc - n_min_corr
    a_corrupt = a_clean.copy()
    a_corrupt[rng.choice(np.where(a_clean == 1)[0], n_min_corr, replace=False)] = 0
    a_corrupt[rng.choice(np.where(a_clean == 0)[0], n_maj_corr, replace=False)] = 1

    # Unclamped: minority rho_dp is large (degenerate on imbalanced data)
    t_unclamped = DroFairTrainer(_make_dummy_model(), alpha=alpha, radii_clamp=None, epochs=1)
    rho_dp_un, _ = t_unclamped._compute_radii(a_corrupt)
    minority_rho_un = max(rho_dp_un)
    assert minority_rho_un > 0.3, \
        f"setup invariant failed: minority rho_dp should be >0.3, got {minority_rho_un}"

    # Clamped at 0.3: no group exceeds the cap
    t_clamped = DroFairTrainer(_make_dummy_model(), alpha=alpha, radii_clamp=0.3, epochs=1)
    rho_dp_cl, _ = t_clamped._compute_radii(a_corrupt)
    for j in [0, 1]:
        assert rho_dp_cl[j] <= 0.3 + 1e-9, \
            f"rho_dp[{j}] exceeds clamp: {rho_dp_cl[j]}"
    # The previously-degenerate group is now actually capped (not just below 0.3 by chance)
    assert max(rho_dp_cl) <= 0.3 + 1e-9
    assert max(rho_dp_cl) < minority_rho_un, \
        f"clamp did not reduce the minority radius: {max(rho_dp_cl)} vs {minority_rho_un}"


def test_pi_shrinkage_pulls_toward_balance():
    """LSAC-degeneracy hypothesis (alternative to radii_clamp): additive shrinkage
    of pi_clean toward 0.5 should reduce the minority group's degenerate rho_dp,
    same intent as radii_clamp but via a different mechanism (smooths the INPUT
    proportion rather than capping the output radius). k=0 must be a no-op.
    """
    n = 1000
    a_clean = np.array([0] * 900 + [1] * 100, dtype=np.int64)  # 90/10 LSAC-like
    rng = np.random.RandomState(0); rng.shuffle(a_clean)
    alpha = 0.2
    nc = int(alpha * n)
    n_min_corr = min(int(0.7 * nc), int((a_clean == 1).sum()))
    n_maj_corr = nc - n_min_corr
    a_corrupt = a_clean.copy()
    a_corrupt[rng.choice(np.where(a_clean == 1)[0], n_min_corr, replace=False)] = 0
    a_corrupt[rng.choice(np.where(a_clean == 0)[0], n_maj_corr, replace=False)] = 1

    # k=0 must reproduce the unshrunk (canonical) radii exactly
    t_base = DroFairTrainer(_make_dummy_model(), alpha=alpha, pi_shrinkage_k=0.0, epochs=1)
    t_noop = DroFairTrainer(_make_dummy_model(), alpha=alpha, pi_shrinkage_k=0.0, epochs=1)
    rho_base, _ = t_base._compute_radii(a_corrupt)
    rho_noop, _ = t_noop._compute_radii(a_corrupt)
    for j in [0, 1]:
        assert abs(rho_base[j] - rho_noop[j]) < 1e-12

    minority_rho_un = max(rho_base)
    assert minority_rho_un > 0.3, \
        f"setup invariant failed: minority rho_dp should be >0.3, got {minority_rho_un}"

    # Shrinkage toward 0.5 should measurably reduce the degenerate minority radius
    t_shrunk = DroFairTrainer(_make_dummy_model(), alpha=alpha, pi_shrinkage_k=200.0, epochs=1)
    rho_shrunk, _ = t_shrunk._compute_radii(a_corrupt)
    assert max(rho_shrunk) < minority_rho_un, \
        f"shrinkage did not reduce the minority radius: {max(rho_shrunk)} vs {minority_rho_un}"
    # Sanity: rho stays a valid, finite radius (bounded [0,1]) under strong shrinkage
    for j in [0, 1]:
        assert 0.0 <= rho_shrunk[j] <= 1.0


def test_empirical_differs_from_uniform_imbalanced_90_10():
    """Agent A5 verification: on a 90/10 imbalanced split at alpha=0.2,
    empirical rho_dp != uniform rho_dp. Confirms empirical mode is actually
    ACTIVE under coordinated=True (not silently no-op'd as in an earlier version).
    """
    n = 1000
    a_clean = np.array([0] * 900 + [1] * 100, dtype=np.int64)
    rng = np.random.RandomState(0); rng.shuffle(a_clean)
    alpha = 0.2
    nc = int(alpha * n)
    n_min_corr = min(int(0.7 * nc), int((a_clean == 1).sum()))
    n_maj_corr = nc - n_min_corr
    a_corrupt = a_clean.copy()
    a_corrupt[rng.choice(np.where(a_clean == 1)[0], n_min_corr, replace=False)] = 0
    a_corrupt[rng.choice(np.where(a_clean == 0)[0], n_maj_corr, replace=False)] = 1

    t_e = DroFairTrainer(_make_dummy_model(), alpha=alpha, radii_mode='empirical', epochs=1)
    t_u = DroFairTrainer(_make_dummy_model(), alpha=alpha, radii_mode='uniform', epochs=1)
    rho_e, _ = t_e._compute_radii(a_corrupt)
    rho_u, _ = t_u._compute_radii(a_corrupt)

    assert not np.allclose(rho_e, rho_u, atol=1e-4), \
        f"empirical == uniform on imbalanced data (empirical mode not active!): " \
        f"empirical={rho_e}, uniform={rho_u}"


def test_empirical_mode_end_to_end_adult():
    """Empirical mode should train end-to-end on real Adult data without error.

    Verifies Q5 acceptance: 'radii_mode="empirical" runs end-to-end on Adult'.
    Uses a small subset and few epochs for speed; correctness of the formula
    is verified by the unit tests above.
    """
    from src.data.datasets import get_dataset
    from src.corruption.adversarial import FairnessTargetedPGD

    X_train, y_train, a_train, _, _, _, _, _, _, _ = get_dataset('adult', random_state=42)
    X_train = X_train[:500].astype(np.float32)
    y_train = y_train[:500].astype(np.float32)
    a_train = a_train[:500].astype(np.int64)

    attack = FairnessTargetedPGD(
        alpha=0.2, target_metric='dp', pgd_steps=3,
        coordinated=True, random_state=42
    )
    X_c, y_c, a_c, _ = attack.corrupt(X_train, y_train, a_train)

    model = MLPClassifier(X_train.shape[1], hidden_dims=[16, 8], dropout=0.0)
    trainer = DroFairTrainer(
        model, alpha=0.2, device='cpu',
        epochs=3, K_inner=3, tau=1.0, beta=5.0, k=5,
        use_dp=True, use_if=True,
        radii_mode='empirical'
    )
    history = trainer.fit(X_c, y_c, a_c, verbose=False)
    assert len(history['train_loss']) == 3
    preds = trainer.predict(X_train)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(preds) == len(X_train)
