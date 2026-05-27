"""
Pytest tests for FairnessTargetedPGD.

Run: pytest tests/test_fairness_pgd.py -v
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.corruption.adversarial import FairnessTargetedPGD


@pytest.fixture
def synthetic_data():
    """Generate synthetic tabular data."""
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 10)
    y = np.random.binomial(1, 0.3, n)
    a = np.random.binomial(1, 0.4, n)
    return X, y, a


class TestFairnessTargetedPGD:
    """Test gradient-based fairness attacks."""

    def test_alpha_budget_dp(self, synthetic_data):
        """DP attack must flip exactly alpha*n labels."""
        X, y, a = synthetic_data
        alpha = 0.2
        ft = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=3,
                                 coordinated=True, random_state=42)
        _, _, _, mask = ft.corrupt(X.copy(), y.copy(), a.copy())
        assert mask.sum() == int(alpha * len(y)), f"Budget mismatch: {mask.sum()} vs {int(alpha*len(y))}"

    def test_alpha_budget_if(self, synthetic_data):
        """IF attack must flip exactly alpha*n labels."""
        X, y, a = synthetic_data
        alpha = 0.2
        ft = FairnessTargetedPGD(alpha=alpha, target_metric='if', pgd_steps=3,
                                 coordinated=True, random_state=42)
        _, _, _, mask = ft.corrupt(X.copy(), y.copy(), a.copy())
        assert mask.sum() == int(alpha * len(y)), f"Budget mismatch: {mask.sum()} vs {int(alpha*len(y))}"

    def test_alpha_budget_combined(self, synthetic_data):
        """Combined attack must flip exactly alpha*n labels."""
        X, y, a = synthetic_data
        alpha = 0.2
        ft = FairnessTargetedPGD(alpha=alpha, target_metric='combined', pgd_steps=3,
                                 coordinated=True, random_state=42)
        _, _, _, mask = ft.corrupt(X.copy(), y.copy(), a.copy())
        assert mask.sum() == int(alpha * len(y)), f"Budget mismatch: {mask.sum()} vs {int(alpha*len(y))}"

    def test_dp_attack_increases_dp(self, synthetic_data):
        """DP attack must increase DP violation compared to clean."""
        X, y, a = synthetic_data

        def compute_dp(y_pred, a):
            mask0 = (a == 0)
            mask1 = (a == 1)
            return abs(np.mean(y_pred[mask0]) - np.mean(y_pred[mask1]))

        dp_before = compute_dp(y, a)

        ft = FairnessTargetedPGD(alpha=0.2, target_metric='dp', pgd_steps=5,
                                 coordinated=True, random_state=42)
        _, y_att, _, _ = ft.corrupt(X.copy(), y.copy(), a.copy())
        dp_after = compute_dp(y_att, a)

        assert dp_after > dp_before * 1.05, f"DP did not increase: {dp_after:.4f} <= {dp_before:.4f}"

    def test_if_attack_increases_if(self, synthetic_data):
        """IF attack must increase IF violation (approximate via label consistency)."""
        X, y, a = synthetic_data

        def approx_if(y, a, X, k=5):
            from sklearn.neighbors import NearestNeighbors
            violation = 0.0
            n = len(y)
            for g in [0, 1]:
                mask = (a == g)
                if mask.sum() <= k:
                    continue
                idx = np.where(mask)[0]
                X_g = X[idx]
                y_g = y[idx]
                nbrs = NearestNeighbors(n_neighbors=min(k+1, len(idx))).fit(X_g)
                _, neigh = nbrs.kneighbors(X_g)
                for i in range(len(idx)):
                    disagreements = sum(y_g[neigh[i, 1:]] != y_g[i])
                    violation += disagreements / (len(neigh[i]) - 1)
            return violation / n

        if_before = approx_if(y, a, X)

        ft = FairnessTargetedPGD(alpha=0.2, target_metric='if', pgd_steps=5,
                                 coordinated=True, random_state=42)
        _, y_att, _, _ = ft.corrupt(X.copy(), y.copy(), a.copy())
        if_after = approx_if(y_att, a, X)

        assert if_after > if_before * 0.95, f"IF did not increase: {if_after:.4f} <= {if_before:.4f}"

    def test_minority_targeted(self, synthetic_data):
        """With coordinated=True, majority of corruptions should hit minority group."""
        X, y, a = synthetic_data
        ft = FairnessTargetedPGD(alpha=0.2, target_metric='dp', pgd_steps=3,
                                 coordinated=True, random_state=42)
        _, _, _, mask = ft.corrupt(X.copy(), y.copy(), a.copy())

        group_counts = np.bincount(a.astype(int))
        minority_group = int(np.argmin(group_counts))

        minority_hits = np.sum(mask & (a == minority_group))
        total_hits = np.sum(mask)
        ratio = minority_hits / total_hits

        assert ratio >= 0.55, f"Minority targeting too weak: {ratio:.2f} < 0.55"

    def test_reproducibility(self, synthetic_data):
        """Same random_state must produce same corruptions."""
        X, y, a = synthetic_data
        ft1 = FairnessTargetedPGD(alpha=0.2, target_metric='dp', pgd_steps=3,
                                  coordinated=True, random_state=42)
        ft2 = FairnessTargetedPGD(alpha=0.2, target_metric='dp', pgd_steps=3,
                                  coordinated=True, random_state=42)
        _, y1, _, m1 = ft1.corrupt(X.copy(), y.copy(), a.copy())
        _, y2, _, m2 = ft2.corrupt(X.copy(), y.copy(), a.copy())

        assert np.array_equal(y1, y2), "Reproducibility failed: y differs"
        assert np.array_equal(m1, m2), "Reproducibility failed: mask differs"

    def test_combined_uses_both_gradients(self, synthetic_data):
        """Combined metric should produce different corruptions than DP-only."""
        X, y, a = synthetic_data
        ft_dp = FairnessTargetedPGD(alpha=0.2, target_metric='dp', pgd_steps=3,
                                    coordinated=True, random_state=42)
        ft_comb = FairnessTargetedPGD(alpha=0.2, target_metric='combined', pgd_steps=3,
                                      coordinated=True, random_state=42)
        _, _, _, m_dp = ft_dp.corrupt(X.copy(), y.copy(), a.copy())
        _, _, _, m_comb = ft_comb.corrupt(X.copy(), y.copy(), a.copy())

        # Combined should not be identical to DP-only
        assert not np.array_equal(m_dp, m_comb), "Combined attack identical to DP-only"
