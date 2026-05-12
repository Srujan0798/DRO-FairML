"""Tests for corruption modules."""

import numpy as np
import pytest
from src.corruption.adversarial import AdversarialCorruptor, RandomCorruptor


def test_corrupt_zero_alpha():
    """When alpha=0, no corruption should occur."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    a = np.random.randint(0, 2, 100)
    
    corruptor = AdversarialCorruptor(alpha=0.0, random_state=42)
    X_c, y_c, a_c, mask = corruptor.corrupt(X, y, a)
    
    assert np.allclose(X, X_c)
    assert np.allclose(y, y_c)
    assert np.allclose(a, a_c)
    assert mask.sum() == 0


def test_corrupt_nonzero_alpha():
    """When alpha>0, some samples should be corrupted."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    a = np.random.randint(0, 2, 100)
    
    corruptor = AdversarialCorruptor(alpha=0.2, random_state=42)
    X_c, y_c, a_c, mask = corruptor.corrupt(X, y, a)
    
    assert mask.sum() == 20
    # At least some features should change
    assert not np.allclose(X, X_c)


def test_corrupt_coordinated_targets_minority():
    """Coordinated corruption should target minority group more."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    a = np.zeros(100, dtype=int)
    a[70:] = 1  # 30% minority
    
    corruptor = AdversarialCorruptor(alpha=0.2, coordinated=True, random_state=42)
    X_c, y_c, a_c, mask = corruptor.corrupt(X, y, a)
    
    minority_corrupted = mask[a == 1].sum()
    majority_corrupted = mask[a == 0].sum()
    
    # Minority should have higher corruption rate
    minority_rate = minority_corrupted / 30
    majority_rate = majority_corrupted / 70
    assert minority_rate > majority_rate


def test_random_corruptor_basic():
    """RandomCorruptor should apply uniform random corruption."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    a = np.random.randint(0, 2, 100)
    
    corruptor = RandomCorruptor(alpha=0.2, random_state=42)
    X_c, y_c, a_c, mask = corruptor.corrupt(X, y, a)
    
    assert mask.sum() == 20
    assert not np.allclose(X, X_c)


def test_random_vs_adversarial_difference():
    """Adversarial and random corruption should produce different results."""
    np.random.seed(123)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    a = np.random.randint(0, 2, 100)
    
    adv = AdversarialCorruptor(alpha=0.2, coordinated=True, random_state=42)
    rand = RandomCorruptor(alpha=0.2, random_state=42)
    
    X_adv, y_adv, a_adv, _ = adv.corrupt(X.copy(), y.copy(), a.copy())
    X_rand, y_rand, a_rand, _ = rand.corrupt(X.copy(), y.copy(), a.copy())
    
    # They should be different (different selection + different perturbations)
    assert not np.allclose(X_adv, X_rand) or not np.allclose(y_adv, y_rand)
