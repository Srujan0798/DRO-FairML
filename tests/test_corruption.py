"""Unit tests for adversarial corruption."""

import numpy as np
from src.corruption.adversarial import AdversarialCorruptor


def test_corrupt_zero_alpha():
    """Test that alpha=0 does not corrupt any samples."""
    n = 100
    X = np.random.randn(n, 5)
    y = np.random.randint(0, 2, n).astype(np.float32)
    a = np.random.randint(0, 2, n)
    
    corruptor = AdversarialCorruptor(alpha=0.0, random_state=42)
    X_c, y_c, a_c, mask = corruptor.corrupt(X, y, a)
    
    assert not np.any(mask)
    assert np.allclose(X, X_c)
    assert np.allclose(y, y_c)
    assert np.allclose(a, a_c)


def test_corrupt_nonzero_alpha():
    """Test that alpha>0 corrupts approximately alpha fraction."""
    n = 1000
    X = np.random.randn(n, 5)
    y = np.random.randint(0, 2, n).astype(np.float32)
    a = np.random.randint(0, 2, n)
    
    alpha = 0.2
    corruptor = AdversarialCorruptor(alpha=alpha, random_state=42)
    X_c, y_c, a_c, mask = corruptor.corrupt(X, y, a)
    
    frac = mask.mean()
    assert 0.15 <= frac <= 0.25  # Approximate due to randomness
    assert not np.allclose(X, X_c) or not np.allclose(y, y_c) or not np.allclose(a, a_c)


def test_corrupt_coordinated_targets_minority():
    """Test that coordinated corruption targets minority group more."""
    n = 1000
    X = np.random.randn(n, 5)
    y = np.random.randint(0, 2, n).astype(np.float32)
    a = np.zeros(n, dtype=np.int64)
    a[800:] = 1  # Group 1 is minority (20%)
    
    alpha = 0.2
    corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=42)
    X_c, y_c, a_c, mask = corruptor.corrupt(X, y, a)
    
    minority_corrupt_rate = mask[a == 1].mean()
    majority_corrupt_rate = mask[a == 0].mean()
    assert minority_corrupt_rate > majority_corrupt_rate
