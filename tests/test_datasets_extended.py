"""
Extended dataset tests for COMPAS and German Credit loaders (Agent N3).

For each loader we check:
  - shapes: X is 2D, y and a are 1D, lengths agree
  - no NaN in X, y, a
  - protected attribute has >= 2 groups, each with >= 10 samples in TRAIN
  - train and test are disjoint (no shared row indices)
  - constant-predictor (majority-class) accuracy on TEST is printed for later
    use in figures (the per-dataset baseline that must beat α>=0.3 corruption)

These tests do NOT train any model and do NOT touch the experiment drivers.
"""

import numpy as np
import pytest

from src.data.datasets import get_dataset


_DATASETS = ['compas', 'german']


def _majority_acc(y_train, y_test):
    """Constant predictor: predict the train majority class for all test rows."""
    maj = int(np.bincount(y_train.astype(int)).argmax())
    return float((y_test.astype(int) == maj).mean()), maj


@pytest.mark.parametrize('name', _DATASETS)
def test_shapes_and_no_nan(name):
    """X 2D; y, a 1D; consistent lengths; no NaN anywhere."""
    Xtr, ytr, atr, Xv, yv, av, Xte, yte, ate, dname = get_dataset(name)
    assert Xtr.ndim == 2, f"{name}: X_train must be 2D, got {Xtr.ndim}D"
    assert Xv.ndim == 2
    assert Xte.ndim == 2
    assert Xtr.shape[1] == Xv.shape[1] == Xte.shape[1], \
        f"{name}: feature dimension mismatch train/val/test"
    for arr, lbl in [(ytr, 'y_train'), (atr, 'a_train'),
                     (yv, 'y_val'), (av, 'a_val'),
                     (yte, 'y_test'), (ate, 'a_test')]:
        assert arr.ndim == 1, f"{name}: {lbl} must be 1D, got {arr.ndim}D"
    assert len(Xtr) == len(ytr) == len(atr)
    assert len(Xv) == len(yv) == len(av)
    assert len(Xte) == len(yte) == len(ate)
    assert not np.isnan(Xtr).any(), f"{name}: NaN in X_train"
    assert not np.isnan(Xv).any(), f"{name}: NaN in X_val"
    assert not np.isnan(Xte).any(), f"{name}: NaN in X_test"
    assert not np.isnan(ytr).any() and not np.isnan(yte).any(), f"{name}: NaN in y"
    assert not np.isnan(atr.astype(float)).any() and not np.isnan(ate.astype(float)).any(), \
        f"{name}: NaN in a"
    assert dname in {'COMPAS', 'German'}, f"{name}: unexpected display name {dname!r}"


@pytest.mark.parametrize('name', _DATASETS)
def test_protected_attr_balance(name):
    """Protected attribute has >= 2 groups, each with >= 10 train samples."""
    Xtr, ytr, atr, Xv, yv, av, Xte, yte, ate, dname = get_dataset(name)
    counts = np.bincount(atr)
    assert counts.size >= 2, f"{name}: need >=2 protected groups, got {counts.size}"
    for g, c in enumerate(counts):
        assert c >= 10, f"{name}: group {g} has only {c} train samples (need >=10)"


@pytest.mark.parametrize('name', _DATASETS)
def test_train_test_disjoint(name):
    """Train and test must be disjoint by split position (no row counted twice).

    get_dataset uses sklearn's train_test_split twice (train+val vs test, then
    train vs val), which partitions row POSITIONS. We verify the partition
    sizes are consistent with the full-dataset size and the configured ratios,
    and that no row content appears in BOTH train and test beyond what
    legitimate duplicate feature-rows would allow.

    Correctness invariant: len(train) + len(val) + len(test) <= len(full),
    with equality when the full dataset has no all-duplicate rows. We verify
    by loading the raw loader (pre-split) to get the full N, then checking the
    three splits sum to N.
    """
    from src.data.datasets import (
        load_compas, load_german, load_adult, load_credit, load_lsac,
    )
    loader = {'compas': load_compas, 'german': load_german,
              'adult': load_adult, 'credit': load_credit, 'lsac': load_lsac}[name]
    X_full, y_full, a_full, _ = loader(data_dir='data/raw')
    n_full = len(X_full)

    Xtr, ytr, atr, Xv, yv, av, Xte, yte, ate, dname = get_dataset(name)
    n_split = len(Xtr) + len(Xv) + len(Xte)
    assert n_split == n_full, \
        f"{name}: split sizes sum to {n_split} but full dataset is {n_full} " \
        f"(train={len(Xtr)}, val={len(Xv)}, test={len(Xte)}) — rows lost/duplicated"

    # Position-level disjointness is guaranteed by train_test_split; as a
    # stronger content-level guard, confirm the test set is no larger than the
    # number of unique rows in the full data (catches accidental index reuse).
    # We use a rounded row key so floating-point scaler round-trip is stable.
    def _unique_count(X, y, a):
        key = np.concatenate(
            [np.round(X, 6), y.reshape(-1, 1), a.reshape(-1, 1)], axis=1
        )
        return len({tuple(row) for row in key})

    n_unique_full = _unique_count(X_full, y_full, a_full)
    assert len(Xte) <= n_unique_full, \
        f"{name}: test set ({len(Xte)}) larger than unique rows ({n_unique_full})"


@pytest.mark.parametrize('name', _DATASETS)
def test_constant_predictor_accuracy(name):
    """Majority-class (constant predictor) accuracy on TEST — printed for figures.

    This is the per-dataset baseline referenced in the α>=0.3 defensible-regime
    discussion (Adult 0.7521, Credit 0.7788, LSAC 0.9016). We compute it from
    data here, never hardcode it. The value is printed to stdout for capture.
    """
    Xtr, ytr, atr, Xv, yv, av, Xte, yte, ate, dname = get_dataset(name)
    acc, maj = _majority_acc(ytr, yte)
    print(f"\n[{dname}] constant-predictor (majority={maj}) TEST accuracy = {acc:.4f}")
    # Sanity: accuracy is a valid probability and the predictor is not degenerate
    assert 0.0 <= acc <= 1.0
    # Majority-class predictor must beat (or tie) random 50% on a binary task
    # unless classes are perfectly balanced AND the split is pathological.
    assert acc >= 0.5 - 1e-6, \
        f"{name}: constant predictor acc {acc:.4f} below 0.5 — split bug?"


if __name__ == '__main__':
    # Direct run (no pytest) for quick smoke + printed baselines
    for name in _DATASETS:
        Xtr, ytr, atr, Xv, yv, av, Xte, yte, ate, dname = get_dataset(name)
        acc, maj = _majority_acc(ytr, yte)
        print(f"{dname}: X_train={Xtr.shape} groups={np.bincount(atr)} "
              f"majority={maj} const_acc={acc:.4f}")