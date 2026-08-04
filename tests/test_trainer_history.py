"""
Unit tests for per-epoch history logging in DroFairTrainer and NaiveFairTrainer.

Agent N2 (convergence diagnostics): both trainers must expose a `history` dict
after fit() with per-epoch train_loss and val_loss (and val_acc/val_dp/val_if)
so the high-alpha convergence plots can be generated. DroFairTrainer already
had this; NaiveFairTrainer was extended to match.

These tests train on a tiny Adult slice (100 samples, 3 epochs) so they run
fast and do not touch results/*.json.
"""

import math
import os
import json
import tempfile

import numpy as np
import torch

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.training.dro_fair import DroFairTrainer
from src.training.naive_fair import NaiveFairTrainer


def _load_tiny_adult(n_train=100, n_val=60, seed=42):
    """Load a tiny slice of Adult for fast history tests.

    Slices the FIRST n_train/n_val rows of the already-split train/val partitions
    returned by get_dataset. No resampling, no leakage — just a small subset.
    """
    X_train, y_train, a_train, X_val, y_val, a_val, _, _, _, _ = \
        get_dataset('adult', random_state=seed)
    return (
        X_train[:n_train], y_train[:n_train], a_train[:n_train],
        X_val[:n_val], y_val[:n_val], a_val[:n_val],
    )


def _train_dro(Xtr, ytr, atr, Xv, yv, av, epochs=3):
    model = MLPClassifier(Xtr.shape[1], hidden_dims=[16, 8], dropout=0.0)
    trainer = DroFairTrainer(
        model, alpha=0.2, device='cpu', epochs=epochs, K_inner=2,
        tau=1.0, k=3, use_dp=True, use_if=True,
    )
    trainer.fit(Xtr, ytr, atr, X_val=Xv, y_val=yv, a_val=av, verbose=False)
    return trainer


def _train_naive(Xtr, ytr, atr, Xv, yv, av, epochs=3):
    model = MLPClassifier(Xtr.shape[1], hidden_dims=[16, 8], dropout=0.0)
    trainer = NaiveFairTrainer(
        model, device='cpu', epochs=epochs, tau=1.0, k=3,
    )
    trainer.fit(Xtr, ytr, atr, X_val=Xv, y_val=yv, a_val=av, verbose=False)
    return trainer


def test_dro_trainer_has_history_dict():
    """DroFairTrainer.fit() must expose self.history as a dict."""
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    trainer = _train_dro(Xtr, ytr, atr, Xv, yv, av, epochs=3)
    assert hasattr(trainer, 'history'), "DroFairTrainer missing self.history"
    assert isinstance(trainer.history, dict), \
        f"history must be a dict, got {type(trainer.history)}"


def test_naive_trainer_has_history_dict():
    """NaiveFairTrainer.fit() must expose self.history as a dict (N2 addition)."""
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    trainer = _train_naive(Xtr, ytr, atr, Xv, yv, av, epochs=3)
    assert hasattr(trainer, 'history'), "NaiveFairTrainer missing self.history"
    assert isinstance(trainer.history, dict), \
        f"history must be a dict, got {type(trainer.history)}"


def _assert_history_schema(history, label):
    """Assert history has the keys N2 convergence plots need."""
    required = ['train_loss', 'val_acc', 'val_loss', 'val_dp', 'val_if']
    for k in required:
        assert k in history, f"{label}: history missing key {k!r}; has {list(history.keys())}"


def test_dro_history_schema():
    """DRO history must contain train_loss, val_acc, val_loss, val_dp, val_if."""
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    trainer = _train_dro(Xtr, ytr, atr, Xv, yv, av, epochs=3)
    _assert_history_schema(trainer.history, 'DRO')


def test_naive_history_schema():
    """Naive history must contain train_loss, val_acc, val_loss, val_dp, val_if."""
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    trainer = _train_naive(Xtr, ytr, atr, Xv, yv, av, epochs=3)
    _assert_history_schema(trainer.history, 'Naive')


def test_dro_history_train_loss_length_matches_epochs():
    """DRO history['train_loss'] must have one entry per epoch."""
    epochs = 3
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    trainer = _train_dro(Xtr, ytr, atr, Xv, yv, av, epochs=epochs)
    assert len(trainer.history['train_loss']) == epochs, \
        f"DRO train_loss len {len(trainer.history['train_loss'])} != {epochs}"
    # val_loss is also per-epoch in DRO (logged every epoch)
    assert len(trainer.history['val_loss']) == epochs, \
        f"DRO val_loss len {len(trainer.history['val_loss'])} != {epochs}"


def test_naive_history_train_loss_length_matches_epochs():
    """Naive history['train_loss'] must have one entry per epoch."""
    epochs = 3
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    trainer = _train_naive(Xtr, ytr, atr, Xv, yv, av, epochs=epochs)
    assert len(trainer.history['train_loss']) == epochs, \
        f"Naive train_loss len {len(trainer.history['train_loss'])} != {epochs}"
    # val_loss is per-epoch (N2 addition)
    assert len(trainer.history['val_loss']) == epochs, \
        f"Naive val_loss len {len(trainer.history['val_loss'])} != {epochs}"


def _assert_val_loss_finite_or_nonincreasing(history, label):
    """val_loss should be finite; ideally non-increasing on average.

    On a tiny 100-sample slice with only 3 epochs the loss curve can be noisy,
    so the hard requirement is: the last val_loss is finite. The softer check
    (non-increasing on average) is reported but not asserted strictly.
    """
    vl = history['val_loss']
    assert len(vl) > 0, f"{label}: val_loss is empty"
    last = vl[-1]
    assert math.isfinite(last), f"{label}: last val_loss is not finite: {last}"
    # Soft check: report mean trend but only assert finiteness (tiny data is noisy).
    if len(vl) >= 2:
        first_half = float(np.mean(vl[:len(vl) // 2 + 1])) if len(vl) >= 2 else float(vl[0])
        second_half = float(np.mean(vl[len(vl) // 2:])) if len(vl) >= 2 else float(vl[-1])
        # Non-increasing on average is the ideal; we don't hard-assert on 3-epoch tiny data.
        # Just make sure nothing is NaN/inf anywhere.
        assert all(math.isfinite(v) for v in vl), \
            f"{label}: val_loss contains non-finite values: {vl}"


def test_dro_val_loss_finite():
    """DRO val_loss last value must be finite (and all entries finite)."""
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    trainer = _train_dro(Xtr, ytr, atr, Xv, yv, av, epochs=3)
    _assert_val_loss_finite_or_nonincreasing(trainer.history, 'DRO')


def test_naive_val_loss_finite():
    """Naive val_loss last value must be finite (and all entries finite)."""
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    trainer = _train_naive(Xtr, ytr, atr, Xv, yv, av, epochs=3)
    _assert_val_loss_finite_or_nonincreasing(trainer.history, 'Naive')


def test_dro_history_path_dump():
    """DroFairTrainer.history_path should dump history JSON at end of fit()."""
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'dro_history.json')
        model = MLPClassifier(Xtr.shape[1], hidden_dims=[16, 8], dropout=0.0)
        trainer = DroFairTrainer(
            model, alpha=0.2, device='cpu', epochs=3, K_inner=2,
            tau=1.0, k=3, history_path=path,
        )
        trainer.fit(Xtr, ytr, atr, X_val=Xv, y_val=yv, a_val=av, verbose=False)
        assert os.path.exists(path), f"history_path file not created: {path}"
        with open(path) as f:
            dumped = json.load(f)
        assert 'train_loss' in dumped and len(dumped['train_loss']) == 3
        assert 'val_loss' in dumped and len(dumped['val_loss']) == 3


def test_naive_history_path_dump():
    """NaiveFairTrainer.history_path should dump history JSON at end of fit() (N2 addition)."""
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'naive_history.json')
        model = MLPClassifier(Xtr.shape[1], hidden_dims=[16, 8], dropout=0.0)
        trainer = NaiveFairTrainer(
            model, device='cpu', epochs=3, tau=1.0, k=3, history_path=path,
        )
        trainer.fit(Xtr, ytr, atr, X_val=Xv, y_val=yv, a_val=av, verbose=False)
        assert os.path.exists(path), f"history_path file not created: {path}"
        with open(path) as f:
            dumped = json.load(f)
        assert 'train_loss' in dumped and len(dumped['train_loss']) == 3
        assert 'val_loss' in dumped and len(dumped['val_loss']) == 3


def test_run_single_experiment_dump_history_off_by_default():
    """dump_history defaults to False — no history file written, result unchanged."""
    from experiments.run_fairness_pgd import run_single_experiment
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            result = run_single_experiment(
                'adult', alpha=0.2, seed=42, attack='dp', method='dro',
                device='cpu', epochs=2, k_inner=2, pgd_steps=2, tau=1.0,
            )
            # No history file should exist in results/
            files = os.listdir('results') if os.path.isdir('results') else []
            history_files = [f for f in files if f.startswith('history_')]
            assert history_files == [], \
                f"dump_history=False but history files written: {history_files}"
            assert 'history_path' not in result, \
                f"result should not contain history_path when dump_history=False: {result.keys()}"
        finally:
            os.chdir(old_cwd)


def test_run_single_experiment_dump_history_writes_dro_only():
    """dump_history=True writes a history JSON for method='dro' only."""
    from experiments.run_fairness_pgd import run_single_experiment
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            result = run_single_experiment(
                'adult', alpha=0.2, seed=42, attack='dp', method='dro',
                device='cpu', epochs=2, k_inner=2, pgd_steps=2, tau=1.0,
                dump_history=True,
            )
            files = os.listdir('results') if os.path.isdir('results') else []
            history_files = [f for f in files if f.startswith('history_')]
            assert len(history_files) == 1, \
                f"expected 1 history file, got {history_files}"
            assert 'adult_dp_0.2_42_dro' in history_files[0], \
                f"history filename wrong: {history_files[0]}"
            with open(os.path.join('results', history_files[0])) as f:
                dumped = json.load(f)
            assert 'train_loss' in dumped and len(dumped['train_loss']) == 2
            assert 'val_loss' in dumped and len(dumped['val_loss']) == 2
            assert 'history_path' in result
        finally:
            os.chdir(old_cwd)


def test_run_single_experiment_dump_history_skips_naive():
    """dump_history=True does NOT write a history file for method='naive'."""
    from experiments.run_fairness_pgd import run_single_experiment
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            result = run_single_experiment(
                'adult', alpha=0.2, seed=42, attack='dp', method='naive',
                device='cpu', epochs=2, k_inner=2, pgd_steps=2, tau=1.0,
                dump_history=True,
            )
            files = os.listdir('results') if os.path.isdir('results') else []
            history_files = [f for f in files if f.startswith('history_')]
            assert history_files == [], \
                f"naive should not dump history, but got: {history_files}"
            assert 'history_path' not in result
        finally:
            os.chdir(old_cwd)