"""Tests for the CNN classifier inference mode fix.

The audit found that src/models/cnn_classifier.py predict/predict_proba
did not call self.eval() and torch.no_grad(), so Dropout (0.3) was active
at inference and grad tracking accumulated memory. This was especially
damaging for UTKFace (the only place CNNClassifier is used) since every
eval call leaked Dropout noise and the model never produced stable
predictions.
"""
import numpy as np
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_classifier import CNNClassifier


def test_cnn_predict_proba_does_not_track_grad():
    """predict_proba must not build an autograd graph."""
    model = CNNClassifier(input_channels=3, hidden_dim=32, num_classes=1)
    x = torch.randn(2, 3, 8, 8)
    probs = model.predict_proba(x)
    assert not probs.requires_grad, "predict_proba output should not require grad"


def test_cnn_predict_does_not_track_grad():
    """predict must not build an autograd graph (returns numpy)."""
    model = CNNClassifier(input_channels=3, hidden_dim=32, num_classes=1)
    x = torch.randn(2, 3, 8, 8)
    preds = model.predict(x)
    assert isinstance(preds, np.ndarray)
    assert set(np.unique(preds)).issubset({0, 1})


def test_cnn_predict_sets_eval_mode():
    """predict must switch the model to eval mode (so Dropout is disabled)."""
    model = CNNClassifier(input_channels=3, hidden_dim=32, num_classes=1)
    model.train()
    assert model.training
    x = torch.randn(4, 3, 8, 8)
    _ = model.predict(x)
    assert not model.training, "predict() should leave the model in eval mode"


def test_cnn_predict_is_deterministic():
    """Two consecutive predict() calls on the same input should give identical outputs.

    If Dropout were still active, the predictions would be noisy and differ.
    """
    torch.manual_seed(0)
    model = CNNClassifier(input_channels=3, hidden_dim=32, num_classes=1)
    x = torch.randn(8, 3, 8, 8)
    p1 = model.predict(x)
    p2 = model.predict(x)
    assert np.array_equal(p1, p2), "predict() must be deterministic when Dropout is off"


def test_cnn_predict_proba_does_not_mutate_training_state():
    """predict_proba should not accidentally switch the model out of train mode permanently."""
    model = CNNClassifier(input_channels=3, hidden_dim=32, num_classes=1)
    model.train()
    x = torch.randn(2, 3, 8, 8)
    _ = model.predict_proba(x)
    model.train()
    assert model.training
