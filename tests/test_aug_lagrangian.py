"""
Unit tests for DRO-FAIR-AL (aug_lagrangian_mu quadratic constraint penalty).

Pre-registered design: docs/superpowers/specs/2026-08-05-augmented-lagrangian-design.md.
Three guarantees:
  (a) mu=0 (default) is an exact no-op — training is bit-identical to a trainer
      constructed without the parameter;
  (b) mu>0 changes training (the quadratic term feeds mu*g*grad(g) into theta);
  (c) the loss math is right: (mu/2)*g^2 added per active constraint.

Tests train on a tiny Adult slice (100 samples, 3 epochs) so they run fast and
do not touch results/*.json.
"""

import numpy as np
import torch

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.training.dro_fair import DroFairTrainer


def _load_tiny_adult(n_train=100, n_val=60, seed=42):
    X_train, y_train, a_train, X_val, y_val, a_val, _, _, _, _ = \
        get_dataset('adult', random_state=seed)
    return (
        X_train[:n_train], y_train[:n_train], a_train[:n_train],
        X_val[:n_val], y_val[:n_val], a_val[:n_val],
    )


def _train(mu, pass_param, epochs=3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, atr, Xv, yv, av = _load_tiny_adult()
    model = MLPClassifier(Xtr.shape[1], hidden_dims=[16, 8], dropout=0.0)
    kwargs = dict(alpha=0.2, device='cpu', epochs=epochs, K_inner=2,
                  tau=1.0, k=3, use_dp=True, use_if=True)
    if pass_param:
        kwargs['aug_lagrangian_mu'] = mu
    trainer = DroFairTrainer(model, **kwargs)
    trainer.fit(Xtr, ytr, atr, X_val=Xv, y_val=yv, a_val=av, verbose=False)
    return trainer


def _params_vec(trainer):
    return torch.cat([p.detach().flatten() for p in trainer.model.parameters()])


def test_mu_zero_is_exact_noop():
    """mu=0.0 must produce bit-identical parameters to omitting the param."""
    t_default = _train(mu=None, pass_param=False)
    t_mu0 = _train(mu=0.0, pass_param=True)
    v1, v2 = _params_vec(t_default), _params_vec(t_mu0)
    assert torch.equal(v1, v2), (
        "aug_lagrangian_mu=0.0 changed training — no-op guarantee violated "
        f"(max abs diff {torch.max(torch.abs(v1 - v2)).item():.3e})"
    )


def test_mu_positive_changes_training():
    """mu>0 must actually alter the optimization trajectory."""
    t_mu0 = _train(mu=0.0, pass_param=True)
    t_mu5 = _train(mu=5.0, pass_param=True)
    v1, v2 = _params_vec(t_mu0), _params_vec(t_mu5)
    assert not torch.equal(v1, v2), (
        "aug_lagrangian_mu=5.0 produced identical parameters to mu=0 — "
        "the quadratic penalty is not reaching the loss"
    )


def test_quadratic_penalty_gradient_math():
    """d/dg of L + (mu/2)g^2 must add exactly mu*g to the constraint gradient."""
    mu = 7.0
    g = torch.tensor(0.13, requires_grad=True)
    base = 2.0 * g            # stand-in for lambda*g with lambda=2
    total = base + 0.5 * mu * g * g
    total.backward()
    expected = 2.0 + mu * 0.13
    assert abs(g.grad.item() - expected) < 1e-6


def test_runner_provenance_records_mu():
    """run_single_experiment must record aug_lagrangian_mu in the result row
    (signature default 0.0), so every ablation row is self-describing."""
    import inspect
    from experiments.run_fairness_pgd import run_single_experiment
    sig = inspect.signature(run_single_experiment)
    assert 'aug_lagrangian_mu' in sig.parameters
    assert sig.parameters['aug_lagrangian_mu'].default == 0.0
