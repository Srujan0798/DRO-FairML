"""
Quick comparison: τ=1 constant vs τ warmup (1→100 after 5 epochs)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


def run_test(dataset='adult', alpha=0.2, seed=42):
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset(dataset, random_state=seed)

    corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
    X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train)

    tau = get_temperature(alpha)
    input_dim = X_train.shape[1]

    results = {}

    # τ=1 constant (old bad way)
    for method_name, tau_train in [('τ=1 constant', 1.0), ('τ warmup 1→100', 100.0)]:
        warmup = 5 if tau_train == 100.0 else 0

        model = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
        if method_name == 'τ=1 constant':
            trainer = NaiveFairTrainer(model, device='cpu', tau=1.0, epochs=30, k=5)
        else:
            trainer = NaiveFairTrainer(model, device='cpu', tau=100.0, epochs=30, k=5, tau_warmup_epochs=5)

        trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
        preds = trainer.predict(X_test)

        results[method_name] = {
            'dp': compute_dp_violation(preds, a_test),
            'acc': compute_accuracy(y_test, preds)
        }

    return results


if __name__ == '__main__':
    print("Testing τ=1 constant vs τ warmup on Adult α=0.2")
    print()

    for seed in [42, 43]:
        print(f"Seed {seed}:")
        r = run_test('adult', 0.2, seed)
        for method, vals in r.items():
            print(f"  {method}: DP={vals['dp']:.4f}, Acc={vals['acc']:.4f}")
        print()