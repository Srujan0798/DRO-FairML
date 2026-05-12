"""
Targeted verification for CYCLE 3 fixes.
Tests the critical cells: Adult α=0.2 (DP regression) and Credit α=0.4 (τ=1 accuracy).
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
from src.training.standard_ml import StandardMLTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


def train_warm_start(X_train, y_train, a_train, input_dim, device='cpu', epochs=10):
    model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer = StandardMLTrainer(model, device=device, epochs=epochs, lr=1e-3)
    trainer.fit(X_train, y_train, verbose=False)
    return model


def run_targeted(dataset, alpha, seed=42):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}, Alpha: {alpha}, Seed: {seed}")
    print(f"{'='*60}")

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset(dataset, random_state=seed)

    tau_train = get_temperature(alpha)
    print(f"Using τ={tau_train} (alpha={alpha})")

    input_dim = X_train.shape[1]

    warm_model = train_warm_start(X_train, y_train, a_train, input_dim, device='cpu', epochs=10)

    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1, pgd_steps=5, pgd_step_size=0.02,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed
    )
    X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train, model=warm_model, device='cpu')

    # Pretrain
    model_pretrained = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    pretrainer = StandardMLTrainer(model_pretrained, device='cpu', epochs=15, lr=1e-3)
    pretrainer.fit(X_tr, y_tr, verbose=False)

    # Naive-FAIR
    model_naive = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    model_naive.load_state_dict(model_pretrained.state_dict())
    trainer_naive = NaiveFairTrainer(
        model_naive, device='cpu',
        lr_theta=1e-3, lr_lambda=5e-3, lambda_max=10.0,
        tau=tau_train, k=5, gamma=0.0,
        epochs=15, weight_decay=1e-4, tau_warmup_epochs=5
    )
    trainer_naive.fit(X_tr, y_tr, a_tr, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
    preds_naive = trainer_naive.predict(X_test)

    # DRO-FAIR
    model_dro = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    model_dro.load_state_dict(model_pretrained.state_dict())
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device='cpu',
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=10.0,
        tau=tau_train, beta=5.0, k=5, gamma=0.0,
        K_inner=10, epochs=15, weight_decay=1e-4, tau_warmup_epochs=5
    )
    trainer_dro.fit(X_tr, y_tr, a_tr, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
    preds_dro = trainer_dro.predict(X_test)

    naive_acc = compute_accuracy(y_test, preds_naive)
    naive_dp = compute_dp_violation(preds_naive, a_test)
    dro_acc = compute_accuracy(y_test, preds_dro)
    dro_dp = compute_dp_violation(preds_dro, a_test)

    print(f"  NAIVE: Acc={naive_acc:.4f}, DP={naive_dp:.4f}")
    print(f"  DRO:   Acc={dro_acc:.4f}, DP={dro_dp:.4f}")
    print(f"  DP WIN: {1 if dro_dp < naive_dp else 0}")

    return {
        'naive_acc': naive_acc, 'naive_dp': naive_dp,
        'dro_acc': dro_acc, 'dro_dp': dro_dp
    }


if __name__ == '__main__':
    results = {}

    # Critical 1: Adult α=0.2 (DP regression test)
    results['adult_0.2'] = run_targeted('adult', 0.2, seed=42)
    results['adult_0.2_s2'] = run_targeted('adult', 0.2, seed=43)
    results['adult_0.2_s3'] = run_targeted('adult', 0.2, seed=44)

    # Critical 2: Credit α=0.4 (τ=1 accuracy test)
    results['credit_0.4'] = run_targeted('credit', 0.4, seed=42)

    # Adult α=0.1 and 0.3 for additional context
    results['adult_0.1'] = run_targeted('adult', 0.1, seed=42)
    results['adult_0.3'] = run_targeted('adult', 0.3, seed=42)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Adult α=0.2 DP
    adult_dp_wins = sum(1 for k in ['adult_0.2', 'adult_0.2_s2', 'adult_0.2_s3']
                        if results[k]['dro_dp'] < results[k]['naive_dp'])
    print(f"Adult α=0.2 DP wins: {adult_dp_wins}/3")

    # Credit α=0.4 accuracy
    credit_acc = results['credit_0.4']['dro_acc']
    print(f"Credit α=0.4 DRO accuracy: {credit_acc:.4f} (need ≥0.60, target ≥0.70)")

    # Overall DP wins on tested cells
    for name, res in results.items():
        dp_win = res['dro_dp'] < res['naive_dp']
        print(f"  {name}: N={res['naive_dp']:.4f}, D={res['dro_dp']:.4f}, win={dp_win}")