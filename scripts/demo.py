"""
Quick demo script showing DRO-FAIR vs Naive-FAIR on a single dataset.
Run this for a fast demonstration of the method.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation


def run_demo(dataset='adult', alpha=0.2, seed=42, epochs=20):
    """Run a quick demo comparing Naive-FAIR and DRO-FAIR."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Dataset: {dataset}, Alpha: {alpha}, Seed: {seed}\n")
    
    # Load data
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset, random_state=seed)
    print(f"Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test, {X_train.shape[1]} features")
    
    # Apply adversarial corruption
    print("Applying adversarial corruption...")
    corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
    X_train_c, y_train_c, a_train_c, _ = corruptor.corrupt(X_train, y_train, a_train)
    X_val_c, y_val_c, a_val_c, _ = corruptor.corrupt(X_val, y_val, a_val)
    
    input_dim = X_train.shape[1]
    
    # Naive-FAIR
    print(f"\n{'='*60}")
    print("Training Naive-FAIR...")
    model_naive = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer_naive = NaiveFairTrainer(model_naive, device=device, epochs=epochs,
                                     tau=100.0, beta=5.0, k=5, gamma=0.0)
    trainer_naive.fit(X_train_c, y_train_c, a_train_c, X_val_c, y_val_c, a_val_c, verbose=True)
    
    preds_naive = trainer_naive.predict(X_test)
    acc_naive = compute_accuracy(y_test, preds_naive)
    dp_naive = compute_dp_violation(preds_naive, a_test)
    if_naive = compute_if_violation(X_test, preds_naive, a_test, k=5)
    
    print(f"\nNaive-FAIR Results:")
    print(f"  Accuracy: {acc_naive:.4f}")
    print(f"  DP Violation: {dp_naive:.4f}")
    print(f"  IF Violation: {if_naive:.4f}")
    
    # DRO-FAIR
    print(f"\n{'='*60}")
    print("Training DRO-FAIR...")
    model_dro = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer_dro = DroFairTrainer(model_dro, alpha=alpha, device=device, epochs=epochs,
                                 tau=100.0, beta=5.0, k=5, gamma=0.0, K_inner=10)
    trainer_dro.fit(X_train_c, y_train_c, a_train_c, X_val_c, y_val_c, a_val_c, verbose=True)
    
    preds_dro = trainer_dro.predict(X_test)
    acc_dro = compute_accuracy(y_test, preds_dro)
    dp_dro = compute_dp_violation(preds_dro, a_test)
    if_dro = compute_if_violation(X_test, preds_dro, a_test, k=5)
    
    print(f"\nDRO-FAIR Results:")
    print(f"  Accuracy: {acc_dro:.4f}")
    print(f"  DP Violation: {dp_dro:.4f}")
    print(f"  IF Violation: {if_dro:.4f}")
    
    # Comparison
    print(f"\n{'='*60}")
    print("Comparison (DRO-FAIR vs Naive-FAIR):")
    print(f"  Accuracy change: {(acc_dro - acc_naive)*100:+.2f}%")
    print(f"  DP reduction: {(dp_naive - dp_dro) / (dp_naive + 1e-8) * 100:.1f}%")
    print(f"  IF reduction: {(if_naive - if_dro) / (if_naive + 1e-8) * 100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='adult', choices=['adult', 'credit', 'lsac'])
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    run_demo(args.dataset, args.alpha, args.seed, args.epochs)
