#!/usr/bin/env python3
"""Test two-stage training: pretrain without fairness, then add constraints."""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.training.dro_fair import DroFairTrainer
from src.training.naive_fair import NaiveFairTrainer
from src.training.standard_ml import StandardMLTrainer
from src.corruption.adversarial import AdversarialCorruptor

def test_two_stage(name, method, alpha=0.1, epochs_pre=10, epochs_fair=20):
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = get_dataset(
        name, data_dir='data/raw', random_state=0
    )
    
    warm = MLPClassifier(X_train.shape[1], hidden_dims=[128, 64])
    StandardMLTrainer(warm, device='cpu', epochs=10).fit(X_train, y_train, verbose=False)
    corruptor = AdversarialCorruptor(alpha=alpha, random_state=0)
    X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train, model=warm, device='cpu')
    
    device = 'cpu'
    model = MLPClassifier(input_dim=X_train.shape[1], hidden_dims=[128, 64]).to(device)
    
    # Stage 1: Pretrain without fairness
    pretrainer = StandardMLTrainer(model, device=device, epochs=epochs_pre, lr=1e-3)
    pretrainer.fit(X_tr, y_tr, verbose=False)
    
    # Stage 2: Fine-tune with fairness
    if method == 'dro':
        trainer = DroFairTrainer(
            model, alpha=alpha, device=device,
            lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=10.0,
            tau=1.0, beta=5.0, k=5, gamma=0.0,
            K_inner=10, epochs=epochs_fair, weight_decay=1e-4,
        )
    else:
        trainer = NaiveFairTrainer(
            model, device=device,
            lr_theta=1e-3, lr_lambda=5e-3, lambda_max=10.0,
            tau=1.0, k=5, gamma=0.0, epochs=epochs_fair, weight_decay=1e-4,
        )
    
    history = trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
    
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits = model(X_t).squeeze().cpu().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(float)
        acc = (preds == y_test).mean()
        
        return {
            'acc': acc,
            'unique': len(np.unique(preds)),
            'logits_range': (float(logits.min()), float(logits.max())),
        }

configs = [
    ('adult', 'dro', 0.1),
    ('lsac', 'naive', 0.0),
    ('lsac', 'dro', 0.1),
    ('credit', 'dro', 0.1),
]

for name, method, alpha in configs:
    print(f"\n{name.upper()} {method.upper()} alpha={alpha}:")
    # No pretraining
    result0 = test_two_stage(name, method, alpha=alpha, epochs_pre=0, epochs_fair=30)
    status0 = "✓" if result0['unique'] > 1 else "✗ DEGENERATE"
    print(f"  no pretrain: Acc={result0['acc']:.4f}, Unique={result0['unique']}, "
          f"Logits=[{result0['logits_range'][0]:.2f}, {result0['logits_range'][1]:.2f}] {status0}")
    
    # With pretraining
    result1 = test_two_stage(name, method, alpha=alpha, epochs_pre=10, epochs_fair=20)
    status1 = "✓" if result1['unique'] > 1 else "✗ DEGENERATE"
    print(f"  pretrain(10): Acc={result1['acc']:.4f}, Unique={result1['unique']}, "
          f"Logits=[{result1['logits_range'][0]:.2f}, {result1['logits_range'][1]:.2f}] {status1}")
    
    # More pretraining
    result2 = test_two_stage(name, method, alpha=alpha, epochs_pre=20, epochs_fair=10)
    status2 = "✓" if result2['unique'] > 1 else "✗ DEGENERATE"
    print(f"  pretrain(20): Acc={result2['acc']:.4f}, Unique={result2['unique']}, "
          f"Logits=[{result2['logits_range'][0]:.2f}, {result2['logits_range'][1]:.2f}] {status2}")
