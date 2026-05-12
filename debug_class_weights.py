#!/usr/bin/env python3
"""Quick test: do class weights fix degeneracy?"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import torch.nn.functional as F
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.training.dro_fair import DroFairTrainer
from src.training.naive_fair import NaiveFairTrainer
from src.training.standard_ml import StandardMLTrainer
from src.corruption.adversarial import AdversarialCorruptor

def test_config(name, method, alpha=0.1, use_weights=False, epochs=30):
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = get_dataset(
        name, data_dir='data/raw', random_state=0
    )
    
    # Corrupt data
    warm = MLPClassifier(X_train.shape[1], hidden_dims=[128, 64])
    StandardMLTrainer(warm, device='cpu', epochs=10).fit(X_train, y_train, verbose=False)
    corruptor = AdversarialCorruptor(alpha=alpha, random_state=0)
    X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train, model=warm, device='cpu')
    
    device = 'cpu'
    model = MLPClassifier(input_dim=X_train.shape[1], hidden_dims=[128, 64]).to(device)
    
    if method == 'dro':
        trainer = DroFairTrainer(
            model, alpha=alpha, device=device,
            lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=10.0,
            tau=1.0, beta=5.0, k=5, gamma=0.0,
            K_inner=10, epochs=epochs, weight_decay=1e-4,
        )
    else:
        trainer = NaiveFairTrainer(
            model, device=device,
            lr_theta=1e-3, lr_lambda=5e-3, lambda_max=10.0,
            tau=1.0, k=5, gamma=0.0, epochs=epochs, weight_decay=1e-4,
        )
    
    # Monkey-patch class weights if needed
    if use_weights:
        pos_weight = (1 - y_tr.mean()) / y_tr.mean()
        original_bce = F.binary_cross_entropy_with_logits
        def weighted_bce(logits, target, *args, **kwargs):
            pw = torch.tensor(pos_weight, device=target.device)
            if 'reduction' in kwargs and kwargs['reduction'] == 'none':
                return original_bce(logits, target, reduction='none', pos_weight=pw)
            return original_bce(logits, target, *args, pos_weight=pw, **kwargs)
        F.binary_cross_entropy_with_logits = weighted_bce
        history = trainer.fit(X_tr, y_tr, a_tr, X_val, y_val, a_val, verbose=False)
        F.binary_cross_entropy_with_logits = original_bce
    else:
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
            'loss_first': history['train_loss'][0],
            'loss_last': history['train_loss'][-1],
        }

configs = [
    ('adult', 'dro', 0.1),
    ('lsac', 'naive', 0.0),
    ('lsac', 'dro', 0.1),
    ('credit', 'dro', 0.1),
]

for name, method, alpha in configs:
    print(f"\n{name.upper()} {method.upper()} alpha={alpha}:")
    for use_weights in [False, True]:
        result = test_config(name, method, alpha=alpha, use_weights=use_weights)
        status = "✓" if result['unique'] > 1 else "✗ DEGENERATE"
        print(f"  weights={use_weights}: Acc={result['acc']:.4f}, Unique={result['unique']}, "
              f"Logits=[{result['logits_range'][0]:.2f}, {result['logits_range'][1]:.2f}] {status}")
