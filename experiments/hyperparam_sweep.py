#!/usr/bin/env python3
"""
Hyperparameter Sweep for DRO-FAIR
==================================

Tests 6 configurations on Adult α=0.2 with 3 seeds each to find
settings where DRO-FAIR beats Naive-FAIR on DP violation.

Run: python experiments/hyperparam_sweep.py

Output: Prints a table comparing Naive vs DRO DP for each config.
If any config shows DRO DP < Naive DP across all 3 seeds, that's your fix.

Time: ~1-2 hours on CPU (18 experiments total).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json
import time
from datetime import datetime

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.training.standard_ml import StandardMLTrainer
from src.evaluation.metrics import compute_metrics_torch


def run_single_config(dataset_name, alpha, seed, config_name, config,
                      device='cpu', verbose=False):
    """Run one (dataset, alpha, seed, config) experiment."""
    
    # Set all random seeds
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)
    
    input_dim = X_train.shape[1]
    
    # Temperature
    tau = 1.0 if alpha >= 0.4 else 100.0
    
    # Corrupt training data (random corruption, matching paper protocol)
    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1,
        feature_attack=True, label_flip=True, attr_flip=True,
        random_state=seed
    )
    X_train_c, y_train_c, a_train_c, _ = corruptor.corrupt(X_train, y_train, a_train)
    
    # === Naive-FAIR ===
    model_naive = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer_naive = NaiveFairTrainer(
        model_naive, device=device,
        lr_theta=1e-3, lr_lambda=config.get('lr_lambda', 5e-3),
        lambda_max=10.0, tau=tau, k=5, gamma=0.0,
        epochs=config.get('epochs', 30),
        weight_decay=1e-4,
        tau_warmup_epochs=5
    )
    trainer_naive.fit(X_train_c, y_train_c, a_train_c,
                      X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)
    
    naive_metrics = compute_metrics_torch(
        trainer_naive.model, X_test, y_test, a_test,
        device=device, temperature=tau, k=5, gamma=0.0
    )
    
    # === DRO-FAIR ===
    model_dro = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device=device,
        lr_theta=1e-3,
        lr_lambda=config.get('lr_lambda', 5e-3),
        lr_p=config.get('lr_p', 5e-3),
        lambda_max=10.0, tau=tau, beta=5.0, k=5, gamma=0.0,
        K_inner=config.get('K_inner', 10),
        epochs=config.get('epochs', 30),
        weight_decay=1e-4,
        use_dp=True,
        use_if=config.get('use_if', True)
    )
    trainer_dro.fit(X_train_c, y_train_c, a_train_c,
                    X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)
    
    dro_metrics = compute_metrics_torch(
        trainer_dro.model, X_test, y_test, a_test,
        device=device, temperature=tau, k=5, gamma=0.0
    )
    
    return {
        'config': config_name,
        'seed': seed,
        'naive_acc': float(naive_metrics['accuracy']),
        'naive_dp': float(naive_metrics['dp_violation']),
        'naive_if': float(naive_metrics['if_violation']),
        'dro_acc': float(dro_metrics['accuracy']),
        'dro_dp': float(dro_metrics['dp_violation']),
        'dro_if': float(dro_metrics['if_violation']),
        'dro_wins_dp': float(dro_metrics['dp_violation']) < float(naive_metrics['dp_violation'])
    }


def main():
    """Run the full sweep."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Configurations to test
    # Each config overrides specific hyperparameters from the baseline
    configs = {
        'baseline': {
            'description': 'Current settings (lr_lambda=5e-3, K=10, lr_p=5e-3, epochs=30, joint DP+IF)',
            'params': {}
        },
        'lr_lambda_0.02': {
            'description': 'Faster dual ascent (lr_lambda=0.02)',
            'params': {'lr_lambda': 0.02}
        },
        'K_inner_50': {
            'description': 'More inner steps (K_inner=50)',
            'params': {'K_inner': 50}
        },
        'lr_p_0.02': {
            'description': 'Faster p-weight updates (lr_p=0.02)',
            'params': {'lr_p': 0.02}
        },
        'epochs_60': {
            'description': 'More training epochs (epochs=60)',
            'params': {'epochs': 60}
        },
        'dp_only': {
            'description': 'DP only, no IF (use_if=False)',
            'params': {'use_if': False}
        },
    }
    
    dataset = 'adult'
    alpha = 0.2
    n_seeds = 3
    
    all_results = []
    
    total_runs = len(configs) * n_seeds
    run_counter = 0
    
    for config_name, config_info in configs.items():
        print(f"\n{'='*80}")
        print(f"Config: {config_name}")
        print(f"  {config_info['description']}")
        print(f"  Params: {config_info['params']}")
        print(f"{'='*80}")
        
        config_results = []
        
        for seed in range(n_seeds):
            run_counter += 1
            print(f"\n  [{run_counter}/{total_runs}] Seed {seed}...", end=' ', flush=True)
            start = time.time()
            
            try:
                result = run_single_config(
                    dataset, alpha, seed, config_name, config_info['params'],
                    device=device, verbose=False
                )
                config_results.append(result)
                elapsed = time.time() - start
                print(f"DONE ({elapsed:.1f}s)")
                print(f"    Naive: Acc={result['naive_acc']:.4f} DP={result['naive_dp']:.4f}")
                print(f"    DRO:   Acc={result['dro_acc']:.4f} DP={result['dro_dp']:.4f} "
                      f"{'✓ WINS' if result['dro_wins_dp'] else '✗ LOSES'}")
                
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        all_results.extend(config_results)
    
    # Save results
    output_path = 'results/hyperparam_sweep.json'
    os.makedirs('results', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Config':<20} {'Naive DP':>12} {'DRO DP':>12} {'DRO Wins':>10} {'Status':>15}")
    print("-" * 80)
    
    best_config = None
    best_wins = -1
    
    for config_name in configs.keys():
        subset = [r for r in all_results if r['config'] == config_name]
        if not subset:
            continue
        
        naive_dps = [r['naive_dp'] for r in subset]
        dro_dps = [r['dro_dp'] for r in subset]
        wins = sum(1 for r in subset if r['dro_wins_dp'])
        
        mean_naive_dp = np.mean(naive_dps)
        mean_dro_dp = np.mean(dro_dps)
        se_naive = np.std(naive_dps) / np.sqrt(len(naive_dps))
        se_dro = np.std(dro_dps) / np.sqrt(len(dro_dps))
        
        if wins > best_wins:
            best_wins = wins
            best_config = config_name
        
        if wins == n_seeds:
            status = "🎉 ALL WINS"
        elif wins >= n_seeds // 2 + 1:
            status = "✓ Majority"
        elif wins > 0:
            status = "~ Mixed"
        else:
            status = "✗ All losses"
        
        print(f"{config_name:<20} {mean_naive_dp:>6.4f}±{se_naive:.3f} "
              f"{mean_dro_dp:>6.4f}±{se_dro:.3f} {wins:>5}/{n_seeds} {status:>15}")
    
    print("-" * 80)
    print(f"\nBest config: {best_config} ({best_wins}/{n_seeds} wins)")
    print(f"Results saved to: {output_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return code: 0 if any config wins all seeds, 1 otherwise
    if best_wins == n_seeds:
        print("\n✅ SUCCESS: Found a config where DRO beats Naive on ALL seeds!")
        print("   Use this config for the full experiment suite.")
        return 0
    elif best_wins >= 2:
        print(f"\n⚠️  PARTIAL: Best config wins {best_wins}/{n_seeds} seeds.")
        print("   Try combining the best configs or tuning further.")
        return 1
    else:
        print("\n❌ FAILURE: No config shows DRO beating Naive.")
        print("   Full-batch training is likely insufficient.")
        print("   Consider implementing minibatch training (see MASTER_PROTOCOL.md Task 1).")
        return 2


if __name__ == '__main__':
    sys.exit(main())
