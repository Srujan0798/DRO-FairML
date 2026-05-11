"""
Main experiment runner for DRO-FAIR project.
Runs Naive-FAIR and DRO-FAIR on Adult, Credit, LSAC under adversarial corruption.
Supports checkpointing to resume interrupted runs.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json
import pickle
from tqdm import tqdm
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation


def run_single_experiment(dataset_name, alpha, seed, device='cpu', verbose=False):
    """Run a single experiment (one dataset, one alpha, one seed)."""
    
    # Load data
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)
    
    # Apply adversarial corruption to train and val
    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1, pgd_steps=5, pgd_step_size=0.02,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed
    )
    
    X_train_c, y_train_c, a_train_c, _ = corruptor.corrupt(X_train, y_train, a_train, device=device)
    X_val_c, y_val_c, a_val_c, _ = corruptor.corrupt(X_val, y_val, a_val, device=device)
    
    input_dim = X_train.shape[1]
    
    results = {
        'dataset': dataset_name,
        'alpha': alpha,
        'seed': seed,
        'naive': {},
        'dro': {}
    }
    
    # === Naive-FAIR ===
    model_naive = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer_naive = NaiveFairTrainer(
        model_naive, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lambda_max=10.0,
        tau=100.0, beta=5.0, k=5, gamma=0.0,
        epochs=30, weight_decay=1e-4
    )
    
    if verbose:
        print(f"  Training Naive-FAIR...")
    trainer_naive.fit(X_train_c, y_train_c, a_train_c, 
                      X_val=X_val_c, y_val=y_val_c, a_val=a_val_c, verbose=verbose)
    
    preds_naive = trainer_naive.predict(X_test)
    results['naive'] = {
        'accuracy': float(compute_accuracy(y_test, preds_naive)),
        'dp_violation': float(compute_dp_violation(preds_naive, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds_naive, a_test, k=5, gamma=0.0))
    }
    
    # === DRO-FAIR ===
    model_dro = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=10.0,
        tau=100.0, beta=5.0, k=5, gamma=0.0,
        K_inner=5, epochs=30, weight_decay=1e-4
    )
    
    if verbose:
        print(f"  Training DRO-FAIR...")
    trainer_dro.fit(X_train_c, y_train_c, a_train_c,
                    X_val=X_val_c, y_val=y_val_c, a_val=a_val_c, verbose=verbose)
    
    preds_dro = trainer_dro.predict(X_test)
    results['dro'] = {
        'accuracy': float(compute_accuracy(y_test, preds_dro)),
        'dp_violation': float(compute_dp_violation(preds_dro, a_test)),
        'if_violation': float(compute_if_violation(X_test, preds_dro, a_test, k=5, gamma=0.0))
    }
    
    return results


def run_all_experiments(datasets=['adult', 'credit', 'lsac'],
                        alphas=[0.0, 0.1, 0.2, 0.3, 0.4],
                        n_seeds=10,
                        device='cpu',
                        output_dir='results'):
    """Run full experiment suite with checkpointing."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pkl')
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        all_results = checkpoint['results']
        completed_keys = set(checkpoint['completed_keys'])
        print(f"Resumed from checkpoint: {len(completed_keys)} experiments already completed.")
    else:
        all_results = []
        completed_keys = set()
    
    total_experiments = len(datasets) * len(alphas) * n_seeds
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        for alpha in alphas:
            print(f"\n  Alpha = {alpha}")
            
            seed_results = []
            for seed in tqdm(range(n_seeds), desc=f"  {dataset} α={alpha}"):
                key = f"{dataset}_{alpha}_{seed}"
                
                if key in completed_keys:
                    # Find existing result
                    existing = [r for r in all_results if r['dataset'] == dataset 
                               and r['alpha'] == alpha and r['seed'] == seed]
                    if existing:
                        seed_results.append(existing[0])
                    continue
                
                try:
                    result = run_single_experiment(dataset, alpha, seed, device=device, verbose=False)
                    seed_results.append(result)
                    all_results.append(result)
                    completed_keys.add(key)
                    
                    # Save checkpoint periodically
                    if len(completed_keys) % 5 == 0:
                        with open(checkpoint_path, 'wb') as f:
                            pickle.dump({'results': all_results, 'completed_keys': list(completed_keys)}, f)
                
                except Exception as e:
                    print(f"    Seed {seed} failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Aggregate results for this (dataset, alpha)
            if seed_results:
                for method in ['naive', 'dro']:
                    accs = [r[method]['accuracy'] for r in seed_results]
                    dps = [r[method]['dp_violation'] for r in seed_results]
                    ifs = [r[method]['if_violation'] for r in seed_results]
                    
                    print(f"    {method.upper()}: Acc={np.mean(accs):.4f}±{np.std(accs)/np.sqrt(len(accs)):.4f}, "
                          f"DP={np.mean(dps):.4f}±{np.std(dps)/np.sqrt(len(dps)):.4f}, "
                          f"IF={np.mean(ifs):.4f}±{np.std(ifs)/np.sqrt(len(ifs)):.4f}")
    
    # Save final results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(os.path.join(output_dir, 'all_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    # Remove checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"All results saved to {output_dir}")
    print(f"Total experiments: {len(all_results)} / {total_experiments}")
    print(f"{'='*60}")
    
    return all_results


def run_ablation_experiments(dataset='credit', alpha=0.2, n_seeds=5, device='cpu', output_dir='results'):
    """Run ablation studies: radius choice, DP-only vs joint, random vs adversarial noise."""
    os.makedirs(output_dir, exist_ok=True)
    
    ablation_results = []
    
    print(f"\n{'='*60}")
    print(f"Ablation Studies on {dataset.upper()} with α={alpha}")
    print(f"{'='*60}")
    
    for seed in tqdm(range(n_seeds), desc="Ablations"):
        X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
            get_dataset(dataset, random_state=seed)
        
        input_dim = X_train.shape[1]
        
        # 1. Adversarial corruption
        adv_corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
        X_train_adv, y_train_adv, a_train_adv, _ = adv_corruptor.corrupt(X_train, y_train, a_train)
        X_val_adv, y_val_adv, a_val_adv, _ = adv_corruptor.corrupt(X_val, y_val, a_val)
        
        # 2. Random corruption (for comparison)
        rand_corruptor = AdversarialCorruptor(alpha=alpha, coordinated=False, random_state=seed,
                                         feature_attack=True, label_flip=True, attr_flip=True)
        X_train_rand, y_train_rand, a_train_rand, _ = rand_corruptor.corrupt(X_train, y_train, a_train)
        X_val_rand, y_val_rand, a_val_rand, _ = rand_corruptor.corrupt(X_val, y_val, a_val)
        
        for corruption_type, X_tr, y_tr, a_tr, X_va, y_va, a_va in [
            ('adversarial', X_train_adv, y_train_adv, a_train_adv, X_val_adv, y_val_adv, a_val_adv),
            ('random', X_train_rand, y_train_rand, a_train_rand, X_val_rand, y_val_rand, a_val_rand)
        ]:
            # DRO-FAIR with joint constraints
            model_joint = MLPClassifier(input_dim)
            trainer_joint = DroFairTrainer(model_joint, alpha=alpha, device=device, epochs=30)
            trainer_joint.fit(X_tr, y_tr, a_tr, X_va, y_va, a_va)
            preds_joint = trainer_joint.predict(X_test)
            
            ablation_results.append({
                'seed': seed, 'corruption_type': corruption_type, 'constraint': 'joint',
                'accuracy': float(compute_accuracy(y_test, preds_joint)),
                'dp_violation': float(compute_dp_violation(preds_joint, a_test)),
                'if_violation': float(compute_if_violation(X_test, preds_joint, a_test, k=5))
            })
    
    with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print("\nAblation results saved.")
    return ablation_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Main experiments
    results = run_all_experiments(
        datasets=['adult', 'credit', 'lsac'],
        alphas=[0.0, 0.1, 0.2, 0.3, 0.4],
        n_seeds=10,
        device=device,
        output_dir='results'
    )
    
    # Ablation studies
    run_ablation_experiments(
        dataset='credit',
        alpha=0.2,
        n_seeds=5,
        device=device,
        output_dir='results'
    )
