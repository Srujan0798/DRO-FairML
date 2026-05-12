"""
Random vs Adversarial Corruption Comparison
============================================
Directly compares the impact of random noise vs adversarial noise on
both Naive-FAIR and DRO-FAIR. This addresses the ablation requirement:
"random vs adversarial noise impact" from the original task.

Key difference:
- Random: Gaussian feature noise + uniform random label/attribute flips
- Adversarial: FGSM/PGD feature attacks + coordinated label/attribute flips
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json
from tqdm import tqdm
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor, RandomCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


def run_comparison(dataset_name, alpha, seed, device='cpu'):
    """Run Naive-FAIR and DRO-FAIR under both random and adversarial corruption."""
    
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)
    
    tau = get_temperature(alpha)
    input_dim = X_train.shape[1]
    
    # --- Random corruption ---
    random_corruptor = RandomCorruptor(
        alpha=alpha, epsilon=0.1,
        feature_attack=True, label_flip=True, attr_flip=True,
        random_state=seed
    )
    X_train_rand, y_train_rand, a_train_rand, _ = random_corruptor.corrupt(
        X_train, y_train, a_train
    )
    
    # --- Adversarial corruption ---
    adv_corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1, pgd_steps=5, pgd_step_size=0.02,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed
    )
    X_train_adv, y_train_adv, a_train_adv, _ = adv_corruptor.corrupt(
        X_train, y_train, a_train, device=device
    )
    
    results = {
        'dataset': dataset_name,
        'alpha': alpha,
        'seed': seed,
        'random': {},
        'adversarial': {}
    }
    
    for corruption_type, X_tr, y_tr, a_tr in [
        ('random', X_train_rand, y_train_rand, a_train_rand),
        ('adversarial', X_train_adv, y_train_adv, a_train_adv)
    ]:
        res = {'naive': {}, 'dro': {}}
        
        # Naive-FAIR
        model_naive = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
        trainer_naive = NaiveFairTrainer(
            model_naive, device=device,
            lr_theta=1e-3, lr_lambda=5e-3, lambda_max=10.0,
            tau=tau, k=5, gamma=0.0, epochs=30, weight_decay=1e-4
        )
        trainer_naive.fit(X_tr, y_tr, a_tr, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
        preds_naive = trainer_naive.predict(X_test)
        res['naive'] = {
            'accuracy': float(compute_accuracy(y_test, preds_naive)),
            'dp_violation': float(compute_dp_violation(preds_naive, a_test)),
            'if_violation': float(compute_if_violation(X_test, preds_naive, a_test, k=5, gamma=0.0))
        }
        
        # DRO-FAIR
        model_dro = MLPClassifier(input_dim, hidden_dims=[64, 32], dropout=0.1)
        trainer_dro = DroFairTrainer(
            model_dro, alpha=alpha, device=device,
            lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=10.0,
            tau=tau, beta=5.0, k=5, gamma=0.0,
            K_inner=10, epochs=30, weight_decay=1e-4
        )
        trainer_dro.fit(X_tr, y_tr, a_tr, X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)
        preds_dro = trainer_dro.predict(X_test)
        res['dro'] = {
            'accuracy': float(compute_accuracy(y_test, preds_dro)),
            'dp_violation': float(compute_dp_violation(preds_dro, a_test)),
            'if_violation': float(compute_if_violation(X_test, preds_dro, a_test, k=5, gamma=0.0))
        }
        
        results[corruption_type] = res
    
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("="*70)
    print("Random vs Adversarial Corruption Comparison")
    print("="*70)
    
    datasets = ['adult', 'credit']
    alphas = [0.2, 0.3]
    n_seeds = 3
    
    all_results = []
    
    for dataset in datasets:
        for alpha in alphas:
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset.upper()} | α = {alpha}")
            print(f"{'='*70}")
            
            for seed in tqdm(range(n_seeds), desc=f"{dataset} α={alpha}"):
                result = run_comparison(dataset, alpha, seed, device)
                all_results.append(result)
            
            # Aggregate and print
            for corruption_type in ['random', 'adversarial']:
                print(f"\n  [{corruption_type.upper()} CORRUPTION]")
                for method in ['naive', 'dro']:
                    accs = [r[corruption_type][method]['accuracy'] for r in all_results
                            if r['dataset'] == dataset and r['alpha'] == alpha]
                    dps = [r[corruption_type][method]['dp_violation'] for r in all_results
                           if r['dataset'] == dataset and r['alpha'] == alpha]
                    ifs = [r[corruption_type][method]['if_violation'] for r in all_results
                           if r['dataset'] == dataset and r['alpha'] == alpha]
                    
                    n = len(accs)
                    print(f"    {method.upper():10s}: Acc={np.mean(accs):.4f}±{np.std(accs)/np.sqrt(n):.4f}, "
                          f"DP={np.mean(dps):.4f}±{np.std(dps)/np.sqrt(n):.4f}, "
                          f"IF={np.mean(ifs):.4f}±{np.std(ifs)/np.sqrt(n):.4f}")
            
            # Compute adversarial advantage
            for method in ['naive', 'dro']:
                rand_dps = [r['random'][method]['dp_violation'] for r in all_results
                           if r['dataset'] == dataset and r['alpha'] == alpha]
                adv_dps = [r['adversarial'][method]['dp_violation'] for r in all_results
                          if r['dataset'] == dataset and r['alpha'] == alpha]
                dp_increase = (np.mean(adv_dps) - np.mean(rand_dps)) / np.mean(rand_dps) * 100 if np.mean(rand_dps) > 0 else 0
                print(f"    {method.upper()} DP increase (adv vs random): {dp_increase:.1f}%")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/random_vs_adversarial.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n" + "="*70)
    print("Results saved to results/random_vs_adversarial.json")
    print("="*70)


if __name__ == '__main__':
    main()
