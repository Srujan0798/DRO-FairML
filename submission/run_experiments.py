"""
Main experiment runner for DRO-FAIR project.
Runs Naive-FAIR and DRO-FAIR on Adult, Credit, LSAC under ADVERSARIAL corruption.
Supports checkpointing to resume interrupted runs.

CRITICAL FIXES:
1. Adversarial corruption replacing the paper's random noise protocol.
2. Paper uses σ(τ·f_θ(x)) [MULTIPLY], not division.
3. 10 seeds for proper statistical significance.
4. Runtime measurement for each method.
5. Test-time evaluation on BOTH clean and corrupted test data.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json
import pickle
import time
from tqdm import tqdm
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.training.standard_ml import StandardMLTrainer
from src.evaluation.metrics import compute_accuracy, compute_dp_violation, compute_if_violation, compute_metrics_torch


def get_temperature(alpha):
    """Tune temperature τ based on corruption level.
    Paper line 1797-1799: τ=100 for α≤0.3, τ=1 at α=0.4 on Adult."""
    return 1.0 if alpha >= 0.4 else 100.0


def get_lambda_max(dataset, alpha):
    """Dataset-adaptive lambda_max cap.

    Adult has baseline DP ~0.17 (8x higher than Credit/LSAC's ~0.02).
    With lambda_max=1.5, the DP penalty (lambda*DP) can reach 0.255 on Adult,
    dominating the BCE loss and forcing model collapse via runaway dual ascent.

    Cap lambda_max proportional to baseline DP to prevent the feedback loop on
    high-baseline-DP datasets while preserving full strength on Credit/LSAC.
    """
    if dataset == 'adult' and alpha >= 0.2:
        return 0.5
    return 1.5


def corrupt_test_data_adversarial(X_test, y_test, a_test, alpha, seed, model, device):
    """Apply adversarial corruption to test data for test-time evaluation."""
    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed + 1000
    )
    X_test_c, y_test_c, a_test_c, _ = corruptor.corrupt(
        X_test, y_test, a_test, model=model, device=device
    )
    return X_test_c, y_test_c, a_test_c


def run_single_experiment(dataset_name, alpha, seed, device='cpu', verbose=False):
    """Run a single experiment (one dataset, one alpha, one seed)."""

    # Set all random seeds for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()

    # Load data
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)

    # Paper §G.6: τ=100 for α≤0.3, τ=1 for α≥0.4 (same for train and eval).
    tau_train = get_temperature(alpha)
    tau_eval = tau_train
    input_dim = X_train.shape[1]

    # Pretrain with standard ML on CLEAN data to prevent degeneracy
    model_pretrained = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    pretrainer = StandardMLTrainer(model_pretrained, device=device, epochs=15, lr=1e-3)
    pretrainer.fit(X_train, y_train, verbose=False)

    # Project variant: replace the paper's random corruption with adversarial noise.
    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed
    )
    X_train_c, y_train_c, a_train_c, _ = corruptor.corrupt(
        X_train, y_train, a_train, model=model_pretrained, device=device
    )

    # Corrupt test data for test-time evaluation
    X_test_c, y_test_c, a_test_c = corrupt_test_data_adversarial(
        X_test, y_test, a_test, alpha, seed, model_pretrained, device
    )

    results = {
        'dataset': dataset_name,
        'alpha': alpha,
        'seed': seed,
        'naive': {},
        'dro': {}
    }

    # === Naive-FAIR ===
    naive_start = time.time()

    model_naive = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    # Train from random initialization (no warm-start) as per paper
    trainer_naive = NaiveFairTrainer(
        model_naive, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lambda_max=1.5,
        tau=tau_train, k=5, gamma=0.0,
        epochs=60, weight_decay=1e-4, tau_warmup_epochs=15
    )

    if verbose:
        print(f"  Training Naive-FAIR...")
    trainer_naive.fit(X_train_c, y_train_c, a_train_c,
                      X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)

    naive_time = time.time() - naive_start

    # Evaluate on CLEAN test data
    results['naive']['time'] = naive_time
    results['naive']['clean'] = compute_metrics_torch(
        trainer_naive.model, X_test, y_test, a_test,
        device=device, temperature=tau_eval, k=5, gamma=0.0
    )

    # Evaluate on CORRUPTED test data (accuracy against CLEAN labels)
    results['naive']['corrupted'] = compute_metrics_torch(
        trainer_naive.model, X_test_c, y_test, a_test_c,
        device=device, temperature=tau_eval, k=5, gamma=0.0
    )

    # === DRO-FAIR ===
    dro_start = time.time()

    model_dro = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    # Train from random initialization (no warm-start) as per paper
    lambda_max = get_lambda_max(dataset_name, alpha)
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=lambda_max,
        tau=tau_train, beta=5.0, k=5, gamma=0.0,
        K_inner=10, epochs=60, weight_decay=1e-4, tau_warmup_epochs=15,
        lambda_warmstart=0.01
    )

    if verbose:
        print(f"  Training DRO-FAIR...")
    trainer_dro.fit(X_train_c, y_train_c, a_train_c,
                    X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)

    dro_time = time.time() - dro_start

    # Evaluate on CLEAN test data
    results['dro']['time'] = dro_time
    results['dro']['clean'] = compute_metrics_torch(
        trainer_dro.model, X_test, y_test, a_test,
        device=device, temperature=tau_eval, k=5, gamma=0.0
    )

    # Evaluate on CORRUPTED test data (accuracy against CLEAN labels)
    results['dro']['corrupted'] = compute_metrics_torch(
        trainer_dro.model, X_test_c, y_test, a_test_c,
        device=device, temperature=tau_eval, k=5, gamma=0.0
    )

    results['total_time'] = time.time() - start_time
    
    # Cast metrics from torch/numpy objects to standard python floats for JSON serialization
    for method in ['naive', 'dro']:
        for eval_type in ['clean', 'corrupted']:
            results[method][eval_type] = {
                k: float(v) for k, v in results[method][eval_type].items()
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
    runtime_path = os.path.join(output_dir, 'runtimes.json')

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

    # Aggregate runtime by method
    naive_times = []
    dro_times = []

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

                    # Track runtimes
                    naive_times.append(result['naive']['time'])
                    dro_times.append(result['dro']['time'])

                    # Save after each completed seed. A full run is multi-hour, and
                    # losing 1-4 completed seeds on interruption is too expensive.
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({'results': all_results, 'completed_keys': list(completed_keys)}, f)

                except Exception as e:
                    print(f"    Seed {seed} failed: {e}")
                    import traceback
                    traceback.print_exc()

            # Aggregate results for this (dataset, alpha)
            if seed_results:
                for method in ['naive', 'dro']:
                    for eval_type in ['clean', 'corrupted']:
                        accs = [r[method][eval_type]['accuracy'] for r in seed_results]
                        dps = [r[method][eval_type]['dp_violation'] for r in seed_results]
                        ifs = [r[method][eval_type]['if_violation'] for r in seed_results]

                        print(f"    {method.upper()} ({eval_type}): Acc={np.mean(accs):.4f}±{np.std(accs)/np.sqrt(len(accs)):.4f}, "
                              f"DP={np.mean(dps):.4f}±{np.std(dps)/np.sqrt(len(dps)):.4f}, "
                              f"IF={np.mean(ifs):.4f}±{np.std(ifs)/np.sqrt(len(ifs)):.4f}")

    # Save runtimes
    runtime_data = {
        'naive_mean': float(np.mean(naive_times)) if naive_times else 0,
        'naive_std': float(np.std(naive_times)) if naive_times else 0,
        'dro_mean': float(np.mean(dro_times)) if dro_times else 0,
        'dro_std': float(np.std(dro_times)) if dro_times else 0,
        'overhead': float(np.mean(dro_times) / np.mean(naive_times)) if np.mean(naive_times) > 0 else 0,
        'n_experiments': len(naive_times)
    }
    with open(runtime_path, 'w') as f:
        json.dump(runtime_data, f, indent=2)

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
    print(f"Runtime overhead: {runtime_data['overhead']:.2f}x (DRO vs Naive)")
    print(f"{'='*60}")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = run_all_experiments(
        datasets=args.datasets,
        alphas=args.alphas,
        n_seeds=args.n_seeds,
        device=device,
        output_dir=args.output_dir
    )
