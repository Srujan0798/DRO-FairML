#!/usr/bin/env python3
"""
Experiment runner for UTKFace dataset.
Trains Naive-FAIR and DRO-FAIR on UTKFace with optional adversarial corruption.

Canonical server usage (tau=1 fixed, K_inner=10, full provenance):
    python3 experiments/run_utkface.py --attack dp --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 --tau 1.0 --k_inner 10
    python3 experiments/run_utkface.py --smoke --attack combined --tau 1.0

Server batch (preferred for flair2):
    python3 experiments/run_utkface_server.py --attack dp --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 --tau 1.0 --k_inner 10
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.models.cnn_classifier import CNNClassifier
from src.corruption.adversarial import AdversarialCorruptor, FairnessTargetedPGD
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch
from src.temperature import get_temperature


def _add_utkface_provenance(result, k_inner, tau, lambda_max, attack, pgd_steps, n_seeds_planned, epochs, coordinated=False, radii_mode='uniform'):
    """Record canonical provenance for every UTKFace row (matches tabular _add_provenance contract).
    Keys: k_inner, tau, lambda_max, attack, pgd_steps, n_seeds_planned, epochs + extras.
    """
    result.update({
        'k_inner': int(k_inner),
        'tau': float(tau),
        'lambda_max': float(lambda_max),
        'attack': str(attack),
        'pgd_steps': int(pgd_steps),
        'n_seeds_planned': int(n_seeds_planned),
        'epochs': int(epochs),
        'coordinated': bool(coordinated),
        'radii_mode': str(radii_mode),
    })
    return result


def _make_synthetic_utkface(n=1000, dim=512, seed=42):
    """Generate synthetic UTKFace-like data when real data unavailable."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim).astype(np.float32)
    y = rng.randint(0, 2, n).astype(np.float32)
    a = rng.randint(0, 2, n).astype(np.int64)
    return X, y, a


def run_single_utkface_experiment(dataset_name, alpha, seed, device='cpu', verbose=False,
                                  lambda_max=1.5, attack='adversarial',
                                  save_lambda_history=False,
                                  tau=None, k_inner=10, epochs=60, pgd_steps=20,
                                  coordinated=False, n_seeds_planned=5):
    """Run single UTKFace experiment.

    attack: 'adversarial' (original AdversarialCorruptor), or 'dp' | 'if' | 'combined'
            for FairnessTargetedPGD modes.
    lambda_max: cap on dual variables — used to test H3 (inner-max overshoot on
                continuous embeddings). Default 1.5; try 0.5 to test.
    save_lambda_history: persist per-epoch λ_DP/λ_IF/g_DP/g_IF in the result
                         dict for the trajectory diagnostic.
    tau: fixed temperature (canonical: 1.0). If None, falls back to stepped get_temperature.
    k_inner: K_inner for DroFairTrainer (canonical: 10).
    epochs, pgd_steps, coordinated, n_seeds_planned: full provenance recording.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()

    try:
        X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
            get_dataset(dataset_name, random_state=seed)
        # UTKFace: override protected attribute to gender (binary) for fairness consistency
        # load_utkface returns a=race (5-class) by default, but DRO trainer assumes binary
        if dataset_name.lower() == 'utkface':
            # Re-extract gender from the raw data path if possible, else use y as proxy
            a_train = y_train.astype(np.int64)
            a_val = y_val.astype(np.int64)
            a_test = y_test.astype(np.int64)
    except RuntimeError as e:
        if 'UTKFace' in str(e) or 'No UTKFace' in str(e):
            print(f"  UTKFace not available ({e}), using synthetic data")
            X_train, y_train, a_train = _make_synthetic_utkface(n=800, seed=seed)
            X_test, y_test, a_test = _make_synthetic_utkface(n=200, seed=seed+999)
            X_val = X_test.copy()
            y_val = y_test.copy()
            a_val = a_test.copy()
            dname = 'UTKFace (synthetic)'
        else:
            raise

    if tau is None:
        tau = get_temperature(alpha)
    input_dim = X_train.shape[1]

    if attack == 'adversarial':
        corruptor = AdversarialCorruptor(
            alpha=alpha, epsilon=0.1,
            feature_attack=True, label_flip=True, attr_flip=True,
            coordinated=coordinated, random_state=seed
        )
    elif attack in ('dp', 'if', 'combined'):
        corruptor = FairnessTargetedPGD(
            alpha=alpha, target_metric=attack, pgd_steps=pgd_steps,
            coordinated=coordinated, random_state=seed
        )
    else:
        raise ValueError(f"unknown attack: {attack!r}")

    X_train_c, y_train_c, a_train_c, _ = corruptor.corrupt(
        X_train, y_train, a_train, model=None, device=device
    )
    X_test_c, y_test_c, a_test_c, _ = corruptor.corrupt(
        X_test, y_test, a_test, model=None, device=device
    )

    results = {
        'dataset': dataset_name,
        'alpha': alpha,
        'seed': seed,
        'attack': attack,
        'lambda_max': lambda_max,
        'naive': {},
        'dro': {}
    }

    model_naive = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer_naive = NaiveFairTrainer(
        model_naive, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lambda_max=lambda_max,
        tau=tau, k=5, gamma=0.0,
        epochs=epochs, weight_decay=1e-4, tau_warmup_epochs=15
    )
    trainer_naive.fit(X_train_c, y_train_c, a_train_c,
                      X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)

    results['naive']['clean'] = compute_metrics_torch(
        trainer_naive.model, X_test, y_test, a_test,
        device=device, temperature=tau, k=5, gamma=0.0
    )
    results['naive']['corrupted'] = compute_metrics_torch(
        trainer_naive.model, X_test_c, y_test, a_test_c,
        device=device, temperature=tau, k=5, gamma=0.0
    )

    model_dro = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer_dro = DroFairTrainer(
        model_dro, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=lambda_max,
        tau=tau, beta=5.0, k=5, gamma=0.0,
        K_inner=k_inner, epochs=epochs, weight_decay=1e-4, tau_warmup_epochs=15
    )
    dro_history = trainer_dro.fit(X_train_c, y_train_c, a_train_c,
                                  X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)
    if save_lambda_history:
        results['dro']['lambda_history'] = {
            'lambda_dp': dro_history.get('lambda_dp', []),
            'lambda_if': dro_history.get('lambda_if', []),
            'g_dp': dro_history.get('g_dp', []),
            'g_if': dro_history.get('g_if', []),
        }

    results['dro']['clean'] = compute_metrics_torch(
        trainer_dro.model, X_test, y_test, a_test,
        device=device, temperature=tau, k=5, gamma=0.0
    )
    results['dro']['corrupted'] = compute_metrics_torch(
        trainer_dro.model, X_test_c, y_test, a_test_c,
        device=device, temperature=tau, k=5, gamma=0.0
    )

    results['total_time'] = time.time() - start_time

    for method in ['naive', 'dro']:
        for eval_type in ['clean', 'corrupted']:
            results[method][eval_type] = {
                k: float(v) for k, v in results[method][eval_type].items()
            }

    # Add full provenance (canonical K_inner=10, tau=1 fixed etc.)
    results = _add_utkface_provenance(
        results, k_inner=k_inner, tau=tau, lambda_max=lambda_max,
        attack=attack, pgd_steps=pgd_steps, n_seeds_planned=n_seeds_planned, epochs=epochs,
        coordinated=coordinated
    )

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="UTKFace runner (Naive-FAIR vs DRO-FAIR). Supports FairnessTargetedPGD (dp/if/combined) and canonical tau=1/K_inner=10.",
        epilog="""Examples (canonical for server):
  python experiments/run_utkface.py --smoke --attack dp --tau 1.0 --k_inner 10
  python experiments/run_utkface.py --attack combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 --tau 1.0 --k_inner 10 --pgd_steps 20
  # Preferred: use experiments/run_utkface_server.py for batched per-attack resume + full grid on flair2.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--datasets', nargs='+', default=['utkface'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--smoke', action='store_true', help='Run single seed only (smoke test; forces n_seeds=1, alpha=0.2, small epochs for speed)')
    parser.add_argument('--attack', choices=['adversarial', 'dp', 'if', 'combined'],
                        default='adversarial',
                        help="Corruption type: 'adversarial' (default, multi-modal) or "
                             "FairnessTargetedPGD modes 'dp'|'if'|'combined' (canonical server path)")
    parser.add_argument('--lambda_max', type=float, default=1.5,
                        help='Cap on DRO dual variables (H3 test: try 0.5)')
    parser.add_argument('--tau', type=float, default=None,
                        help='Temperature (canonical: 1.0 fixed for all alphas; if omitted uses stepped schedule)')
    parser.add_argument('--k_inner', type=int, default=10,
                        help='K_inner for DRO inner maximization (canonical: 10)')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs (canonical: 60)')
    parser.add_argument('--pgd_steps', type=int, default=20, help='PGD steps for FairnessTargetedPGD (canonical: 20)')
    parser.add_argument('--coordinated', action='store_true', help='Use coordinated corruption (default off for canonical)')
    parser.add_argument('--save_lambda_history', action='store_true',
                        help='Persist per-epoch λ_DP/λ_IF trajectories for diagnostic')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output JSON path')
    args = parser.parse_args()

    if args.smoke:
        args.n_seeds = 1
        args.alphas = [0.2]
        args.epochs = 5
        args.pgd_steps = 3
        args.k_inner = 3
        print("SMOKE TEST MODE: 1 seed, alpha=0.2, reduced epochs/k_inner/pgd for fast CPU check")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eff_tau = args.tau if args.tau is not None else '(stepped)'
    print(f"Using device: {device} | attack={args.attack} | tau={eff_tau} | k_inner={args.k_inner} | lambda_max={args.lambda_max}")

    os.makedirs('results', exist_ok=True)
    if args.output:
        results_path = args.output
    else:
        tag = f"_{args.attack}" if args.attack != 'adversarial' else ''
        tag += f"_lmax{args.lambda_max}" if args.lambda_max != 1.5 else ''
        if args.tau is not None:
            tag += f"_tau{args.tau}"
        results_path = f'results/utkface_results{tag}.json'

    all_results = []
    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                print(f"\n[{dataset}] alpha={alpha} seed={seed}")
                try:
                    t0 = time.time()
                    result = run_single_utkface_experiment(
                        dataset, alpha, seed, device=device, verbose=False,
                        lambda_max=args.lambda_max, attack=args.attack,
                        save_lambda_history=args.save_lambda_history,
                        tau=args.tau, k_inner=args.k_inner, epochs=args.epochs,
                        pgd_steps=args.pgd_steps, coordinated=args.coordinated,
                        n_seeds_planned=args.n_seeds,
                    )
                    elapsed = time.time() - t0
                    print(f"  Done in {elapsed:.0f}s | "
                          f"Naive clean: acc={result['naive']['clean']['accuracy']:.3f} "
                          f"dp={result['naive']['clean']['dp_violation']:.3f} "
                          f"if={result['naive']['clean']['if_violation']:.3f} | "
                          f"DRO clean: acc={result['dro']['clean']['accuracy']:.3f} "
                          f"dp={result['dro']['clean']['dp_violation']:.3f} "
                          f"if={result['dro']['clean']['if_violation']:.3f}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {results_path}")
    # Quick provenance check on last result if present
    if all_results:
        last = all_results[-1]
        prov = {k: last.get(k) for k in ('k_inner', 'tau', 'pgd_steps', 'epochs', 'n_seeds_planned')}
        print(f"Provenance on last row: {prov}")


if __name__ == '__main__':
    main()