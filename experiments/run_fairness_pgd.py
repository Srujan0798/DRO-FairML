#!/usr/bin/env python3
"""
Experiment driver for Fairness-Targeted PGD attacks on tabular datasets.
Corrupts the training set with a fairness-targeted PGD attack, then trains
Naive-FAIR and DRO-FAIR on the corrupted data and evaluates on clean test.

Usage:
    python3 experiments/run_fairness_pgd.py --smoke
    python3 experiments/run_fairness_pgd.py --datasets adult --alphas 0.1 0.2 0.3 0.4
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
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch
from src.temperature import get_temperature


def _add_provenance(result, k_inner, tau, radii_mode, lambda_init, coordinated, pgd_steps, n_seeds_planned, epochs,
                    attack_k=5):
    """Ensure EVERY saved row records full config provenance per §1.4 and §4 of MASTER_PLAN.
    Mandatory keys: k_inner, tau, radii_mode, lambda_init, coordinated, pgd_steps, n_seeds_planned, epochs.
    """
    result.update({
        'k_inner': int(k_inner),
        'tau': float(tau),
        'radii_mode': str(radii_mode),
        'lambda_init': float(lambda_init),
        'coordinated': bool(coordinated),
        'pgd_steps': int(pgd_steps),
        'n_seeds_planned': int(n_seeds_planned),
        'epochs': int(epochs),
        'attack_k': int(attack_k),
    })
    return result


def run_single_experiment(dataset_name, alpha, seed, attack, method, device='cpu', verbose=False, epochs=60, k_inner=10, pgd_steps=20,
                           tau=None, lambda_init=0.0, radii_mode='uniform', coordinated=False, n_seeds_planned=3,
                           corruptor_type='adversarial', lr_lambda=5e-3, attack_k=5,
                           radii_scale=1.0, radii_clamp=None, dump_history=False,
                           pi_shrinkage_k=0.0, lambda_max=1.5, beta=5.0):
    """Run single (dataset, alpha, seed, attack, method) experiment.
    All callers must pass (or rely on defaults for) full provenance params so every row
    includes k_inner, tau, radii_mode, lambda_init, coordinated, pgd_steps, n_seeds_planned, epochs.

    Extended for Wave 1 ablations (backward-compatible defaults preserve canonical grid):
      corruptor_type: 'adversarial' (FairnessTargetedPGD, canonical) or 'random' (RandomCorruptor, A4).
      lr_lambda: dual learning rate override (A3 lambda grid; default 5e-3 = canonical).
      attack_k: k-NN neighborhood for IF attack (A1 kNN ablation; default 5 = canonical attack k).
      radii_scale (Agent N1): global multiplier on rho_dp/rho_if (default 1.0 = canonical).
      radii_clamp (Agent L2): per-group cap on rho_dp[j] (default None = canonical).
      pi_shrinkage_k: additive shrinkage of pi_clean toward 0.5 before computing rho_dp
        (LSAC-degeneracy hypothesis, alternative to radii_clamp; default 0.0 = canonical).
      lambda_max, beta (DRO only; Naive's lambda_max stays hardcoded 1.5 for baseline
        comparability): both were set once and never revisited after the tau=100 -> 1.0
        fix. lambda_max=1.5 was a conservative cap chosen to survive the OLD tau=100
        instability; beta=5.0 (tilted-risk sharpness) has never been ablated at all.
        Testing whether either can be raised now that the actual instability source
        (stepped tau) is gone, to see if DRO's fairness margin over Naive can grow
        meaningfully rather than staying marginal. Defaults are canonical (unchanged).

    Extended for Agent N2 convergence diagnostics:
      dump_history: if True AND method=='dro', dump trainer.history (per-epoch train_loss,
        val_acc, val_dp, val_if, val_loss, current_tau, ...) to
        results/history_{dataset}_{attack}_{alpha}_{seed}_{method}.json. OFF by default so
        canonical reruns are byte-identical and unaffected. Only DRO history is dumped
        because N2's convergence plots are for the DRO trainer.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset_name, random_state=seed)

    if tau is None:
        tau = get_temperature(alpha)
    input_dim = X_train.shape[1]

    if corruptor_type == 'random':
        from src.corruption.adversarial import RandomCorruptor
        attack_obj = RandomCorruptor(
            alpha=alpha, epsilon=0.3,
            feature_attack=True, label_flip=True, attr_flip=True,
            random_state=seed
        )
    else:
        attack_obj = FairnessTargetedPGD(
            alpha=alpha,
            target_metric=attack,
            pgd_steps=pgd_steps,
            epsilon=0.3,
            pgd_step_size=0.02,
            coordinated=coordinated,
            random_state=seed,
            k=attack_k
        )

    X_train_att, y_train_att, a_train_att, _ = attack_obj.corrupt(
        X_train, y_train, a_train
    )

    # Agent N1 (Kuldeep May 29): MEASURED attack effectiveness — the ΔDP the
    # corruption itself induces on the TRAINING labels, computed pre-training
    # on numpy arrays (no torch). This is the key measurement Kuldeep asked
    # for: "Does the attack affect the radius? ... if the attack is too weak,
    # then DRO would perform well?" Strength must be MEASURED, not assumed.
    # Guarded so existing callers are unaffected (defaults to None on any
    # failure). Uses compute_dp_violation (binary |P(y=1|a=0) - P(y=1|a=1)|).
    try:
        from src.evaluation.metrics import compute_dp_violation
        y_tr_np = np.asarray(y_train).astype(np.float32)
        a_tr_np = np.asarray(a_train)
        y_att_np = np.asarray(y_train_att).astype(np.float32)
        a_att_np = np.asarray(a_train_att)
        dp_clean_train = float(compute_dp_violation(y_tr_np, a_tr_np))
        dp_corrupted_train = float(compute_dp_violation(y_att_np, a_att_np))
        attack_effectiveness = abs(dp_corrupted_train - dp_clean_train)
    except Exception:
        dp_clean_train = None
        dp_corrupted_train = None
        attack_effectiveness = None

    result = {
        'dataset': dataset_name,
        'alpha': alpha,
        'seed': seed,
        'attack': attack,
        'method': method,
        'corruptor_type': corruptor_type,
        'lr_lambda': float(lr_lambda),
        'attack_k': int(attack_k),
        'radii_scale': float(radii_scale),
        'radii_clamp': (None if radii_clamp is None else float(radii_clamp)),
        'pi_shrinkage_k': float(pi_shrinkage_k),
        'lambda_max': float(lambda_max),
        'beta': float(beta),
        'attack_effectiveness': attack_effectiveness,
        'dp_clean_train': dp_clean_train,
        'dp_corrupted_train': dp_corrupted_train,
    }

    if method == 'naive':
        model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
        trainer = NaiveFairTrainer(
            model, device=device,
            lr_theta=1e-3, lr_lambda=lr_lambda, lambda_max=1.5,
            tau=tau, k=attack_k, gamma=0.0,
            epochs=epochs, weight_decay=1e-4, tau_warmup_epochs=15
        )
        trainer.fit(X_train_att, y_train_att, a_train_att,
                     X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)
        metrics = compute_metrics_torch(
            trainer.model, X_test, y_test, a_test,
            device=device, temperature=tau, k=attack_k, gamma=0.0
        )
    else:
        model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
        trainer = DroFairTrainer(
            model, alpha=alpha, device=device,
            lr_theta=1e-3, lr_lambda=lr_lambda, lr_p=5e-3, lambda_max=lambda_max,
            tau=tau, beta=beta, k=attack_k, gamma=0.0,
            K_inner=k_inner, epochs=epochs, weight_decay=1e-4, tau_warmup_epochs=15,
            lambda_init=lambda_init, radii_mode=radii_mode,
            radii_scale=radii_scale, radii_clamp=radii_clamp,
            pi_shrinkage_k=pi_shrinkage_k
        )
        trainer.fit(X_train_att, y_train_att, a_train_att,
                     X_val=X_val, y_val=y_val, a_val=a_val, verbose=verbose)
        metrics = compute_metrics_torch(
            trainer.model, X_test, y_test, a_test,
            device=device, temperature=tau, k=attack_k, gamma=0.0
        )

    result['acc_clean'] = float(metrics['accuracy'])
    result['dp_clean'] = float(metrics['dp_violation'])
    result['if_clean'] = float(metrics['if_violation'])
    result['total_time'] = time.time() - start_time

    result = _add_provenance(
        result, k_inner, tau, radii_mode, lambda_init, coordinated, pgd_steps,
        n_seeds_planned, epochs, attack_k=attack_k,
    )

    # Agent N2 convergence diagnostics: dump per-epoch DRO history to a JSON file
    # alongside the result row. Gated by dump_history (default False) so canonical
    # reruns are unaffected. Only DRO history is dumped — N2's convergence plots are
    # for the DRO trainer; Naive history stays accessible via trainer.history in-process.
    if dump_history and method == 'dro':
        os.makedirs('results', exist_ok=True)
        history_path = os.path.join(
            'results',
            f"history_{dataset_name}_{attack}_{alpha}_{seed}_{method}.json"
        )
        with open(history_path, 'w') as _hf:
            json.dump(trainer.history, _hf, indent=2)
        result['history_path'] = history_path

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['adult', 'credit', 'lsac'])
    parser.add_argument('--attacks', nargs='+', default=['dp', 'if', 'combined'])
    parser.add_argument('--methods', nargs='+', default=['naive', 'dro'])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--smoke', action='store_true', help='1 seed, 1 dataset, 1 alpha')
    args = parser.parse_args()

    if args.smoke:
        args.datasets = ['adult']
        args.alphas = [0.2]
        args.n_seeds = 1
        smoke_epochs = 10
        smoke_k_inner = 3
        smoke_pgd_steps = 2
        print("SMOKE TEST MODE: 1 dataset, 1 alpha, 1 seed")
        print("Attacks: dp, if, combined | Methods: naive, dro")
        print("Expected: 6 rows\n")
        print("NOTE: smoke uses epochs=10, K_inner=3, pgd_steps=2 for speed")
    else:
        smoke_epochs = 60
        smoke_k_inner = 10   # mandatory per spec
        smoke_pgd_steps = 20  # full attack strength

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs('results', exist_ok=True)
    results_path = 'results/canonical_tau1.json'

    # Load any existing results to support resuming
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {results_path}")
    else:
        all_results = []

    def save_incremental():
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Build set of already-completed keys for resume support
    completed_keys = {
        (r.get('dataset'), r.get('alpha'), r.get('seed'), r.get('attack'), r.get('method'))
        for r in all_results
    }

    total = len(args.datasets) * len(args.alphas) * args.n_seeds * len(args.attacks) * len(args.methods)
    count = 0
    skipped = 0

    for dataset in args.datasets:
        for alpha in args.alphas:
            for seed in range(args.n_seeds):
                for attack in args.attacks:
                    for method in args.methods:
                        count += 1
                        key = (dataset, alpha, seed, attack, method)
                        if key in completed_keys:
                            skipped += 1
                            print(f"[{count}/{total}] SKIP (already done): {dataset} α={alpha} seed={seed} attack={attack} method={method}")
                            continue

                        label = f"[{count}/{total}] {dataset} α={alpha} seed={seed} attack={attack} method={method}"
                        print(label)

                        try:
                            t0 = time.time()
                            result = run_single_experiment(
                                dataset, alpha, seed, attack, method, device=device, verbose=False,
                                epochs=smoke_epochs, k_inner=smoke_k_inner, pgd_steps=smoke_pgd_steps,
                                tau=1.0,  # Use fixed tau=1 for all alphas (artifact bug: stepped tau caused DRO fragility)
                                lambda_init=0.0,
                                radii_mode='uniform',
                                coordinated=False,
                                n_seeds_planned=args.n_seeds
                            )
                            elapsed = time.time() - t0
                            all_results.append(result)
                            completed_keys.add(key)
                            save_incremental()

                            print(f"  → acc={result['acc_clean']:.3f} dp={result['dp_clean']:.4f} "
                                  f"if={result['if_clean']:.4f} ({elapsed:.0f}s)")

                        except Exception as e:
                            print(f"  → FAILED: {e}")
                            import traceback
                            traceback.print_exc()

    print(f"\nSkipped {skipped} already-completed experiments")
    print(f"Saved {len(all_results)} results to {results_path}")

    if args.smoke and len(all_results) > 0:
        print("\nSMOKE TEST JSON:")
        print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()