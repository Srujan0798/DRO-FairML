#!/usr/bin/env python3
"""
Extended UTKFace experiments — server-ready follow-ups from June 2 meeting.

Three modes:
  --mode alpha_sweep        : extend alpha sweep to {0.3, 0.4} (item #5)
  --mode fairness_pgd       : run FairnessTargetedPGD on UTKFace features (item #6)
  --mode lambda_max_cap     : run UTKFace with capped lambda_max=0.5 (item #2, tests H3)

Run on flair2.iitgn.ac.in:
    cd /data/srujan.sai/DRO-FairML
    venv/bin/python3 experiments/run_utkface_extended.py --mode alpha_sweep \
        --feature_cache /data/srujan.sai/utkface_features.npz --n_seeds 5
    venv/bin/python3 experiments/run_utkface_extended.py --mode fairness_pgd \
        --feature_cache /data/srujan.sai/utkface_features.npz --n_seeds 5
    venv/bin/python3 experiments/run_utkface_extended.py --mode lambda_max_cap \
        --feature_cache /data/srujan.sai/utkface_features.npz --n_seeds 5
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from src.data.datasets import load_utkface
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor, FairnessTargetedPGD
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.temperature import get_temperature


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_utkface_split(feature_cache, seed):
    X, gender, race, _ = load_utkface(data_dir='', feature_cache=feature_cache)
    y = gender.astype(np.float32)
    a = (gender.astype(np.int64))  # binary protected attribute = gender for consistency with v1
    X_tv, X_te, y_tv, y_te, a_tv, a_te = train_test_split(
        X, y, a, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_v, y_tr, y_v, a_tr, a_v = train_test_split(
        X_tv, y_tv, a_tv, test_size=0.1875, random_state=seed, stratify=y_tv)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_v = scaler.transform(X_v).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    return X_tr, y_tr, a_tr, X_v, y_v, a_v, X_te, y_te, a_te


def corrupt(mode, X, y, a, alpha, seed):
    if mode == 'fairness_pgd_dp':
        atk = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=5,
                                  coordinated=True, random_state=seed)
        Xc, yc, ac, _ = atk.corrupt(X, y, a)
        return Xc, yc, ac
    elif mode == 'fairness_pgd_if':
        atk = FairnessTargetedPGD(alpha=alpha, target_metric='if', pgd_steps=5,
                                  coordinated=True, random_state=seed)
        Xc, yc, ac, _ = atk.corrupt(X, y, a)
        return Xc, yc, ac
    elif mode == 'fairness_pgd_combined':
        atk = FairnessTargetedPGD(alpha=alpha, target_metric='combined', pgd_steps=5,
                                  coordinated=True, random_state=seed)
        Xc, yc, ac, _ = atk.corrupt(X, y, a)
        return Xc, yc, ac
    else:
        corr = AdversarialCorruptor(
            alpha=alpha, epsilon=0.1,
            feature_attack=True, label_flip=True, attr_flip=True,
            coordinated=True, random_state=seed)
        Xc, yc, ac, _ = corr.corrupt(X, y, a, model=None, device='cpu')
        return Xc, yc, ac


def run_pair(X_tr_c, y_tr_c, a_tr_c, X_v, y_v, a_v, X_te, y_te, a_te,
             alpha, tau, lambda_max, device, record_history=False):
    input_dim = X_tr_c.shape[1]

    m_n = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    t_n = NaiveFairTrainer(
        m_n, device=device, lr_theta=1e-3, lr_lambda=5e-3, lambda_max=1.5,
        tau=tau, k=5, gamma=0.0, epochs=60, weight_decay=1e-4,
        tau_warmup_epochs=15)
    t_n.fit(X_tr_c, y_tr_c, a_tr_c, X_val=X_v, y_val=y_v, a_val=a_v, verbose=False)
    naive = compute_metrics_torch(t_n.model, X_te, y_te, a_te,
                                  device=device, temperature=tau, k=5, gamma=0.0)

    m_d = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    t_d = DroFairTrainer(
        m_d, alpha=alpha, device=device,
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=lambda_max,
        tau=tau, beta=5.0, k=5, gamma=0.0,
        K_inner=10, epochs=60, weight_decay=1e-4, tau_warmup_epochs=15,
        lambda_warmstart=0.01)
    hist = t_d.fit(X_tr_c, y_tr_c, a_tr_c, X_val=X_v, y_val=y_v, a_val=a_v, verbose=False)
    dro = compute_metrics_torch(t_d.model, X_te, y_te, a_te,
                                device=device, temperature=tau, k=5, gamma=0.0)

    out = {
        'naive': {k: float(v) for k, v in naive.items()},
        'dro':   {k: float(v) for k, v in dro.items()},
    }
    if record_history:
        out['dro_history'] = {
            'lambda_dp': hist['lambda_dp'],
            'lambda_if': hist['lambda_if'],
            'g_dp': hist['g_dp'],
            'g_if': hist['g_if'],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['alpha_sweep', 'fairness_pgd', 'lambda_max_cap'])
    p.add_argument('--feature_cache', default='/data/srujan.sai/utkface_features.npz')
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"mode={args.mode}  device={device}  feature_cache={args.feature_cache}")

    os.makedirs('results', exist_ok=True)
    default_out = {
        'alpha_sweep':     'results/utkface_alpha_sweep.json',
        'fairness_pgd':    'results/utkface_fairness_pgd.json',
        'lambda_max_cap':  'results/utkface_lambda_max_cap.json',
    }
    out_path = args.out or default_out[args.mode]

    runs = []
    seeds = list(range(args.n_seeds))

    if args.mode == 'alpha_sweep':
        for alpha in [0.3, 0.4]:
            tau = get_temperature(alpha)
            for s in seeds:
                set_seed(s)
                print(f"\n[alpha_sweep] alpha={alpha} seed={s}")
                X_tr, y_tr, a_tr, X_v, y_v, a_v, X_te, y_te, a_te = load_utkface_split(
                    args.feature_cache, s)
                Xc, yc, ac = corrupt('adversarial', X_tr, y_tr, a_tr, alpha, s)
                t0 = time.time()
                res = run_pair(Xc, yc, ac, X_v, y_v, a_v, X_te, y_te, a_te,
                               alpha, tau, lambda_max=1.5, device=device)
                res.update({'alpha': alpha, 'seed': s, 'attack': 'adversarial',
                            'elapsed': time.time() - t0})
                runs.append(res)
                print(f"  naive: dp={res['naive']['dp_violation']:.4f} "
                      f"acc={res['naive']['accuracy']:.3f} | "
                      f"dro: dp={res['dro']['dp_violation']:.4f} "
                      f"acc={res['dro']['accuracy']:.3f}  ({res['elapsed']:.0f}s)")
                with open(out_path, 'w') as f:
                    json.dump(runs, f, indent=2)

    elif args.mode == 'fairness_pgd':
        for attack in ['dp', 'if', 'combined']:
            for alpha in [0.1, 0.2, 0.3]:
                tau = get_temperature(alpha)
                for s in seeds:
                    set_seed(s)
                    print(f"\n[fpgd] attack={attack} alpha={alpha} seed={s}")
                    X_tr, y_tr, a_tr, X_v, y_v, a_v, X_te, y_te, a_te = load_utkface_split(
                        args.feature_cache, s)
                    Xc, yc, ac = corrupt(f'fairness_pgd_{attack}', X_tr, y_tr, a_tr, alpha, s)
                    t0 = time.time()
                    res = run_pair(Xc, yc, ac, X_v, y_v, a_v, X_te, y_te, a_te,
                                   alpha, tau, lambda_max=1.5, device=device)
                    res.update({'alpha': alpha, 'seed': s, 'attack': attack,
                                'elapsed': time.time() - t0})
                    runs.append(res)
                    print(f"  naive: dp={res['naive']['dp_violation']:.4f} | "
                          f"dro: dp={res['dro']['dp_violation']:.4f}  ({res['elapsed']:.0f}s)")
                    with open(out_path, 'w') as f:
                        json.dump(runs, f, indent=2)

    elif args.mode == 'lambda_max_cap':
        # Compare lambda_max=1.5 vs 0.5 at alpha in {0.1, 0.2} (where DRO inverts).
        # H3 prediction: capping lambda_max recovers DRO performance.
        for lmax in [1.5, 0.5]:
            for alpha in [0.1, 0.2]:
                tau = get_temperature(alpha)
                for s in seeds:
                    set_seed(s)
                    print(f"\n[lmax_cap] lambda_max={lmax} alpha={alpha} seed={s}")
                    X_tr, y_tr, a_tr, X_v, y_v, a_v, X_te, y_te, a_te = load_utkface_split(
                        args.feature_cache, s)
                    Xc, yc, ac = corrupt('adversarial', X_tr, y_tr, a_tr, alpha, s)
                    t0 = time.time()
                    res = run_pair(Xc, yc, ac, X_v, y_v, a_v, X_te, y_te, a_te,
                                   alpha, tau, lambda_max=lmax, device=device,
                                   record_history=True)
                    res.update({'alpha': alpha, 'seed': s, 'lambda_max': lmax,
                                'attack': 'adversarial', 'elapsed': time.time() - t0})
                    runs.append(res)
                    print(f"  dro: dp={res['dro']['dp_violation']:.4f} "
                          f"acc={res['dro']['accuracy']:.3f}  ({res['elapsed']:.0f}s)")
                    with open(out_path, 'w') as f:
                        json.dump(runs, f, indent=2)

    print(f"\nDone. {len(runs)} runs -> {out_path}")


if __name__ == '__main__':
    main()
