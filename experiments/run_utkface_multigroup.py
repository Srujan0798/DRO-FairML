#!/usr/bin/env python3
"""U2 — Multi-group UTKFace race (5 groups) with binary-trained models.

Trainers (DRO/Naive) still use binary White/non-White for the DP Lagrangian
(as designed for two-group TV radii). Evaluation reports BOTH:
  - dp_binary: White vs non-White (training protected attr)
  - dp_multigroup: max_g P(Ŷ=1|race=g) − min_g (5 race labels 0..4)

This answers: does DRO's clean DP advantage (if any) survive multi-group max-min
evaluation, or concentrate/vanish for small groups?

Output: results/utkface_multigroup.json (new file only).
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import get_dataset, load_utkface
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch, compute_dp_violation

RACE_NAMES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}


def _load_race_raw():
    """Load feature cache and return raw race labels aligned with get_dataset splits."""
    path = Path("data/raw/utkface_features.npz")
    if not path.exists():
        path = Path("data/raw/utkface/utkface_features.npz")
    data = np.load(path)
    return data["race"].astype(np.int64)


def _split_like_dataset(random_state, test_size=0.2, val_size=0.15):
    """Match get_dataset (test_size=0.2, val_size=0.15) + StandardScaler + raw race."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y, a_bin, dname = load_utkface()
    race = _load_race_raw()
    assert len(race) == len(y) == len(a_bin)
    # Mirror get_dataset stratification
    try:
        n_groups = int(a_bin.max() + 1)
        joint = y.astype(int) * n_groups + a_bin.astype(int)
        strat = joint if np.min(np.bincount(joint)) >= 2 else y
    except Exception:
        strat = y
    X_tv, X_te, y_tv, y_te, a_tv, a_te, r_tv, r_te = train_test_split(
        X, y, a_bin, race, test_size=test_size, random_state=random_state, stratify=strat
    )
    try:
        n_groups = int(a_tv.max() + 1)
        joint = y_tv.astype(int) * n_groups + a_tv.astype(int)
        strat_v = joint if np.min(np.bincount(joint)) >= 2 else y_tv
    except Exception:
        strat_v = y_tv
    X_tr, X_va, y_tr, y_va, a_tr, a_va, r_tr, r_va = train_test_split(
        X_tv, y_tv, a_tv, r_tv,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=strat_v,
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)
    return X_tr, y_tr, a_tr, r_tr, X_va, y_va, a_va, r_va, X_te, y_te, a_te, r_te


def _predict_soft(model, X, device, tau):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(xt)
        return torch.sigmoid(logits * tau).cpu().numpy().reshape(-1)


def run_one(alpha, seed, device, tau=1.0, k_inner=10, epochs=60, pgd_steps=20):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_tr, y_tr, a_tr, race_tr, X_va, y_va, a_va, race_va, X_te, y_te, a_te, race_te = (
        _split_like_dataset(seed)
    )
    # attack on binary a (training contract)
    attack = FairnessTargetedPGD(
        alpha=alpha, target_metric="dp", pgd_steps=pgd_steps,
        epsilon=0.3, pgd_step_size=0.02, coordinated=False,
        random_state=seed, k=5,
    )
    X_att, y_att, a_att, _ = attack.corrupt(X_tr, y_tr, a_tr)
    dim = X_tr.shape[1]

    out = {
        "dataset": "utkface",
        "alpha": float(alpha),
        "seed": int(seed),
        "attack": "dp",
        "device": str(device),
        "tau": float(tau),
        "k_inner": int(k_inner),
        "epochs": int(epochs),
        "pgd_steps": int(pgd_steps),
        "n_seeds_planned": 6,
        "data_provenance": "REAL",
        "protected_train": "race_binary_White_vs_nonWhite",
        "protected_eval_multi": "race_5way",
        "race_names": RACE_NAMES,
    }

    for method in ("naive", "dro"):
        model = MLPClassifier(dim, hidden_dims=[128, 64], dropout=0.1)
        if method == "naive":
            trainer = NaiveFairTrainer(
                model, device=device, lr_theta=1e-3, lr_lambda=5e-3, lambda_max=1.5,
                tau=tau, k=5, gamma=0.0, epochs=epochs, weight_decay=1e-4,
                tau_warmup_epochs=15,
            )
        else:
            trainer = DroFairTrainer(
                model, alpha=alpha, device=device, lr_theta=1e-3, lr_lambda=5e-3,
                lr_p=5e-3, lambda_max=1.5, tau=tau, beta=5.0, k=5, gamma=0.0,
                K_inner=k_inner, epochs=epochs, weight_decay=1e-4, tau_warmup_epochs=15,
                lambda_init=0.0, radii_mode="uniform",
            )
        trainer.fit(X_att, y_att, a_att, X_val=X_va, y_val=y_va, a_val=a_va, verbose=False)
        soft = _predict_soft(trainer.model, X_te, device, tau)
        hard = (soft >= 0.5).astype(np.float32)
        m = compute_metrics_torch(
            trainer.model, X_te, y_te, a_te, device=device, temperature=tau, k=5, gamma=0.0
        )
        # per-group rates on 5 race labels
        rates = {}
        for g in range(5):
            mask = race_te == g
            rates[RACE_NAMES[g]] = float(np.mean(soft[mask])) if mask.sum() else float("nan")
        out[method] = {
            "acc": float(m["accuracy"]),
            "dp_binary": float(m["dp_violation"]),
            "dp_multigroup": float(compute_dp_violation(soft, race_te)),
            "if_cosine": float(m["if_violation"]),
            "group_pos_rates": rates,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--k_inner", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--pgd_steps", type=int, default=20)
    ap.add_argument("--output", default="results/utkface_multigroup.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = []
    if os.path.exists(args.output):
        rows = json.load(open(args.output))
    have = {(r["alpha"], r["seed"]) for r in rows}

    for a in args.alphas:
        for s in range(args.n_seeds):
            if (a, s) in have:
                print(f"skip a={a} s={s}")
                continue
            print(f"RUN a={a} s={s}", flush=True)
            t0 = time.time()
            r = run_one(a, s, args.device, args.tau, args.k_inner, args.epochs, args.pgd_steps)
            r["total_time"] = time.time() - t0
            rows.append(r)
            with open(args.output, "w") as f:
                json.dump(rows, f, indent=2)
            print(
                f"  done {r['total_time']:.0f}s "
                f"naive_dp_multi={r['naive']['dp_multigroup']:.4f} "
                f"dro_dp_multi={r['dro']['dp_multigroup']:.4f}",
                flush=True,
            )
    print(f"COMPLETE {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
