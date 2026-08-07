#!/usr/bin/env python3
"""TASK E step 6: is the mu=20 predicted-positive-rate collapse seed-0-specific?

Follows the exact training pattern of experiments/run_al_mechanism.py (which
replicates run_single_experiment's setup: same seeding, data loading, attack)
but for seeds 1, 2, 3 at mu=0 and mu=20, and reports the per-group
predicted-positive rate plus test-set accuracy. Seed 0 is included for
consistency with the TASK B caveat. Writes results/al_mechanism_seed_scan.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
import numpy as np
import torch

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.dro_fair import DroFairTrainer

ALPHA = 0.2
DATASET = 'adult'
SEEDS = [0, 1, 2, 3]
MUS = [0.0, 20.0]


def run_one(seed, mu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset(DATASET, random_state=seed)

    attack_obj = FairnessTargetedPGD(
        alpha=ALPHA, target_metric='dp', pgd_steps=20, epsilon=0.3,
        pgd_step_size=0.02, coordinated=False, random_state=seed, k=5,
    )
    X_att, y_att, a_att, _ = attack_obj.corrupt(X_train, y_train, a_train)

    model = MLPClassifier(input_dim=X_train.shape[1])
    trainer = DroFairTrainer(
        model, alpha=ALPHA, device='cpu', epochs=60, K_inner=10, tau=1.0,
        lambda_init=0.0, radii_mode='uniform', lr_lambda=5e-3,
        lambda_max=1.5, beta=5.0, aug_lagrangian_mu=mu,
    )
    trainer.fit(X_att, y_att, a_att, X_val, y_val, a_val)

    model.eval()
    with torch.no_grad():
        # predicted-positive rate on ATTACKED TRAINING data (per group), like run_al_mechanism
        logits_tr = model(torch.tensor(X_att, dtype=torch.float32))
        preds_tr = (torch.sigmoid(logits_tr).squeeze(-1).numpy() > 0.5).astype(np.float32)
        # test-set metrics
        logits_te = model(torch.tensor(X_test, dtype=torch.float32))
        probs_te = torch.sigmoid(logits_te).squeeze(-1).numpy()
        preds_te = (probs_te > 0.5).astype(np.float32)
        y_te = np.asarray(y_test)
        a_te = np.asarray(a_test)

    a_att_np = np.asarray(a_att)
    pos_rate_train = {}
    for g in np.unique(a_att_np):
        gm = a_att_np == g
        pos_rate_train[int(g)] = float(preds_tr[gm].mean())
    pos_rate_test = {}
    for g in np.unique(a_te):
        gm = a_te == g
        pos_rate_test[int(g)] = float(preds_te[gm].mean())

    from src.evaluation.metrics import compute_dp_violation
    return dict(
        seed=seed, mu=mu,
        pos_rate_by_group_train=pos_rate_train,
        pos_rate_by_group_test=pos_rate_test,
        acc_test=float((preds_te == y_te).mean()),
        dp_test=float(compute_dp_violation(probs_te, a_te)),
    )


if __name__ == '__main__':
    results = []
    for seed in SEEDS:
        for mu in MUS:
            print(f"=== seed {seed} mu={mu} ===", flush=True)
            r = run_one(seed, mu)
            results.append(r)
            print(json.dumps(r), flush=True)
    with open('results/al_mechanism_seed_scan.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("wrote results/al_mechanism_seed_scan.json")
