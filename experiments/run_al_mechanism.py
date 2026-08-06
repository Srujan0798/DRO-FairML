#!/usr/bin/env python3
"""TASK B — why does AL's accuracy go UP? Test the "suppresses attacked
points" hypothesis directly, by measuring accuracy on the CORRUPTED vs CLEAN
subsets of the training data separately.

Hypothesis (docs/TASKS_AL_VALIDATION.md TASK B): under a DP-targeted attack
the corrupted points are exactly the ones driving the group-rate gap, so
penalising that gap hard suppresses the attack's influence -- i.e. AL acts as
attack-specific denoising, fitting the corrupted points LESS than canonical
DRO does.

This replicates run_single_experiment's setup exactly (same seeding, same
data loading, same corruption call) but captures the corrupt_mask that the
canonical runner discards, and evaluates each trained model on the corrupted
and clean training subsets separately. Read-only with respect to all existing
result files; writes only to results/al_mechanism_summary.md.

Adult, alpha=0.2, seed=0. Three trainers: canonical DRO (mu=0), AL mu=5
(original discovery value), AL mu=20 (TASK C's recommended value).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.dro_fair import DroFairTrainer
from src.temperature import get_temperature

ALPHA = 0.2
SEED = 0
DATASET = 'adult'


def run_one(mu, label):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset(DATASET, random_state=SEED)
    tau = get_temperature(ALPHA)
    input_dim = X_train.shape[1]

    attack_obj = FairnessTargetedPGD(
        alpha=ALPHA, target_metric='dp', pgd_steps=20, epsilon=0.3,
        pgd_step_size=0.02, coordinated=False, random_state=SEED, k=5,
    )
    X_att, y_att, a_att, corrupt_mask = attack_obj.corrupt(X_train, y_train, a_train)
    corrupt_mask = np.asarray(corrupt_mask).astype(bool)
    print(f"  corrupted {corrupt_mask.sum()}/{len(corrupt_mask)} training points", flush=True)

    model = MLPClassifier(input_dim=input_dim)
    trainer = DroFairTrainer(
        model, alpha=ALPHA, device='cpu', epochs=60, K_inner=10, tau=tau,
        lambda_init=0.0, radii_mode='uniform', lr_lambda=5e-3,
        lambda_max=1.5, beta=5.0, aug_lagrangian_mu=mu,
    )
    trainer.fit(X_att, y_att, a_att, X_val, y_val, a_val)

    import torch as T
    model.eval()
    with T.no_grad():
        logits = model(T.tensor(X_att, dtype=T.float32))
        preds = (T.sigmoid(logits).squeeze(-1).numpy() > 0.5).astype(np.float32)
    y_att_np = np.asarray(y_att).astype(np.float32)
    a_att_np = np.asarray(a_att)

    acc_corrupted = float((preds[corrupt_mask] == y_att_np[corrupt_mask]).mean()) if corrupt_mask.sum() else float('nan')
    acc_clean = float((preds[~corrupt_mask] == y_att_np[~corrupt_mask]).mean())

    pos_rate_by_group = {}
    for g in np.unique(a_att_np):
        gm = a_att_np == g
        pos_rate_by_group[int(g)] = float(preds[gm].mean())

    return dict(label=label, mu=mu, acc_on_corrupted_subset=acc_corrupted,
                acc_on_clean_subset=acc_clean, n_corrupted=int(corrupt_mask.sum()),
                n_clean=int((~corrupt_mask).sum()), pos_rate_by_group=pos_rate_by_group)


if __name__ == '__main__':
    results = []
    for mu, label in [(0.0, 'canonical_DRO'), (5.0, 'AL_mu5'), (20.0, 'AL_mu20')]:
        print(f"=== {label} (mu={mu}) ===", flush=True)
        r = run_one(mu, label)
        results.append(r)
        print(f"  acc_on_corrupted={r['acc_on_corrupted_subset']:.4f} "
              f"acc_on_clean={r['acc_on_clean_subset']:.4f} "
              f"pos_rate_by_group={r['pos_rate_by_group']}", flush=True)

    L = ["# TASK B — mechanism: does AL suppress the corrupted points? (Adult, α=0.2, seed 0)",
         "",
         "Hypothesis: AL penalises the DP gap hard enough to suppress the attacked ",
         "points' influence on training, fitting them LESS than canonical DRO does. ",
         "If true, acc_on_corrupted should be LOWER (or grow less) for AL than canonical, ",
         "while acc_on_clean stays comparable or improves.",
         "",
         "| trainer | μ | n_corrupted | n_clean | acc on corrupted subset | acc on clean subset | pos rate by group |",
         "|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['label']} | {r['mu']} | {r['n_corrupted']} | {r['n_clean']} | "
                  f"{r['acc_on_corrupted_subset']:.4f} | {r['acc_on_clean_subset']:.4f} | "
                  f"{r['pos_rate_by_group']} |")

    can, mu5, mu20 = results
    L += ["", "## Verdict"]
    if mu5['acc_on_corrupted_subset'] < can['acc_on_corrupted_subset'] and \
       mu20['acc_on_corrupted_subset'] < can['acc_on_corrupted_subset']:
        L.append(f"**Hypothesis SUPPORTED.** Both AL variants fit the corrupted subset less "
                 f"than canonical DRO (corrupted-subset accuracy: canonical "
                 f"{can['acc_on_corrupted_subset']:.4f} vs μ=5 {mu5['acc_on_corrupted_subset']:.4f} "
                 f"vs μ=20 {mu20['acc_on_corrupted_subset']:.4f}), while clean-subset accuracy is "
                 f"{'held or improved' if mu20['acc_on_clean_subset'] >= can['acc_on_clean_subset'] - 0.01 else 'also affected'} "
                 f"(canonical {can['acc_on_clean_subset']:.4f} vs μ=20 {mu20['acc_on_clean_subset']:.4f}). "
                 f"This is consistent with AL acting as attack-specific denoising: the quadratic "
                 f"penalty forces the model away from the group-rate gap the corrupted points create, "
                 f"which means fitting those specific points less, not a uniform accuracy change.")
    else:
        L.append("**Hypothesis NOT cleanly supported** by this single-seed check — "
                 "corrupted-subset accuracy did not decrease monotonically with μ. "
                 "Reported honestly rather than as a confirmed mechanism; see raw numbers above.")

    out = 'results/al_mechanism_summary.md'
    with open(out, 'w') as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")
