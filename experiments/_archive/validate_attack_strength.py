"""
Validation: FairnessTargetedPGD must produce DP > 3× random noise.

Madam's requirement: "by attacking DP, you are increasing DP significantly
more than the random noise."

We train Naive-FAIR on:
  (a) clean data
  (b) RandomCorruptor (random noise baseline)
  (c) FairnessTargetedPGD with real gradient-based feature attack

Expected: DP(c) / DP(b) > 3.0
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.training.naive_fair import NaiveFairTrainer
from src.corruption.adversarial import FairnessTargetedPGD, RandomCorruptor
from src.evaluation.metrics import compute_metrics_torch

def train_naive_and_eval(X_tr, y_tr, a_tr, X_te, y_te, a_te, seed=0):
    """Train Naive-FAIR and return test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPClassifier(X_tr.shape[1], hidden_dims=[128, 64], dropout=0.1)
    trainer = NaiveFairTrainer(
        model, device='cpu', lr_theta=1e-3, lambda_max=1.0,
        epochs=60, weight_decay=1e-4, tau=100.0
    )
    trainer.fit(X_tr, y_tr, a_tr, verbose=False)
    metrics = compute_metrics_torch(trainer.model, X_te, y_te, a_te, device='cpu', temperature=100.0)
    return metrics['accuracy'], metrics['dp_violation'], metrics['if_violation']

print("=" * 70)
print("ATTACK STRENGTH VALIDATION")
print("=" * 70)

alpha = 0.2
print(f"\nDataset: Adult  |  alpha={alpha}")

results = {'clean': {}, 'random': {}, 'adversarial': {}}

for seed in [0, 1, 2]:
    print(f"\n--- Seed {seed} ---")
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset('adult', random_state=seed)

    # (a) Clean
    acc, dp, if_v = train_naive_and_eval(X_train, y_train, a_train, X_test, y_test, a_test, seed)
    results['clean'][seed] = {'acc': acc, 'dp': dp, 'if': if_v}
    print(f"  Clean:       acc={acc:.3f}  dp={dp:.4f}")

    # (b) Random corruption
    rand = RandomCorruptor(alpha=alpha, random_state=seed)
    X_rand, y_rand, a_rand, _ = rand.corrupt(X_train, y_train, a_train)
    acc, dp, if_v = train_naive_and_eval(X_rand, y_rand, a_rand, X_test, y_test, a_test, seed)
    results['random'][seed] = {'acc': acc, 'dp': dp, 'if': if_v}
    print(f"  Random:      acc={acc:.3f}  dp={dp:.4f}")

    # (c) Adversarial (NEW FairnessTargetedPGD with surrogate-based PGD)
    atk = FairnessTargetedPGD(
        alpha=alpha, target_metric='dp', pgd_steps=20,
        epsilon=0.3, pgd_step_size=0.02,
        coordinated=False, random_state=seed
    )
    X_adv, y_adv, a_adv, _ = atk.corrupt(X_train, y_train, a_train)
    acc, dp, if_v = train_naive_and_eval(X_adv, y_adv, a_adv, X_test, y_test, a_test, seed)
    results['adversarial'][seed] = {'acc': acc, 'dp': dp, 'if': if_v}
    print(f"  Adversarial: acc={acc:.3f}  dp={dp:.4f}")

    ratio = results['adversarial'][seed]['dp'] / (results['random'][seed]['dp'] + 1e-12)
    print(f"  Ratio (adv/random): {ratio:.2f}x")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for condition in ['clean', 'random', 'adversarial']:
    dps = [results[condition][s]['dp'] for s in [0, 1, 2]]
    accs = [results[condition][s]['acc'] for s in [0, 1, 2]]
    print(f"{condition:<12}: dp={np.mean(dps):.4f}±{np.std(dps):.4f}  acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

ratios = [results['adversarial'][s]['dp'] / (results['random'][s]['dp'] + 1e-12) for s in [0, 1, 2]]
mean_ratio = np.mean(ratios)
print(f"\nMean Ratio (adv/random): {mean_ratio:.2f}x")

if mean_ratio > 3.0:
    print("\n✅ PASS: Adversarial attack produces >3× more DP than random noise!")
else:
    print(f"\n❌ FAIL: Ratio {mean_ratio:.2f}x < 3.0. Attack needs more strength.")
