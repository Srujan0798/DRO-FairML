"""Quick check: does FairnessTargetedPGD DP attack actually produce high DP on corrupted data?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.datasets import load_adult, load_credit, load_lsac
from src.corruption.adversarial import FairnessTargetedPGD

def compute_dp(y, a):
    m0 = a == 0
    m1 = a == 1
    p0 = np.mean(y[m0]) if m0.sum() > 0 else 0
    p1 = np.mean(y[m1]) if m1.sum() > 0 else 0
    return abs(p0 - p1)

# Test on LSAC (smallest dataset)
print("=== LSAC ===")
X, y, a, _ = load_lsac()
print(f"  Clean: n={len(y)}, groups={np.bincount(a.astype(int))}, dp={compute_dp(y, a):.4f}")

for alpha in [0.1, 0.2, 0.3]:
    atk = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=1, coordinated=True, random_state=0)
    X_c, y_c, a_c, mask = atk.corrupt(X, y, a)
    dp_c = compute_dp(y_c, a_c)
    n_flip = mask.sum()
    print(f"  α={alpha:.1f}: n_flipped={n_flip}, dp_corrupted={dp_c:.4f}")

# Test on Credit (medium)
print("\n=== CREDIT ===")
X, y, a, _ = load_credit()
print(f"  Clean: n={len(y)}, groups={np.bincount(a.astype(int))}, dp={compute_dp(y, a):.4f}")

for alpha in [0.1, 0.2, 0.3]:
    atk = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=1, coordinated=True, random_state=0)
    X_c, y_c, a_c, mask = atk.corrupt(X, y, a)
    dp_c = compute_dp(y_c, a_c)
    n_flip = mask.sum()
    print(f"  α={alpha:.1f}: n_flipped={n_flip}, dp_corrupted={dp_c:.4f}")

# Test on Adult (largest) — only alpha=0.1
print("\n=== ADULT ===")
X, y, a, _ = load_adult()
print(f"  Clean: n={len(y)}, groups={np.bincount(a.astype(int))}, dp={compute_dp(y, a):.4f}")

for alpha in [0.1]:
    atk = FairnessTargetedPGD(alpha=alpha, target_metric='dp', pgd_steps=1, coordinated=True, random_state=0)
    X_c, y_c, a_c, mask = atk.corrupt(X, y, a)
    dp_c = compute_dp(y_c, a_c)
    n_flip = mask.sum()
    print(f"  α={alpha:.1f}: n_flipped={n_flip}, dp_corrupted={dp_c:.4f}")
