#!/usr/bin/env python3
"""Agent A1 — kNN ablation attack_k∈{5,15} (Kuldeep's explicit ask #5).

Canonical IF-attack rows use attack_k=5 and K_inner=10 in results/canonical_tau1.json
(do NOT re-run or rewrite that file).

This ablation varies the IF-attack / IF-metric neighbourhood k (attack_k) while
keeping DRO trainer K_inner=10 fixed:
  attack='if', attack_k∈{5,15}, 3 datasets × 5 α × 6 seeds × 2 methods × 2 k = 360
→ results/knn_ablation.json only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASETS = ["adult", "credit", "lsac"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(6)
METHODS = ["naive", "dro"]
ATTACK = "if"
ATTACK_KS = [5, 15]  # neighbourhood k for IF attack + IF metric; K_inner stays 10


def build_configs():
    configs = []
    for ds in DATASETS:
        for a in ALPHAS:
            for s in SEEDS:
                for m in METHODS:
                    for ak in ATTACK_KS:
                        # tuple: ds,a,s,m,attack,k_inner,pgd,tau,λ0,lrλ,radii,coord,corruptor,attack_k
                        configs.append((ds, a, s, m, ATTACK, 10, 20, 1.0,
                                        0.0, 5e-3, 'uniform', False, 'adversarial', ak))
    return configs


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run("results/knn_ablation.json", build_configs(),
        provenance_extras={"ablation": "a1_knn", "n_seeds_planned": 6},
        workers=w, label="A1-kNN")