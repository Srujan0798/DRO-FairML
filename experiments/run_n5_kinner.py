#!/usr/bin/env python3
"""Agent N5 — K_inner ablation {5, 20} (D5, Kuldeep Q10 closed with data).

K_inner∈{5, 20} (10 = canonical, do not re-run), DRO only,
3 datasets × dp × 5 α × 6 seeds × 2 K = 180 configs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASETS = ["adult", "credit", "lsac"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(6)
METHOD = "dro"
ATTACK = "dp"
K_INNERS = [5, 20]


def build_configs():
    configs = []
    for ds in DATASETS:
        for a in ALPHAS:
            for s in SEEDS:
                for k in K_INNERS:
                    configs.append((ds, a, s, METHOD, ATTACK, k, 20, 1.0,
                                    0.0, 5e-3, 'uniform', False, 'adversarial', 5))
    return configs


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run("results/kinner_ablation.json", build_configs(),
        provenance_extras={"ablation": "n5_kinner", "n_seeds_planned": 6},
        workers=w, label="N5-Kinner")