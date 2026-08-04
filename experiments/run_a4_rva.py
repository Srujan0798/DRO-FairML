#!/usr/bin/env python3
"""Agent A4 — Random vs adversarial under canonical protocol (backs the quoted 12–40×).

RandomCorruptor vs FairnessTargetedPGD(dp), canonical config, 3 datasets ×
α∈{0.1, 0.2} × 6 seeds × 2 methods × 2 corruptions = 144 configs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASETS = ["adult", "credit", "lsac"]
ALPHAS = [0.1, 0.2]
SEEDS = range(6)
METHODS = ["naive", "dro"]
ATTACK = "dp"
CORRUPTORS = ['adversarial', 'random']


def build_configs():
    configs = []
    for ds in DATASETS:
        for a in ALPHAS:
            for s in SEEDS:
                for m in METHODS:
                    for c in CORRUPTORS:
                        configs.append((ds, a, s, m, ATTACK, 10, 20, 1.0,
                                        0.0, 5e-3, 'uniform', False, c, 5))
    return configs


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run("results/random_vs_adversarial.json", build_configs(),
        provenance_extras={"ablation": "a4_rva", "n_seeds_planned": 6},
        workers=w, label="A4-RvA")