#!/usr/bin/env python3
"""Agent A5 — Empirical radii Q5 (Kuldeep's ask #7).

Adult only: radii_mode arms × attack='dp' × 5 alphas × 6 seeds × 2 methods × 3 arms = 180 configs.
Arms: (uniform,uncoordinated)=canonical reference, (uniform,coordinated), (empirical,coordinated).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASET = "adult"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(6)
METHODS = ["naive", "dro"]
ATTACK = "dp"
ARMS = [('uniform', False), ('uniform', True), ('empirical', True)]


def build_configs():
    configs = []
    for a in ALPHAS:
        for s in SEEDS:
            for m in METHODS:
                for radii_mode, coord in ARMS:
                    configs.append((DATASET, a, s, m, ATTACK, 10, 20, 1.0,
                                    0.0, 5e-3, radii_mode, coord, 'adversarial', 5))
    return configs


if __name__ == "__main__":
    run("results/empirical_radii.json", build_configs(),
        provenance_extras={"ablation": "a5_empirical", "n_seeds_planned": 6},
        workers=4, label="A5-EmpR")