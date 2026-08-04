#!/usr/bin/env python3
"""Agent A3 — Lambda/LR scoped grid (Kuldeep's explicit ask #4).

Pathology-aware scope (EXCLUDE λ0=1.0 that ran 17.9h): λ_init∈{0.0, 0.01, 0.1} ×
lr_λ∈{0.001, 0.005} on Adult, attack='dp', α∈{0.2, 0.3}, 6 seeds, DRO only = 72 configs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASET = "adult"
ALPHAS = [0.2, 0.3]
SEEDS = range(6)
METHOD = "dro"
ATTACK = "dp"
LAMBDA_INITS = [0.0, 0.01, 0.1]
LR_LAMBDAS = [0.001, 0.005]


def build_configs():
    configs = []
    for a in ALPHAS:
        for s in SEEDS:
            for li in LAMBDA_INITS:
                for lrl in LR_LAMBDAS:
                    configs.append((DATASET, a, s, METHOD, ATTACK, 10, 20, 1.0,
                                    li, lrl, 'uniform', False, 'adversarial', 5))
    return configs


if __name__ == "__main__":
    run("results/lambda_grid.json", build_configs(),
        provenance_extras={"ablation": "a3_lambda_grid", "n_seeds_planned": 6},
        workers=4, label="A3-Lambda")