#!/usr/bin/env python3
"""Agent A2 — Tau ablation τ∈{10,100} (Kuldeep's explicit ask #6, clean this time).

τ=1 rows = canonical (do not re-run). New runs: τ∈{10,100}, attack='dp',
3 datasets × 5 alphas × 6 seeds × 2 methods × 2 tau = 360 configs
→ results/tau_ablation.json.

CRITICAL: k_inner=10 everywhere (old ablation confounded τ with k_inner — do not repeat).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASETS = ["adult", "credit", "lsac"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(6)
METHODS = ["naive", "dro"]
ATTACK = "dp"
TAUS = [10.0, 100.0]


def build_configs():
    configs = []
    for ds in DATASETS:
        for a in ALPHAS:
            for s in SEEDS:
                for m in METHODS:
                    for tau in TAUS:
                        configs.append((ds, a, s, m, ATTACK, 10, 20, tau,
                                        0.0, 5e-3, 'uniform', False, 'adversarial', 5))
    return configs


if __name__ == "__main__":
    run("results/tau_ablation.json", build_configs(),
        provenance_extras={"ablation": "a2_tau", "n_seeds_planned": 6},
        workers=4, label="A2-Tau")