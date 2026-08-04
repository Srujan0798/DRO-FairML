#!/usr/bin/env python3
"""Agent N3 — COMPAS + German Credit extended datasets (Manisha May 19: "Adult etc").

A fairness paper with no COMPAS is conspicuous to any reviewer. This driver runs
the FULL canonical protocol on the two new tabular datasets so the main results
table can span FIVE tabular datasets (+ UTKFace) and an honest replication
verdict can be made: does the Adult/Credit pattern (DRO better on DP at α≤0.2)
REPLICATE on COMPAS and German?

Loaders (src/data/datasets.py::load_compas / load_german) and tests
(tests/test_datasets_extended.py) already land green. This driver only runs the
360-config canonical grid — it does NOT touch the loaders.

Canonical config (identical to results/canonical_tau1.json for Adult/Credit/LSAC):
  tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0,
  radii_mode='uniform', coordinated=False, corruptor_type='adversarial',
  attack_k=5, lr_lambda=5e-3.

Grid: 2 datasets × 3 attacks × 5 α × 6 seeds × 2 methods = 360 configs
  → results/extended_datasets.json  (NEW file, NEVER canonical_tau1.json).

The shared driver (experiments/run_ablation_parallel.py) refuses to write
locked science, is resume-safe (missing-key enumeration), and stamps full
provenance on every row in the PARENT (not the worker).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

DATASETS = ["compas", "german"]
ATTACKS = ["dp", "if", "combined"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(6)
METHODS = ["naive", "dro"]

# Canonical constants (must match results/canonical_tau1.json exactly).
K_INNER = 10
PGD_STEPS = 20
TAU = 1.0
LAMBDA_INIT = 0.0
LR_LAMBDA = 5e-3
RADII_MODE = 'uniform'
COORDINATED = False
CORRUPTOR_TYPE = 'adversarial'
ATTACK_K = 5


def build_configs():
    """2 ds × 3 attacks × 5 α × 6 seeds × 2 methods = 360 configs."""
    configs = []
    for ds in DATASETS:
        for attack in ATTACKS:
            for a in ALPHAS:
                for s in SEEDS:
                    for m in METHODS:
                        # 14-tuple schema (Wave 1 drivers): ds,a,s,m,attack,
                        # k_inner,pgd,tau,λ0,lrλ,radii,coord,corruptor,attack_k
                        configs.append((ds, a, s, m, attack, K_INNER, PGD_STEPS,
                                        TAU, LAMBDA_INIT, LR_LAMBDA, RADII_MODE,
                                        COORDINATED, CORRUPTOR_TYPE, ATTACK_K))
    return configs


def _expected_total():
    return len(DATASETS) * len(ATTACKS) * len(ALPHAS) * len(SEEDS) * len(METHODS)


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    configs = build_configs()
    assert len(configs) == 360, f"expected 360 configs, got {len(configs)}"
    run("results/extended_datasets.json", configs,
        provenance_extras={
            "ablation": "n3_extended_datasets",
            "n_seeds_planned": len(SEEDS),
            "protocol": "canonical_tau1",
        },
        workers=w, label="N3-Extended")
    print(f"[N3-Extended] expected {360} configs; built {len(configs)}", flush=True)