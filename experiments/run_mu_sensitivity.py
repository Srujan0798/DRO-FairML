#!/usr/bin/env python3
"""TASK C — mu sensitivity: build the 6-point curve and find the safe maximum.

Pre-registration: docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md

New grid (this file): Adult + Credit, alpha=0.2, 6 seeds, 2 methods (naive + dro),
  mu in {0.5, 1.0, 2.0, 20.0}  -> 2*1*6*2*4 = 96 configs.
  Output: results/mu_sensitivity.json (existing DRO rows are reused; naive rows are
  added on top; never overwrite aug_lagrangian.json).

Existing data reused (not re-run here, merged at summary time):
  results/aug_lagrangian.json -> mu in {5.0, 10.0} for both datasets at alpha=0.2.

Combined 6-point curve: mu in {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}.

Uses the shared driver experiments/run_ablation_parallel.run() with the machine-wide
_AblationLock so the 14-core box is never oversubscribed. ABLATION_WORKERS=12.
Config tuple (17-element): (dataset, alpha, seed, method, attack, k_inner, pgd_steps,
  tau, lambda_init, lr_lambda, radii_mode, coordinated, corruptor_type, attack_k,
  radii_scale, radii_clamp, aug_lagrangian_mu).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ['adult', 'credit']
ALPHAS = [0.2]
SEEDS = [0, 1, 2, 3, 4, 5]
METHODS = ['naive', 'dro']
MUS = [0.5, 1.0, 2.0, 20.0]

# Canonical shared settings for every config.
ATTACK = 'k_inner', 'pgd_steps', 'tau', 'lambda_init', 'lr_lambda', 'radii_mode', 'coordinated', 'corruptor_type', 'attack_k'
K_INNER = 10
PGD_STEPS = 20
TAU = 1.0
LAMBDA_INIT = 0.0
LR_LAMBDA = 0.005
RADII_MODE = 'uniform'
COORDINATED = False
CORRUPTOR_TYPE = 'adversarial'
ATTACK_K = 5
RADII_SCALE = 1.0
RADII_CLAMP = None


def build_configs():
    cfgs = []
    for ds in DATASETS:
        for a in ALPHAS:
            for s in SEEDS:
                for m in METHODS:
                    for mu in MUS:
                        cfgs.append((ds, a, s, m, 'dp', K_INNER, PGD_STEPS, TAU,
                                     LAMBDA_INIT, LR_LAMBDA, RADII_MODE, COORDINATED,
                                     CORRUPTOR_TYPE, ATTACK_K, RADII_SCALE, RADII_CLAMP, mu))
    return cfgs


if __name__ == '__main__':
    from experiments.run_ablation_parallel import run, _assert_safe_results_path

    results_file = 'results/mu_sensitivity.json'
    _assert_safe_results_path(results_file)
    configs = build_configs()
    workers = int(os.environ.get('ABLATION_WORKERS', '12'))
    run(results_file, configs, provenance_extras=None, workers=workers, label='MuSens')
