#!/usr/bin/env python3
"""TASK E step 2: mu=0 no-op end-to-end check.

Runs ONE canonical DRO experiment (Adult, alpha=0.2, seed=0, dp attack)
and prints acc_clean / dp_clean / if_clean at full float precision, plus
optional aug_lagrangian_mu. Designed to be run identically at the git
commit immediately before the augmented-Lagrangian introduction and at
HEAD with mu=0.0, so the outputs can be diffed.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from experiments.run_fairness_pgd import run_single_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='adult')
    p.add_argument('--alpha', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--attack', default='dp')
    p.add_argument('--method', default='dro')
    p.add_argument('--mu', type=float, default=0.0)
    args = p.parse_args()

    import inspect
    kwargs = dict(
        device='cpu', verbose=False, epochs=60, k_inner=10, pgd_steps=20,
        tau=1.0, lambda_init=0.0, radii_mode='uniform', coordinated=False,
        n_seeds_planned=6,
    )
    if 'aug_lagrangian_mu' in inspect.signature(run_single_experiment).parameters:
        kwargs['aug_lagrangian_mu'] = args.mu
    result = run_single_experiment(
        args.dataset, args.alpha, args.seed, args.attack, args.method, **kwargs
    )
    out = {
        'acc_clean': result['acc_clean'],
        'dp_clean': result['dp_clean'],
        'if_clean': result['if_clean'],
        'mu': args.mu,
    }
    print(json.dumps(out, indent=2))
    with open('noop_check_output.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
