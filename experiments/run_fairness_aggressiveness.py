#!/usr/bin/env python3
"""lambda_max / beta ablation: can DRO's margin over Naive be made bigger, not just
present? (Manisha's explicit ask: "the current DRO-FAIR is not very much better than
Naive, just better -- any improvement so the approach and everything gets improved.")

HONEST PREMISE, established before running: lambda_max=1.5 (vs the paper's stated 2.0)
and beta=5.0 (tilted-risk sharpness, never ablated at all) were both set once and never
revisited after the tau=100 -> tau=1 fix. lambda_max was conservatively capped to
survive the OLD tau=100 lambda-runaway instability (see docs/KEY_FORMULAS.md /
paper discussion.tex "The Temperature Effect"). That instability source is gone under
the locked tau=1 protocol. This tests whether the conservative cap is still needed, or
whether it can be raised to let DRO fight harder for fairness -- producing a genuinely
larger DP margin over Naive rather than a marginal one.

PRINCIPLED VALUES (chosen before seeing any result):
  lambda_max in {1.5 (canonical), 2.0 (the paper's own originally-stated value)}
  beta       in {5.0 (canonical), 10.0 (2x sharper worst-case tilting)}
Varied ONE AT A TIME (not a full cross-product) so the story stays interpretable:
does raising the cap alone help? does sharper tilting alone help?

SCOPE: Adult only (most-examined dataset, most headroom), attack='dp',
alpha in {0.1, 0.2} (the defensible regime where the current margin is smallest and
where a real improvement would matter most for the headline claim), 3 seeds, DRO only
(Naive's lambda_max stays hardcoded canonical -- this tests DRO's own aggressiveness,
not a baseline change). 1.5/5.0 (canonical) rows are pulled read-only from
canonical_tau1.json, not re-run.
  Arm A (lambda_max=2.0, beta=5.0):  2 alphas x 3 seeds = 6 configs
  Arm B (lambda_max=1.5, beta=10.0): 2 alphas x 3 seeds = 6 configs
TOTAL: 12 configs -> results/fairness_aggressiveness.json

RULE: hypothesis testing, not tuning-until-it-wins. Run once, report the actual delta
vs the canonical margin honestly, whether it grows, shrinks, or costs accuracy.
Uses the shared lock so it queues behind whatever else is running.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALPHAS = [0.1, 0.2]
SEEDS = [0, 1, 2]
ARMS = [
    (2.0, 5.0),   # Arm A: raise lambda_max only
    (1.5, 10.0),  # Arm B: sharpen beta only
]


def build_configs():
    configs = []
    for lam, b in ARMS:
        for a in ALPHAS:
            for s in SEEDS:
                configs.append((lam, b, a, s))
    return configs


def _worker(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    lam, b, a, s = cfg
    return run_single_experiment(
        'adult', a, s, 'dp', 'dro', device='cpu', epochs=60, k_inner=10,
        pgd_steps=20, tau=1.0, lambda_init=0.0, radii_mode='uniform',
        coordinated=False, n_seeds_planned=3, lambda_max=lam, beta=b,
    )


if __name__ == '__main__':
    import json, time
    from experiments.run_ablation_parallel import _AblationLock, _assert_safe_results_path, atomic_save

    results_file = 'results/fairness_aggressiveness.json'
    _assert_safe_results_path(results_file)
    configs = build_configs()

    with _AblationLock():
        rows = json.load(open(results_file)) if os.path.exists(results_file) else []
        have = {(r['lambda_max'], r['beta'], r['alpha'], r['seed']) for r in rows}
        todo = [c for c in configs if c not in have]
        print(f"[FairAggr] {results_file}: have {len(rows)} rows; {len(todo)} missing", flush=True)
        t0 = time.time()
        for i, c in enumerate(todo, 1):
            res = _worker(c)
            rows.append(res)
            atomic_save(rows, results_file)
            print(f"[FairAggr][{i}/{len(todo)}] lmax={c[0]} beta={c[1]} a={c[2]} s={c[3]} "
                  f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[FairAggr] DONE. {len(rows)} rows total.", flush=True)
