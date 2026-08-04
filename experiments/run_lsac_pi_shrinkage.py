#!/usr/bin/env python3
"""LSAC degeneracy: pi_shrinkage_k hypothesis test (alternative to L2's radii_clamp).

HONEST FRAMING, established before running: LSAC's validation set is n_val=2804 with
minority fraction ~6.4% (measured). That is a LARGE, RELIABLE sample -- pi_clean is not
noisily estimated here. So pi_shrinkage_k is NOT "correcting estimation noise" (its usual
statistical justification) -- it is a deliberate regularizer that intentionally biases
pi_clean toward balance to control the resulting radius, conceptually adjacent to L2's
radii_clamp rather than a distinct noise-correction mechanism. Testing it anyway because
it reshapes the cap differently (continuous, alpha-dependent via the rho formula) than a
flat post-hoc clamp, and L2's own report says no clamp arm fixed LSAC -- this is a
different-shaped attempt at the same underlying problem, not a resubmission of the same idea.

PRINCIPLED k VALUES (chosen from n_val=2804, minority=0.064, BEFORE seeing any result):
  k=700  -> shifts pi_clean[minority] from 0.064 to ~0.15  (moderate pull)
  k=2100 -> shifts pi_clean[minority] from 0.064 to ~0.25  (strong pull, order n_val itself)

ARMS (LSAC only, attack='dp', DRO only -- naive does not use radii so shrinkage cannot
affect it; the canonical naive/DRO-unshrunk reference is pulled read-only from
canonical_tau1.json, not re-run here):
  5 alphas x 3 seeds x 2 k values = 30 configs -> results/lsac_pi_shrinkage.json
Seeds 0-2 only for this quick hypothesis check (not the full 6) -- extend to 6 if the
3-seed result looks promising enough to be worth the extra compute.

RULE: hypothesis testing, not tuning-until-it-wins. Run once, report honestly either way.
Uses the shared lock (experiments.run_ablation_parallel) so it queues behind whatever
else is running rather than competing for CPU.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_ablation_parallel import run

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = [0, 1, 2]
K_VALUES = [700.0, 2100.0]


def build_configs():
    # 16-tuple schema: dataset, alpha, seed, method, attack, k_inner, pgd_steps, tau,
    # lambda_init, lr_lambda, radii_mode, coordinated, corruptor_type, attack_k,
    # radii_scale, radii_clamp  -- pi_shrinkage_k is NOT in this tuple (added after
    # radii_clamp support was built), so we pass it via provenance_extras + a thin
    # wrapper below instead of extending the shared 16-tuple schema.
    configs = []
    for k in K_VALUES:
        for a in ALPHAS:
            for s in SEEDS:
                configs.append((k, a, s))
    return configs


def _worker(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    k, a, s = cfg
    return run_single_experiment(
        'lsac', a, s, 'dp', 'dro', device='cpu', epochs=60, k_inner=10,
        pgd_steps=20, tau=1.0, lambda_init=0.0, radii_mode='uniform',
        coordinated=False, n_seeds_planned=3, pi_shrinkage_k=k,
    )


if __name__ == '__main__':
    import json, time, tempfile
    from experiments.run_ablation_parallel import _AblationLock, _assert_safe_results_path, atomic_save

    results_file = 'results/lsac_pi_shrinkage.json'
    _assert_safe_results_path(results_file)
    configs = build_configs()

    with _AblationLock():
        rows = json.load(open(results_file)) if os.path.exists(results_file) else []
        have = {(r['pi_shrinkage_k'], r['alpha'], r['seed']) for r in rows}
        todo = [c for c in configs if c not in have]
        print(f"[LSAC-shrinkage] {results_file}: have {len(rows)} rows; {len(todo)} missing", flush=True)
        t0 = time.time()
        for i, c in enumerate(todo, 1):
            res = _worker(c)
            rows.append(res)
            atomic_save(rows, results_file)
            print(f"[LSAC-shrinkage][{i}/{len(todo)}] k={c[0]} a={c[1]} s={c[2]} "
                  f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[LSAC-shrinkage] DONE. {len(rows)} rows total.", flush=True)
