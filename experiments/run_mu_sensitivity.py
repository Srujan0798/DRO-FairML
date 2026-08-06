#!/usr/bin/env python3
"""TASK C — mu sensitivity: where is AL SAFE, and where does it destroy the model?

Pre-registration: docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md

Mandatory because TASK A found mu=5 produces a catastrophic failure at
alpha=0.4 on Adult: DP 0.0551 (-80.7%, p=0.0156, the best-looking number in the
study) with accuracy 0.3495 -- below chance. Fairness metric superb, model
destroyed. No mu can be recommended until the safe boundary is mapped as a
function of corruption level.

Also tests whether a SMALLER mu rescues Credit, which collapsed at mu=5 and
mu=10. If yes, AL generalises to a second dataset with a dataset-specific mu;
if no, AL is Adult-only among tested datasets and the paper must say so.

Grid (DRO only, attack=dp, 6 seeds); mu in {5,10} already exist elsewhere:
  adult  alpha in {0.0, 0.2, 0.4} x mu in {0.5, 1, 2, 20}  = 72
  credit alpha = 0.2              x mu in {0.5, 1, 2}      = 18
  -> 90 runs -> results/mu_sensitivity.json (new file, nothing locked touched)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MUS_ADULT = [0.5, 1.0, 2.0, 20.0]
MUS_CREDIT = [0.5, 1.0, 2.0]
ALPHAS_ADULT = [0.0, 0.2, 0.4]
SEEDS = range(6)


def build_configs():
    cfgs = [('adult', a, mu, s) for a in ALPHAS_ADULT for mu in MUS_ADULT for s in SEEDS]
    cfgs += [('credit', 0.2, mu, s) for mu in MUS_CREDIT for s in SEEDS]
    return cfgs


def _worker(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    ds, alpha, mu, seed = cfg
    return run_single_experiment(
        ds, alpha, seed, 'dp', 'dro', device='cpu', epochs=60, k_inner=10,
        pgd_steps=20, tau=1.0, lambda_init=0.0, radii_mode='uniform',
        coordinated=False, n_seeds_planned=6, aug_lagrangian_mu=mu,
    )


if __name__ == '__main__':
    import json, time
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from experiments.run_ablation_parallel import (
        _AblationLock, _assert_safe_results_path, atomic_save)

    results_file = 'results/mu_sensitivity.json'
    _assert_safe_results_path(results_file)
    configs = build_configs()
    workers = int(os.environ.get('ABLATION_WORKERS', '12'))

    with _AblationLock():
        rows = json.load(open(results_file)) if os.path.exists(results_file) else []
        have = {(r['dataset'], r['alpha'], r['aug_lagrangian_mu'], r['seed']) for r in rows}
        todo = [c for c in configs if c not in have]
        print(f"[MuSens] have {len(rows)}; {len(todo)} missing; workers={workers}", flush=True)
        t0 = time.time()
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                res = fut.result()
                rows.append(res)
                atomic_save(rows, results_file)
                print(f"[MuSens][{i}/{len(todo)}] {c[0]} a={c[1]} mu={c[2]} s={c[3]} "
                      f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[MuSens] DONE. {len(rows)} rows.", flush=True)
