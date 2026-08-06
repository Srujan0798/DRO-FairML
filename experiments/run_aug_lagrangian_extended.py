#!/usr/bin/env python3
"""TASK A — does the DRO-FAIR-AL improvement generalise, or is it one lucky cell?

Pre-registration (criterion fixed BEFORE this ran):
  docs/superpowers/specs/2026-08-05-al-generalisation-prereg.md

The AL win is currently proven on exactly one cell (Adult, alpha=0.2, mu=5:
DP 0.2334 -> 0.1358, p=0.0156, 6/6 seeds). This tests four things:

  1. FALSIFICATION (run first, matters most): alpha=0.0 on Adult. With no
     corruption, does AL still help as much? AL is a fairness penalty so it will
     reduce DP somewhat -- expected, not a failure. The question is MAGNITUDE.
     Pre-registered rule: robustness framing is supported only if the relative
     DP reduction R(0.0) < 20.9% (= half of the measured R(0.2)=41.8%).
     If R(0.0) >= 20.9%, AL is a GENERIC fairness regulariser and the
     corruption-robustness framing is withdrawn.
  2. Higher corruption: Adult alpha in {0.3, 0.4}.
  3. Other attacks: Adult alpha=0.2, attack in {if, combined}.
  4. Degenerate dataset: LSAC alpha in {0.1, 0.2} -- expected to collapse;
     confirming that IS the success condition.

mu=5 only. DRO only. 42 runs -> results/aug_lagrangian_extended.json (NEW file;
never appends to aug_lagrangian.json, never touches any locked science file).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MU = 5.0

# Ordered so the falsification control finishes first.
BLOCKS = (
    [('adult', 'dp', 0.0, s) for s in range(6)]          # 1. falsification
    + [('adult', 'dp', a, s) for a in (0.3, 0.4) for s in range(6)]   # 2. higher alpha
    + [('adult', atk, 0.2, s) for atk in ('if', 'combined') for s in range(6)]  # 3. attacks
    + [('lsac', 'dp', a, s) for a in (0.1, 0.2) for s in range(6)]    # 4. degenerate ds
)


def build_configs():
    return list(BLOCKS)


def _worker(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    ds, attack, alpha, seed = cfg
    return run_single_experiment(
        ds, alpha, seed, attack, 'dro', device='cpu', epochs=60, k_inner=10,
        pgd_steps=20, tau=1.0, lambda_init=0.0, radii_mode='uniform',
        coordinated=False, n_seeds_planned=6, aug_lagrangian_mu=MU,
    )


if __name__ == '__main__':
    import json, time
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from experiments.run_ablation_parallel import (
        _AblationLock, _assert_safe_results_path, atomic_save)

    results_file = 'results/aug_lagrangian_extended.json'
    _assert_safe_results_path(results_file)
    configs = build_configs()
    workers = int(os.environ.get('ABLATION_WORKERS', '12'))

    with _AblationLock():
        rows = json.load(open(results_file)) if os.path.exists(results_file) else []
        have = {(r['dataset'], r['attack'], r['alpha'], r['seed']) for r in rows}
        todo = [c for c in configs if c not in have]
        print(f"[AugLagExt] have {len(rows)}; {len(todo)} missing; workers={workers}", flush=True)
        t0 = time.time()
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                res = fut.result()
                rows.append(res)
                atomic_save(rows, results_file)
                print(f"[AugLagExt][{i}/{len(todo)}] {c[0]} {c[1]} a={c[2]} s={c[3]} "
                      f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[AugLagExt] DONE. {len(rows)} rows.", flush=True)
