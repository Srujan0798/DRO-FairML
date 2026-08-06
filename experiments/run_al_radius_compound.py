#!/usr/bin/env python3
"""TASK C2 — does the AL fix (mu=20) compound with the radius fix (radii_scale=2.0)?

Pre-registration (criterion fixed BEFORE this ran):
  docs/superpowers/specs/2026-08-07-al-radius-compound-prereg.md

Design: Adult, attack=dp, DRO only, alpha in {0.2, 0.3}, 6 seeds (0-5),
2x2: radii_scale in {1.0, 2.0} x aug_lagrangian_mu in {0, 20}.

Arms that ALREADY exist (reused read-only, never re-run):
  canonical  (r=1.0, mu=0)   -> results/canonical_tau1.json
  AL-only    (r=1.0, mu=20)  -> alpha=0.2 in results/mu_sensitivity.json
  radius-only (r=2.0, mu=0)  -> results/radius_sensitivity.json (N1 ablation)
This driver therefore runs ONLY the missing cells into a NEW file:
  6  AL-only alpha=0.3 (mu=20, r=1.0)
  12 combined  (r=2.0, mu=20) at alpha in {0.2, 0.3}
= 18 runs -> results/al_radius_compound.json (new file; nothing locked touched).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MU = 20.0
RADII = 2.0

# (alpha, seed) for the cells this driver must produce.
MISSING = (
    # AL-only at alpha=0.3, radii=1.0 (mu_sensitivity covered alpha 0.2 only)
    [(0.3, s) for s in range(6)]
    # combined: radii=2.0, mu=20 at both alphas
    + [(a, s) for a in (0.2, 0.3) for s in range(6)]
)


def build_configs():
    """Return list of (alpha, seed, radii_scale, mu) to run."""
    out = []
    # AL-only alpha=0.3: radii=1.0, mu=20 (first 6 MISSING entries)
    for alpha, seed in MISSING[:6]:
        out.append((alpha, seed, 1.0, MU))
    # combined: radii=2.0, mu=20 (remaining 12 MISSING entries)
    for alpha, seed in MISSING[6:]:
        out.append((alpha, seed, RADII, MU))
    return out


def _worker(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    alpha, seed, radii_scale, mu = cfg
    return run_single_experiment(
        'adult', alpha, seed, 'dp', 'dro', device='cpu', epochs=60, k_inner=10,
        pgd_steps=20, tau=1.0, lambda_init=0.0, radii_mode='uniform',
        coordinated=False, n_seeds_planned=6,
        radii_scale=radii_scale, aug_lagrangian_mu=mu,
    )


if __name__ == '__main__':
    import json, time
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from experiments.run_ablation_parallel import (
        _AblationLock, _assert_safe_results_path, atomic_save)

    results_file = 'results/al_radius_compound.json'
    _assert_safe_results_path(results_file)
    configs = build_configs()
    workers = int(os.environ.get('ABLATION_WORKERS', '12'))

    with _AblationLock():
        rows = json.load(open(results_file)) if os.path.exists(results_file) else []
        have = {(r['alpha'], r['seed'], r['radii_scale'], r['aug_lagrangian_mu']) for r in rows}
        todo = [c for c in configs if tuple(c) not in have]
        print(f"[ALRadiusCompound] have {len(rows)}; {len(todo)} missing; workers={workers}", flush=True)
        t0 = time.time()
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                res = fut.result()
                rows.append(res)
                atomic_save(rows, results_file)
                print(f"[ALRadiusCompound][{i}/{len(todo)}] alpha={c[0]} s={c[1]} "
                      f"r={c[2]} mu={c[3]} acc={res['acc_clean']:.3f} "
                      f"dp={res['dp_clean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[ALRadiusCompound] DONE. {len(rows)} rows.", flush=True)
