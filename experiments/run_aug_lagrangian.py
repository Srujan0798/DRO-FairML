#!/usr/bin/env python3
"""DRO-FAIR-AL: augmented-Lagrangian quadratic constraint penalty (pre-registered).

Design + full rationale: docs/superpowers/specs/2026-08-05-augmented-lagrangian-design.md

DIAGNOSIS (established before running): the geometric dual decay
(lr_lambda * 0.95^epoch) caps lambda's total accumulation at ~0.01-0.02 over a
60-epoch run — two orders of magnitude below the 1.5 ceiling — so the linear
lambda*g fairness term is nearly inert. Confirmed by the Arm-A byte-identical
lambda_max 1.5->2.0 no-op (results/fairness_aggressiveness_summary.md): the cap
never binds; the accumulation RATE is the bottleneck. The classical remedy is an
augmented Lagrangian: add (mu/2)*g^2, which supplies a constraint gradient
mu*g*grad(g) from epoch 0 regardless of lambda.

PRE-REGISTERED VALUES (chosen before seeing any result):
  mu in {5.0, 10.0} — at typical g~0.1, penalty gradient mu*g ~ 0.5-1.0, i.e.
  comparable to an effective lambda of 0.5-1.0 (vs the ~0.01 the dual actually
  reaches), while the penalty VALUE (mu/2)g^2 ~ 0.025-0.05 stays well below
  L_tilt (~0.4), so accuracy should not collapse.

SCOPE: {adult, credit} x attack='dp' x alpha in {0.1, 0.2} x seeds 0-5, DRO only.
LSAC excluded (degenerate DP regime, documented separately). Canonical DRO and
Naive reference rows come read-only from canonical_tau1.json (same seeds and
protocol — seed-paired comparison valid, no reruns needed).
  2 mus x 2 datasets x 2 alphas x 6 seeds = 48 configs -> results/aug_lagrangian.json

SUCCESS CRITERION (fixed in the design doc): AL-DRO beats canonical DRO on DP
(one-sided Wilcoxon p<0.05) in >=2 of 4 (dataset, alpha) cells for at least one
mu arm, with mean accuracy cost <= 0.005. Otherwise: honest negative, written up
like the lambda_max/beta/shrinkage negatives. RULE: run once, no post-hoc mu tuning.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ['adult', 'credit']
ALPHAS = [0.1, 0.2]
SEEDS = [0, 1, 2, 3, 4, 5]
MUS = [5.0, 10.0]


def build_configs():
    return [(d, mu, a, s) for mu in MUS for d in DATASETS for a in ALPHAS for s in SEEDS]


def _worker(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    d, mu, a, s = cfg
    return run_single_experiment(
        d, a, s, 'dp', 'dro', device='cpu', epochs=60, k_inner=10,
        pgd_steps=20, tau=1.0, lambda_init=0.0, radii_mode='uniform',
        coordinated=False, n_seeds_planned=6, aug_lagrangian_mu=mu,
    )


if __name__ == '__main__':
    import json, time
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from experiments.run_ablation_parallel import _AblationLock, _assert_safe_results_path, atomic_save

    results_file = 'results/aug_lagrangian.json'
    _assert_safe_results_path(results_file)
    configs = build_configs()
    workers = int(os.environ.get('ABLATION_WORKERS', '12'))

    with _AblationLock():
        rows = json.load(open(results_file)) if os.path.exists(results_file) else []
        have = {(r['dataset'], r['aug_lagrangian_mu'], r['alpha'], r['seed']) for r in rows}
        todo = [c for c in configs if c not in have]
        print(f"[AugLag] {results_file}: have {len(rows)} rows; {len(todo)} missing; workers={workers}", flush=True)
        t0 = time.time()
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                c = futs[fut]
                res = fut.result()
                rows.append(res)
                atomic_save(rows, results_file)
                print(f"[AugLag][{i}/{len(todo)}] {c[0]} mu={c[1]} a={c[2]} s={c[3]} "
                      f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[AugLag] DONE. {len(rows)} rows total.", flush=True)
