#!/usr/bin/env python3
"""TASK F — canonical reproducibility re-run.

Re-runs the full canonical grid (540 rows) with CURRENT code (post-k-NN-cosine-fix)
into a NEW file (never overwrites canonical_tau1.json), then diffs every row
against the original. Purpose: close the reproducibility gap documented in
docs/MEMO_FOR_ADVISOR.md §6(b) — the IF column of DP/COMBINED rows was
floating-point noise (~1e-11) under the old Euclidean training graph.

Expected from AL-run calibration (48 rows in ~30 min at 12 workers): ~5-6 hours
for 540 rows at 12 workers (DRO configs ~30 min each, naive ~15s).

Diff after completion: accuracy should reproduce EXACTLY, DP shift ~1e-7 (no
conclusion moves), IF goes from noise to a real value (~0.045).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_ablation_parallel import run, atomic_save

DATASETS = ["adult", "credit", "lsac"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(6)
METHODS = ["naive", "dro"]
ATTACKS = ["dp", "if", "combined"]


def build_configs():
    configs = []
    for ds in DATASETS:
        for a in ALPHAS:
            for s in SEEDS:
                for atk in ATTACKS:
                    for m in METHODS:
                        # 14-tuple: the shared driver defaults the rest
                        configs.append((ds, a, s, m, atk, 10, 20, 1.0,
                                        0.0, 5e-3, 'uniform', False, 'adversarial', 5))
    return configs


def main():
    configs = build_configs()
    # Single writer via the shared driver (uses the lock, resume-safe)
    run("results/canonical_tau1_cosine.json", configs,
        provenance_extras={"provenance": "TASK_F_reproducibility_rerun",
                           "code_version": "post_cosine_fix",
                           "n_seeds_planned": 6},
        workers=int(os.environ.get("ABLATION_WORKERS", 12)),
        label="TASK-F-Repro")


if __name__ == "__main__":
    main()
