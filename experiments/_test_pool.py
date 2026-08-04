#!/usr/bin/env python3
"""Minimal ProcessPool smoke for macOS spawn."""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _job(seed):
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    r = run_single_experiment(
        "credit", 0.2, seed, "dp", "naive",
        device="cpu", verbose=False,
        epochs=5, k_inner=3, pgd_steps=2, tau=1.0, n_seeds_planned=6,
    )
    return seed, r["acc_clean"], r["dp_clean"]


def main():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_job, s) for s in range(2)]
        for fut in as_completed(futs):
            print("result", fut.result(), f"{time.time()-t0:.1f}s", flush=True)
    print("POOL OK", flush=True)


if __name__ == "__main__":
    main()
