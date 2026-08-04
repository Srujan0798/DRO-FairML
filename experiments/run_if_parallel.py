#!/usr/bin/env python3
"""Safe parallel completion of the IF-attack third of the canonical grid.

The stock runner (run_fairness_pgd.py) writes the whole results file after every
row, so running several copies in parallel corrupts the file via write races. This
driver avoids that: each missing IF config runs in an isolated worker process that
returns a dict (no file I/O), and the parent merges everything into
results/canonical_tau1.json ONCE at the end (atomic temp+rename).

Resume-safe: only configs missing from canonical are scheduled. Provenance is pinned
to the canonical contract (tau=1.0, k_inner=10, epochs=60, pgd_steps=20,
lambda_init=0.0, radii_mode='uniform', coordinated=False, n_seeds_planned=6).
"""
import os, sys, json, time, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "canonical_tau1.json")

DATASETS = ["adult", "credit", "lsac"]
ALPHAS   = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS    = range(6)
METHODS  = ["naive", "dro"]
ATTACK   = "if"


def _worker(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    ds, a, s, m = cfg
    return run_single_experiment(
        ds, a, s, ATTACK, m, device="cpu", verbose=False,
        epochs=60, k_inner=10, pgd_steps=20,
        tau=1.0, lambda_init=0.0, radii_mode="uniform",
        coordinated=False, n_seeds_planned=6,
    )


def missing_configs(rows):
    have = {(r["dataset"], r["alpha"], r["seed"], r["method"])
            for r in rows if r.get("attack") == ATTACK}
    return [(ds, a, s, m)
            for ds in DATASETS for a in ALPHAS for s in SEEDS for m in METHODS
            if (ds, a, s, m) not in have]


def atomic_save(rows):
    d = os.path.dirname(CANON)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, CANON)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    with open(CANON) as f:
        rows = json.load(f)
    todo = missing_configs(rows)
    print(f"canonical has {len(rows)} rows; {len(todo)} IF configs missing; workers={workers}", flush=True)
    if not todo:
        print("nothing to do — IF third already complete."); return

    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, c): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                res = fut.result()
                rows.append(res)
                done += 1
                atomic_save(rows)   # checkpoint after each completion (single writer = parent)
                print(f"[{done}/{len(todo)}] {c[0]} a={c[1]} s={c[2]} {c[3]} "
                      f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} if={res['if_clean']:.4f} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
            except Exception as e:
                print(f"FAILED {c}: {e}", flush=True)

    ifn = sum(1 for r in rows if r.get("attack") == ATTACK)
    print(f"DONE. canonical now {len(rows)} rows; IF rows={ifn}/180; {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
