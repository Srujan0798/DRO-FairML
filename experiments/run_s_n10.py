#!/usr/bin/env python3
"""AGENT S — n=6 -> n=10 seed extension (APPEND ONLY).

This is the ONE sanctioned exception to the locked-file rule for
results/canonical_tau1.json. It appends seeds 6-9 of the canonical tabular
grid to the existing 540 rows, producing 900 rows total.

Contract:
  - APPEND ONLY. The existing 540 rows are never modified or recomputed.
  - Resume-safe: only schedule configs (seeds 6-9) missing from canonical.
  - Single writer: the PARENT is the only process that touches the file;
    workers return dicts (no file I/O). Check `ps aux | grep run_` first.
  - Atomic write: temp file + os.replace after each completion.
  - Hard post-check: file has exactly 900 rows AND first 540 byte-identical.

Canonical config pinned on every new row:
  tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0,
  radii_mode='uniform', coordinated=False, n_seeds_planned=10.

  (n_seeds_planned is stamped 10 on the NEW rows to record the extended
  plan; the original 540 rows keep n_seeds_planned=6 and are untouched.)
"""
import os, sys, json, time, tempfile, hashlib, traceback
import multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CANON = os.path.join(ROOT, "results", "canonical_tau1.json")
SNAPSHOT = "/tmp/canonical_tau1_first540.json"
EXPECTED_EXISTING = 540
EXPECTED_FINAL = 900

DATASETS = ["adult", "credit", "lsac"]
ALPHAS   = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS    = [6, 7, 8, 9]          # APPEND ONLY — seeds 0-5 already locked
METHODS  = ["naive", "dro"]
ATTACKS  = ["dp", "if", "combined"]


def _worker(cfg):
    """Module-level worker (importable by spawn/fork). No file I/O."""
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    ds, a, s, m, attack = cfg
    return run_single_experiment(
        ds, a, s, attack, m, device="cpu", verbose=False,
        epochs=60, k_inner=10, pgd_steps=20,
        tau=1.0, lambda_init=0.0, radii_mode="uniform",
        coordinated=False, n_seeds_planned=10,
    )


def _row_key(r):
    """Identity key for a canonical row (matches existing 540 schema)."""
    return (r.get("dataset"), r.get("alpha"), r.get("seed"),
            r.get("method"), r.get("attack"))


def missing_configs(rows):
    """Only seeds 6-9 configs not already present."""
    have = {_row_key(r) for r in rows}
    out = []
    seen = set()
    for ds in DATASETS:
        for a in ALPHAS:
            for s in SEEDS:
                for m in METHODS:
                    for attack in ATTACKS:
                        k = (ds, a, s, m, attack)
                        if k in have or k in seen:
                            continue
                        seen.add(k)
                        out.append(k)
    # stable ordering: dataset, attack, alpha, seed, method
    order = {"adult": 0, "credit": 1, "lsac": 2,
             "dp": 0, "if": 1, "combined": 2,
             "naive": 0, "dro": 1}
    out.sort(key=lambda c: (order[c[0]], order[c[4]], c[1], c[2], order[c[3]]))
    return out


def atomic_save(rows):
    d = os.path.dirname(CANON)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, CANON)


def _sha_first540(path):
    with open(path) as f:
        rows = json.load(f)
    return hashlib.sha256(json.dumps(rows[:EXPECTED_EXISTING], indent=2).encode()).hexdigest()


def verify(head_hash_before):
    """Hard post-check. Returns True iff file is exactly 900 rows AND
    the first 540 are byte-identical to the pre-run snapshot."""
    with open(CANON) as f:
        rows = json.load(f)
    n = len(rows)
    head_hash_after = hashlib.sha256(
        json.dumps(rows[:EXPECTED_EXISTING], indent=2).encode()
    ).hexdigest()
    head_ok = (head_hash_after == head_hash_before)
    count_ok = (n == EXPECTED_FINAL)
    # also compare against the saved snapshot file for belt-and-braces
    snap_ok = True
    if os.path.exists(SNAPSHOT):
        with open(SNAPSHOT) as f:
            snap = json.load(f)
        snap_ok = (snap == rows[:EXPECTED_EXISTING])
    print(f"[VERIFY] rows={n} (expected {EXPECTED_FINAL})  count_ok={count_ok}", flush=True)
    print(f"[VERIFY] first-540 sha256 before={head_hash_before}", flush=True)
    print(f"[VERIFY] first-540 sha256 after ={head_hash_after}", flush=True)
    print(f"[VERIFY] first-540 byte-identical={head_ok}  snapshot_match={snap_ok}", flush=True)
    return count_ok and head_ok and snap_ok


def main():
    workers = 4
    env_w = os.environ.get("ABLATION_WORKERS")
    if env_w is not None:
        try:
            workers = int(env_w)
        except ValueError:
            pass
    if workers < 1:
        workers = 1

    # --- pre-flight: snapshot + hash the first 540 rows ---
    with open(CANON) as f:
        rows = json.load(f)
    if len(rows) != EXPECTED_EXISTING:
        print(f"ABORT: canonical has {len(rows)} rows, expected {EXPECTED_EXISTING}. "
              f"Another writer may have already run. Inspect before proceeding.", flush=True)
        return 2
    head_hash_before = hashlib.sha256(
        json.dumps(rows[:EXPECTED_EXISTING], indent=2).encode()
    ).hexdigest()
    with open(SNAPSHOT, "w") as f:
        json.dump(rows[:EXPECTED_EXISTING], f, indent=2)
    print(f"[PRE] canonical rows={len(rows)}; first-540 sha256={head_hash_before}", flush=True)
    print(f"[PRE] snapshot saved -> {SNAPSHOT}", flush=True)

    todo = missing_configs(rows)
    print(f"[PRE] {len(todo)} configs missing (seeds 6-9); workers={workers}", flush=True)
    if not todo:
        print("Nothing to do — seeds 6-9 already present. Running verify only.", flush=True)
        verify(head_hash_before)
        return 0

    done = 0
    failed = 0
    t0 = time.time()
    n_todo = len(todo)

    use_pool = workers > 1
    if use_pool:
        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        print(f"[RUN] ProcessPoolExecutor(workers={workers}, {ctx.get_start_method()})", flush=True)
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(_worker, c): c for c in todo}
                for fut in as_completed(futs):
                    c = futs[fut]
                    try:
                        res = fut.result()
                        # APPEND ONLY — never touch existing rows
                        rows.append(res)
                        done += 1
                        atomic_save(rows)
                        print(
                            f"[{done}/{n_todo}] {c[0]} {c[4]} a={c[1]} s={c[2]} {c[3]} "
                            f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} "
                            f"if={res['if_clean']:.4f} ({time.time()-t0:.0f}s)",
                            flush=True,
                        )
                    except Exception as e:
                        failed += 1
                        print(f"FAILED {c}: {e}", flush=True)
                        traceback.print_exc()
        except BrokenExecutor as e:
            print(f"[RUN] pool broken ({e}); falling back to sequential for the rest.", flush=True)
            use_pool = False

    # sequential fallback for anything the pool didn't finish
    remaining = missing_configs(rows)
    if remaining:
        print(f"[RUN] sequential pass for {len(remaining)} remaining configs…", flush=True)
        for c in remaining:
            try:
                res = _worker(c)
                rows.append(res)
                done += 1
                atomic_save(rows)
                print(
                    f"[{done}/{n_todo}] {c[0]} {c[4]} a={c[1]} s={c[2]} {c[3]} "
                    f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} "
                    f"if={res['if_clean']:.4f} ({time.time()-t0:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                failed += 1
                print(f"SEQUENTIAL FAILED {c}: {e}", flush=True)
                traceback.print_exc()

    print(f"[DONE] appended {done}/{n_todo} ({failed} failed); "
          f"canonical now {len(rows)} rows; {time.time()-t0:.0f}s total", flush=True)

    # --- hard post-check ---
    ok = verify(head_hash_before)
    if not ok:
        print("ABORT: VERIFY FAILED — first 540 rows changed or count != 900. "
              "Restore results/canonical_tau1.json from git immediately.", flush=True)
        return 1
    print("[VERIFY] PASS — 900 rows, first 540 byte-identical.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())