#!/usr/bin/env python3
"""Generic parallel ablation driver — shared by Agents A1, A2, A3, A4, A5, N5.

Isolated worker processes call run_single_experiment (no file I/O in workers);
parent-only atomic merge with checkpoint after each result; resume-safe keys;
full provenance on every row.

HARD RULE: results_file must never be results/canonical_tau1.json or
results/utkface_canonical.json (locked science for Aug 10).

Each agent supplies:
  - results_file: output path (separate ablation JSON only)
  - configs: list of (dataset, alpha, seed, method, attack, k_inner, pgd_steps,
              tau, lambda_init, lr_lambda, radii_mode, coordinated, corruptor_type, attack_k)
  - provenance_extras: extra fields stamped in the PARENT (not worker) on every row
"""
import os, sys, json, time, tempfile, traceback
import multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths that must never be written by ablation drivers
_LOCKED_SCIENCE = {
    os.path.abspath("results/canonical_tau1.json"),
    os.path.abspath("results/utkface_canonical.json"),
}


# Config tuple schema (in order): dataset, alpha, seed, method, attack, k_inner,
# pgd_steps, tau, lambda_init, lr_lambda, radii_mode, coordinated, corruptor_type,
# attack_k, [radii_scale, radii_clamp]. The last two are OPTIONAL for backward
# compatibility with Wave 1 drivers that use 14-tuples; when absent they default to
# (1.0, None) so the canonical closed-form radii are unchanged. Agent L2 (LSAC
# degeneracy fix) supplies the 16-tuple to thread the per-group radius clamp
# through to DroFairTrainer._compute_radii without disturbing other agents.
_KEY_FIELDS = ['dataset', 'alpha', 'seed', 'method', 'attack', 'k_inner', 'pgd_steps',
               'tau', 'lambda_init', 'lr_lambda', 'radii_mode', 'coordinated',
               'corruptor_type', 'attack_k', 'radii_scale', 'radii_clamp']


def _worker(cfg):
    """Module-level worker (must be importable by spawn). No globals — fully self-contained.

    Accepts either a 14-tuple (Wave 1 drivers; radii_scale=1.0, radii_clamp=None)
    or a 16-tuple (Agent L2 passes the per-group radius clamp). Backward-compatible.
    """
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    if len(cfg) == 14:
        (ds, a, s, m, attack, k_inner, pgd_steps, tau,
         lambda_init, lr_lambda, radii_mode, coordinated, corruptor_type, attack_k) = cfg
        radii_scale, radii_clamp = 1.0, None
    else:
        (ds, a, s, m, attack, k_inner, pgd_steps, tau,
         lambda_init, lr_lambda, radii_mode, coordinated, corruptor_type, attack_k,
         radii_scale, radii_clamp) = cfg
    return run_single_experiment(
        ds, a, s, attack, m, device="cpu", verbose=False,
        epochs=60, k_inner=k_inner, pgd_steps=pgd_steps,
        tau=tau, lambda_init=lambda_init, radii_mode=radii_mode,
        coordinated=coordinated, n_seeds_planned=6,
        corruptor_type=corruptor_type, lr_lambda=lr_lambda, attack_k=attack_k,
        radii_scale=radii_scale, radii_clamp=radii_clamp,
    )


def _key(r):
    return (r.get('dataset'), r.get('alpha'), r.get('seed'), r.get('method'),
            r.get('attack'), r.get('k_inner'), r.get('tau'),
            r.get('lambda_init'), r.get('lr_lambda'), r.get('radii_mode'),
            r.get('coordinated'), r.get('corruptor_type', 'adversarial'),
            r.get('attack_k', 5),
            r.get('radii_scale', 1.0), r.get('radii_clamp', None))


def missing_configs(rows, configs):
    have = {_key(r) for r in rows}
    out = []
    seen = set()
    for c in configs:
        # Normalize to a 16-element dict by padding missing trailing fields with
        # their defaults so 14-tuple configs key identically to legacy rows.
        pad = list(c) + [1.0, None][len(c):]
        k = _key(dict(zip(_KEY_FIELDS, pad)))
        if k in have or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def atomic_save(rows, path):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, path)


def _assert_safe_results_path(results_file):
    abspath = os.path.abspath(results_file)
    if abspath in _LOCKED_SCIENCE:
        raise RuntimeError(
            f"REFUSING to write ablation results to locked science file: {results_file}. "
            "Use a separate results/*_ablation.json (never canonical_tau1 / utkface_canonical)."
        )
    base = os.path.basename(abspath)
    if base in {"canonical_tau1.json", "utkface_canonical.json"}:
        raise RuntimeError(f"REFUSING locked basename: {base}")


# Machine-wide lock so at most ONE ablation job runs at a time. History: on
# 2026-08-04, 11 ablation jobs were launched concurrently by different agent
# sessions with no shared coordination — each spun up its own worker pool,
# oversubscribing the 14-core machine ~4x (load avg 280+, near-zero real
# throughput for 3+ hours). Fixed once by killing everything and running
# scripts/orchestrate_wave1.sh sequentially — then a SECOND uncoordinated
# agent session relaunched all remaining jobs concurrently again 2 minutes
# later, recreating the exact same problem. This lock makes that impossible:
# any run() call blocks (with a clear message) if another job already holds
# the lock, regardless of which script or which agent session calls it.
_LOCK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "logs", "ablation.lock")
)


class _AblationLock:
    """Advisory file lock (flock) shared by every run() call, from any process."""

    def __init__(self, path=_LOCK_PATH):
        self.path = path
        self._fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "a+")
        import fcntl
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"[ablation-lock] another ablation job is already running "
                f"(lock held on {self.path}). Waiting for it to finish rather "
                f"than oversubscribing the machine — see history above.",
                flush=True,
            )
            import fcntl as _fcntl
            _fcntl.flock(self._fh.fileno(), _fcntl.LOCK_EX)  # blocks until free
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc):
        import fcntl
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()


def _append_result(rows, res, provenance_extras, results_file, label, c, done, todo_n, t0):
    """Append one result with disk re-read + key-merge under a per-file flock.

    Concurrent parents writing the same JSON (should not happen when the
    machine-wide _AblationLock is held, but has happened when that lock was
    bypassed) must not clobber each other: always re-load disk, union by
    ``_key``, then atomic replace. Updates ``rows`` in place to the merged list.
    """
    if provenance_extras:
        for k, v in provenance_extras.items():
            res[k] = v
    import fcntl
    lock_path = results_file + ".writelock"
    os.makedirs(os.path.dirname(os.path.abspath(results_file)) or ".", exist_ok=True)
    with open(lock_path, "a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        disk = []
        if os.path.exists(results_file):
            try:
                with open(results_file) as f:
                    disk = json.load(f)
                if not isinstance(disk, list):
                    disk = []
            except (json.JSONDecodeError, OSError):
                disk = list(rows)
        by = {}
        for r in disk:
            by[_key(r)] = r
        for r in rows:
            by[_key(r)] = r
        by[_key(res)] = res
        merged = list(by.values())
        atomic_save(merged, results_file)
        rows[:] = merged
    print(
        f"[{label}][{done}/{todo_n}] {c[0]} a={c[1]} s={c[2]} {c[3]} {c[4]} "
        f"acc={res['acc_clean']:.3f} dp={res['dp_clean']:.4f} "
        f"if={res['if_clean']:.4f} ({time.time()-t0:.0f}s) rows_on_disk={len(rows)}",
        flush=True,
    )


def run(results_file, configs, provenance_extras=None, workers=4, label="ablation"):
    """Run missing configs into results_file. Resume-safe. Default workers=4 (stable on macOS).

    Acquires the machine-wide ablation lock first, so at most one ablation job
    (from any script, any agent session) ever runs at a time. See _AblationLock
    docstring/comment above for why this exists.
    """
    with _AblationLock():
        return _run_locked(results_file, configs, provenance_extras, workers, label)


def _run_locked(results_file, configs, provenance_extras=None, workers=4, label="ablation"):
    _assert_safe_results_path(results_file)
    rows = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{results_file} is not a JSON list")
    todo = missing_configs(rows, configs)
    print(
        f"[{label}] {results_file}: have {len(rows)} rows; {len(todo)} missing; workers={workers}",
        flush=True,
    )
    if not todo:
        print(f"[{label}] nothing to do — already complete.")
        return rows

    done = 0
    t0 = time.time()
    n_todo = len(todo)

    # Process pools on some macOS/Python combos die with 0 results (BrokenProcessPool /
    # leaked semaphores). Prefer sequential unless ABLATION_WORKERS>1 is proven stable.
    # Override: ABLATION_WORKERS env or workers arg; if pool yields nothing, sequential.
    env_w = os.environ.get("ABLATION_WORKERS")
    if env_w is not None:
        try:
            workers = int(env_w)
        except ValueError:
            pass
    try:
        workers = int(workers)
    except (TypeError, ValueError):
        workers = 1
    # 0 or negative → sequential (same as 1)
    if workers < 1:
        workers = 1

    use_pool = workers > 1
    if use_pool:
        # ProcessPool with fork gives real parallelism for torch CPU-bound work
        # (torch releases the GIL during matmuls, but Python orchestration between
        # ops still needs the GIL, so ThreadPoolExecutor doesn't parallelize well).
        # Fork inherits interpreter state so the worker imports cleanly on macOS.
        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        print(f"[{label}] ProcessPoolExecutor(workers={workers}, {ctx.get_start_method()})…", flush=True)
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(_worker, c): c for c in todo}
                for fut in as_completed(futs):
                    c = futs[fut]
                    try:
                        res = fut.result()
                        done += 1
                        _append_result(
                            rows, res, provenance_extras, results_file,
                            label, c, done, n_todo, t0,
                        )
                    except Exception as e:
                        print(f"[{label}] FAILED {c}: {e}", flush=True)
                        traceback.print_exc()
        except Exception as e:
            print(f"[{label}] pool error ({e}).", flush=True)

        if done == 0 and n_todo > 0:
            print(
                f"[{label}] pool produced 0/{n_todo} rows — falling back to sequential.",
                flush=True,
            )
            use_pool = False

    remaining = missing_configs(rows, configs)
    if remaining and (not use_pool or done < n_todo):
        print(f"[{label}] sequential pass for {len(remaining)} configs…", flush=True)
        for c in remaining:
            try:
                res = _worker(c)
                done += 1
                _append_result(
                    rows, res, provenance_extras, results_file,
                    label, c, done, n_todo, t0,
                )
            except Exception as e:
                print(f"[{label}] SEQUENTIAL FAILED {c}: {e}", flush=True)
                traceback.print_exc()

    print(
        f"[{label}] DONE. {results_file} now {len(rows)} rows; {time.time()-t0:.0f}s total",
        flush=True,
    )
    return rows


if __name__ == "__main__":
    print("This is a library — run the specific agent script (run_a1_knn.py, etc.)")