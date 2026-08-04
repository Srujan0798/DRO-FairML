#!/usr/bin/env python3
"""Agent N2 — high-α rescue, Kuldeep's Jun-16 3-step protocol LITERALLY.

Kuldeep, Jun 16, dictated a procedure we never executed:
  "Different tau value 1st if not improving then change learning rates for
   lamda or something else check loss convergence plots and choose according
   to it on validation set".
Execute it LITERALLY, IN ORDER, all rows into ONE file.

STEP 1 — per-α tau:  tau ∈ {2, 5, 20}  × α ∈ {0.3, 0.4} × 6 seeds × 2 methods
                    = 72 configs  (Adult, dp)
STEP 2 — lr_lambda arm: lr_lambda=0.01 (canonical is 5e-3) × α ∈ {0.3, 0.4}
                    × 6 seeds × 2 methods = 24 configs  (Adult, dp, tau=1.0)
STEP 3 — convergence diagnostics: epochs=200 WITH dump_history=True
                    (history JSONs land in results/history_*.json per the
                    dump_history code in run_fairness_pgd.py), tau=1.0
                    (canonical) × α ∈ {0.3, 0.4} × 6 seeds × 2 methods
                    = 24 configs  (Adult, dp)
TOTAL = 120 configs -> results/high_alpha_tau.json

This driver does NOT touch results/canonical_tau1.json. The shared
run_ablation_parallel.run() worker hardcodes epochs=60 and never passes
dump_history, so STEP 3 cannot reuse it — this driver has its own worker.

Launch: ABLATION_WORKERS=4 (6+ other drivers in flight; do NOT exceed 4).
"""
import os, sys, json, time, tempfile, traceback
import multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LOCKED_SCIENCE = {
    os.path.abspath("results/canonical_tau1.json"),
    os.path.abspath("results/utkface_canonical.json"),
}
RESULTS_FILE = "results/high_alpha_tau.json"

DATASET = "adult"
ATTACK = "dp"
SEEDS = list(range(6))
ALPHAS = [0.3, 0.4]
METHODS = ["naive", "dro"]

# Step params
TAUS_STEP1 = [2.0, 5.0, 20.0]
LR_LAMBDA_STEP2 = 0.01      # canonical lr_lambda is 5e-3
EPOCHS_STEP3 = 200
CANONICAL_TAU = 1.0
CANONICAL_LR_LAMBDA = 5e-3

ADULT_CONSTANT_PREDICTOR = 0.7521


def _assert_safe(path):
    abspath = os.path.abspath(path)
    if abspath in _LOCKED_SCIENCE:
        raise RuntimeError(f"REFUSING to write to locked science file: {path}")
    if os.path.basename(abspath) in {"canonical_tau1.json", "utkface_canonical.json"}:
        raise RuntimeError(f"REFUSING locked basename: {os.path.basename(abspath)}")


def _worker(cfg):
    """Module-level worker (must be importable by spawn). No globals — self-contained.

    cfg: dict with all knobs needed by run_single_experiment, INCLUDING epochs
    and dump_history (which the shared run_ablation_parallel._worker hardcodes
    away). Returns one result-row dict.
    """
    import torch
    torch.set_num_threads(1)
    from experiments.run_fairness_pgd import run_single_experiment
    return run_single_experiment(
        cfg["dataset"], cfg["alpha"], cfg["seed"], cfg["attack"], cfg["method"],
        device="cpu", verbose=False,
        epochs=cfg["epochs"], k_inner=10, pgd_steps=20,
        tau=cfg["tau"], lambda_init=0.0, radii_mode="uniform",
        coordinated=False, n_seeds_planned=6,
        corruptor_type="adversarial", lr_lambda=cfg["lr_lambda"], attack_k=5,
        radii_scale=1.0, radii_clamp=None,
        dump_history=cfg["dump_history"],
    )


def _key(r):
    """Resume-safe key — distinguishes every arm in this file.

    Includes step, epochs, tau, lr_lambda, alpha, seed, method so that:
      - STEP 1 (tau varies), STEP 2 (lr_lambda varies), STEP 3 (epochs=200 +
        dump_history) never collide with each other or with canonical rows.
    """
    return (
        r.get("ablation_step"),
        r.get("dataset"), r.get("alpha"), r.get("seed"), r.get("method"),
        r.get("attack"),
        float(r.get("tau", 1.0)),
        float(r.get("lr_lambda", 5e-3)),
        int(r.get("epochs", 60)),
        int(r.get("dump_history", 0)),
    )


def build_configs():
    """Build the 120-config list (STEP1 + STEP2 + STEP3), de-duplicated."""
    configs = []
    seen = set()

    def _add(step, alpha, seed, method, tau, lr_lambda, epochs, dump_history):
        k = (step, DATASET, alpha, seed, method, ATTACK,
             float(tau), float(lr_lambda), int(epochs), int(bool(dump_history)))
        if k in seen:
            return
        seen.add(k)
        configs.append({
            "ablation_step": step,
            "dataset": DATASET, "alpha": alpha, "seed": seed,
            "attack": ATTACK, "method": method,
            "tau": float(tau), "lr_lambda": float(lr_lambda),
            "epochs": int(epochs), "dump_history": bool(dump_history),
        })

    # STEP 1 — per-α tau: tau ∈ {2, 5, 20}, Adult, dp, α ∈ {0.3, 0.4},
    # 6 seeds, 2 methods = 72 configs. epochs=60 (canonical), lr_lambda=5e-3.
    for alpha in ALPHAS:
        for seed in SEEDS:
            for method in METHODS:
                for tau in TAUS_STEP1:
                    _add("step1_tau", alpha, seed, method, tau,
                         CANONICAL_LR_LAMBDA, epochs=60, dump_history=False)

    # STEP 2 — lr_lambda extra arm: lr_lambda=0.01 at α ∈ {0.3, 0.4}, Adult,
    # dp, 6 seeds, 2 methods = 24 configs. tau=1.0 (canonical), epochs=60.
    for alpha in ALPHAS:
        for seed in SEEDS:
            for method in METHODS:
                _add("step2_lr_lambda", alpha, seed, method, CANONICAL_TAU,
                     LR_LAMBDA_STEP2, epochs=60, dump_history=False)

    # STEP 3 — convergence diagnostics: epochs=200 WITH dump_history=True,
    # tau=1.0 (canonical), α ∈ {0.3, 0.4}, Adult, dp, 6 seeds, 2 methods
    # = 24 configs. dump_history code writes results/history_*.json.
    for alpha in ALPHAS:
        for seed in SEEDS:
            for method in METHODS:
                _add("step3_convergence", alpha, seed, method, CANONICAL_TAU,
                     CANONICAL_LR_LAMBDA, epochs=EPOCHS_STEP3, dump_history=True)

    return configs


def _load_rows():
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{RESULTS_FILE} is not a JSON list")
    return rows


def _atomic_save(rows, path):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, path)


def _missing(rows, configs):
    have = {_key(r) for r in rows}
    out, seen = [], set()
    for c in configs:
        k = (c["ablation_step"], c["dataset"], c["alpha"], c["seed"],
             c["method"], c["attack"], c["tau"], c["lr_lambda"],
             c["epochs"], int(c["dump_history"]))
        if k in have or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _provenance_stamp(res, cfg):
    res["ablation_step"] = cfg["ablation_step"]
    res["dump_history"] = int(bool(cfg["dump_history"]))
    res["adult_constant_predictor"] = ADULT_CONSTANT_PREDICTOR
    res["n_seeds_planned"] = 6
    return res


def main():
    _assert_safe(RESULTS_FILE)

    env_w = os.environ.get("ABLATION_WORKERS")
    workers = 4
    if env_w is not None:
        try:
            workers = int(env_w)
        except ValueError:
            pass
    if workers < 1:
        workers = 1
    if workers > 4:
        print(f"[N2] WARNING: ABLATION_WORKERS={workers} > 4; capping at 4 "
              f"(other drivers in flight).", flush=True)
        workers = 4

    rows = _load_rows()
    configs = build_configs()
    todo = _missing(rows, configs)
    print(f"[N2] {RESULTS_FILE}: have {len(rows)} rows; {len(todo)} missing; "
          f"workers={workers}", flush=True)
    if not todo:
        print(f"[N2] nothing to do — already complete.", flush=True)
        return

    done = 0
    n_todo = len(todo)
    t0 = time.time()

    use_pool = workers > 1
    if use_pool:
        ctx = mp.get_context("fork") if sys.platform != "win32" \
            else mp.get_context("spawn")
        print(f"[N2] ProcessPoolExecutor(workers={workers}, "
              f"{ctx.get_start_method()})…", flush=True)
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(_worker, c): c for c in todo}
                for fut in as_completed(futs):
                    c = futs[fut]
                    try:
                        res = fut.result()
                        _provenance_stamp(res, c)
                        rows.append(res)
                        _atomic_save(rows, RESULTS_FILE)
                        done += 1
                        print(f"[N2][{done}/{n_todo}] step={c['ablation_step']} "
                              f"a={c['alpha']} s={c['seed']} {c['method']} "
                              f"tau={c['tau']} lr={c['lr_lambda']} "
                              f"epochs={c['epochs']} dh={int(c['dump_history'])} "
                              f"acc={res['acc_clean']:.3f} "
                              f"dp={res['dp_clean']:.4f} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                    except Exception as e:
                        print(f"[N2] FAILED {c}: {e}", flush=True)
                        traceback.print_exc()
        except BrokenExecutor as e:
            print(f"[N2] pool broken ({e}).", flush=True)
        if done == 0 and n_todo > 0:
            print(f"[N2] pool produced 0/{n_todo} — falling back to sequential.",
                  flush=True)
            use_pool = False

    remaining = _missing(rows, configs)
    if remaining and (not use_pool or done < n_todo):
        print(f"[N2] sequential pass for {len(remaining)} configs…", flush=True)
        for c in remaining:
            try:
                res = _worker(c)
                _provenance_stamp(res, c)
                rows.append(res)
                _atomic_save(rows, RESULTS_FILE)
                done += 1
                print(f"[N2][{done}/{n_todo}] step={c['ablation_step']} "
                      f"a={c['alpha']} s={c['seed']} {c['method']} "
                      f"tau={c['tau']} lr={c['lr_lambda']} "
                      f"epochs={c['epochs']} dh={int(c['dump_history'])} "
                      f"acc={res['acc_clean']:.3f} "
                      f"dp={res['dp_clean']:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[N2] SEQUENTIAL FAILED {c}: {e}", flush=True)
                traceback.print_exc()

    print(f"[N2] DONE. {RESULTS_FILE} now {len(rows)} rows; "
          f"{time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()