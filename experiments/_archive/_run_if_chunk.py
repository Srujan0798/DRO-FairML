
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.run_fairness_pgd import run_single_experiment

chunk_idx = int(sys.argv[1])
chunk_path = f"results/_if_chunk_{chunk_idx}.json"
out_path = f"results/_if_chunk_results_{chunk_idx}.json"
log_path = f"logs/if_chunk_{chunk_idx}.log"

with open(chunk_path) as f:
    specs = json.load(f)

results = []
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)

with open(log_path, "a") as logf:
    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    log(f"[chunk {chunk_idx}] {len(results)} already done, {len(specs)} total")

    for i, spec in enumerate(specs[len(results):], start=len(results)+1):
        t0 = time.time()
        try:
            result = run_single_experiment(
                spec["dataset"], spec["alpha"], spec["seed"], "if", spec["method"],
                device="cpu", verbose=False,
                epochs=60, k_inner=10, pgd_steps=20,
                tau=1.0, lambda_init=0.0, radii_mode="uniform",
                coordinated=False, n_seeds_planned=6
            )
            results.append(result)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            log(f"[chunk {chunk_idx} {i}/{len(specs)}] {spec} -> acc={result['acc_clean']:.3f} if={result['if_clean']:.4f} ({time.time()-t0:.0f}s)")
        except Exception as e:
            log(f"[chunk {chunk_idx} {i}/{len(specs)}] {spec} FAILED: {e}")

    log(f"[chunk {chunk_idx}] DONE. total={len(results)}")
