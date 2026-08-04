#!/usr/bin/env python3
"""Sequential Wave-1 ablation launcher (stable on macOS; no process-pool hangs)."""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scripts = [
    "experiments/run_a3_lambda.py",
    "experiments/run_a4_rva.py",
    "experiments/run_a5_empirical.py",
    "experiments/run_n5_kinner.py",
    "experiments/run_a1_knn.py",
    "experiments/run_a2_tau.py",
]
for s in scripts:
    print(f"\n==== LAUNCH {s} sequential {time.strftime('%H:%M:%S')} ====", flush=True)
    # workers=1 → sequential in driver (stable on macOS). Never pass 0.
    rc = subprocess.call([sys.executable, s, "1"])
    print(f"==== EXIT {s} rc={rc} {time.strftime('%H:%M:%S')} ====", flush=True)
print("ALL WAVE1 DONE", flush=True)
