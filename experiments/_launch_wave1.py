#!/usr/bin/env python3
"""Sequential Wave-1 ablation launcher (spawn-safe)."""
import subprocess, sys, os, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scripts = [
    ("experiments/run_a3_lambda.py", "4"),
    ("experiments/run_a4_rva.py", "4"),
    ("experiments/run_a5_empirical.py", "4"),
    ("experiments/run_n5_kinner.py", "4"),
    ("experiments/run_a1_knn.py", "4"),
    ("experiments/run_a2_tau.py", "4"),
]
for s, w in scripts:
    print(f"\n==== LAUNCH {s} workers={w} {time.strftime('%H:%M:%S')} ====", flush=True)
    rc = subprocess.call([sys.executable, s, w])
    print(f"==== EXIT {s} rc={rc} {time.strftime('%H:%M:%S')} ====", flush=True)
print("ALL WAVE1 DONE", flush=True)
