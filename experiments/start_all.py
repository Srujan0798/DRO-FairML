#!/usr/bin/env python3
"""Start all experiment processes in new sessions so they survive shell exit."""
import os
import subprocess
import time

os.makedirs("results/full_credit", exist_ok=True)
os.makedirs("results/full_lsac", exist_ok=True)

# Credit
subprocess.Popen(
    ["python3", "-u", "experiments/run_experiments.py",
     "--datasets", "credit", "--n_seeds", "10",
     "--output_dir", "results/full_credit"],
    stdout=open("results/full_credit/run.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("Started credit")

# LSAC
subprocess.Popen(
    ["python3", "-u", "experiments/run_experiments.py",
     "--datasets", "lsac", "--n_seeds", "10",
     "--output_dir", "results/full_lsac"],
    stdout=open("results/full_lsac/run.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("Started lsac")

# Adult (parallel by alpha)
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4]:
    d = f"results/adult_a{alpha}"
    os.makedirs(d, exist_ok=True)
    subprocess.Popen(
        ["python3", "-u", "experiments/run_experiments.py",
         "--datasets", "adult", "--alphas", str(alpha), "--n_seeds", "10",
         "--output_dir", d],
        stdout=open(f"{d}/run.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"Started adult α={alpha}")

print("All processes started with start_new_session=True")
