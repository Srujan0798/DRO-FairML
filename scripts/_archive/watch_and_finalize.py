#!/usr/bin/env python3
"""Simple watcher: run this in another terminal. When lambda hits 72 or canonical grows, it auto finalizes."""
import json, os, time, subprocess, sys

def get_counts():
    try:
        lam = len(json.load(open("results/lambda_lr_grid.json")))
    except:
        lam = 0
    try:
        can = len(json.load(open("results/canonical_tau1.json")))
    except:
        can = 0
    return lam, can

def run(cmd, desc=""):
    print(f"\n>>> {desc or cmd}")
    os.system(cmd)

def main():
    print("Watcher started. Polling every 20s. Ctrl-C to stop.")
    last_lam, last_can = get_counts()
    print(f"Initial: lambda={last_lam}/72 canonical={last_can}/540")

    while True:
        time.sleep(20)
        lam, can = get_counts()
        if lam != last_lam or can != last_can:
            print(f"\n=== CHANGE DETECTED @ {time.strftime('%H:%M:%S')} ===")
            print(f"lambda: {lam}/72   canonical: {can}/540")
            if lam >= 72 and last_lam < 72:
                print("Lambda reached 72! Running finalize...")
                run("python3 finalize_experiments.py", "finalize")
            if can > last_can:
                print("Canonical grew. Refreshing summary + tables (safe)...")
                run("python3 experiments/analyze_tau1.py", "analyze_tau1")
                run("python3 experiments/generate_report_tables.py", "tables")
            last_lam, last_can = lam, can
        else:
            # quiet heartbeat
            sys.stdout.write(".")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
