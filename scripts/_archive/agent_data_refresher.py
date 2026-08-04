#!/usr/bin/env python3
"""Agent Data-Refresher: periodic artifact refresh without launching new runs."""
import json
import time
import subprocess
import os
import sys

LOG_FILE = "logs/agent_data_refresher.log"
STATUS_FILES = ["ORCHESTRATOR_LIVE_STATUS.txt", "DELIVERABLES_CHECKLIST.txt"]

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def get_counts():
    try:
        lam = len(json.load(open("results/lambda_lr_grid.json")))
    except:
        lam = -1
    try:
        can = len(json.load(open("results/canonical_tau1.json")))
        datasets = {}
        for r in json.load(open("results/canonical_tau1.json")):
            ds = r.get("dataset", "?")
            datasets[ds] = datasets.get(ds, 0) + 1
    except:
        can = -1
        datasets = {}
    return lam, can, datasets

def run_cmd(cmd, desc):
    log(f"RUN: {desc} -> {cmd}")
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        out = (res.stdout or "")[-1500:]
        if res.returncode != 0:
            log(f"ERROR in {desc}: {res.stderr[-500:]}")
            return False
        log(f"DONE: {desc} (rc=0, out tail: {out[-200:].strip()})")
        return True
    except Exception as e:
        log(f"EXCEPTION in {desc}: {e}")
        return False

def update_status_files(lam, can, datasets, ts):
    adult_only = datasets.get('adult', 0) == can and can > 0
    status = f"""=== AGENT DATA-REFRESHER UPDATE @ {ts} ===
lambda: {lam}/72
canonical: {can}/540
canonical datasets: {dict(datasets)}
Adult only: {adult_only}
Last runs: analyze_tau1.py + generate_report_tables.py + finalize status
"""

    for sf in STATUS_FILES:
        try:
            with open(sf, "a") as f:
                f.write(f"\n\n{status}\n")
            log(f"Updated {sf} with counts {lam}/{can}")
        except Exception as e:
            log(f"Failed to update {sf}: {e}")

def main_loop():
    log("=== AGENT DATA-REFRESHER STARTED ===")
    last_lam, last_can, _ = get_counts()
    log(f"Initial: lambda {last_lam}/72, canonical {last_can}/540")
    
    # Initial refresh
    run_cmd("python3 finalize_experiments.py status", "finalize status")
    run_cmd("python3 experiments/analyze_tau1.py", "analyze_tau1")
    run_cmd("python3 experiments/generate_report_tables.py", "generate_report_tables")
    lam, can, ds = get_counts()
    update_status_files(lam, can, ds, time.strftime("%Y-%m-%d %H:%M:%S"))
    
    poll_interval = 300  # 5 min
    while True:
        time.sleep(poll_interval)
        lam, can, ds = get_counts()
        log(f"Poll: lambda {lam}/72, canonical {can}/540, datasets {ds}")
        
        changed = (lam != last_lam) or (can != last_can)
        if changed:
            log(f"DETECTED CHANGE: lambda {last_lam}->{lam}, canonical {last_can}->{can}")
            run_cmd("python3 finalize_experiments.py status", "finalize status (on change)")
            run_cmd("python3 experiments/analyze_tau1.py", "analyze_tau1 (on change)")
            run_cmd("python3 experiments/generate_report_tables.py", "generate_report_tables (on change)")
            update_status_files(lam, can, ds, time.strftime("%Y-%m-%d %H:%M:%S"))
            last_lam, last_can = lam, can
        else:
            # Periodic even without change (every ~10min effective)
            if int(time.time()) % 600 < 30:  # rough every 10 min
                log("Periodic refresh (no count change)")
                run_cmd("python3 finalize_experiments.py status", "finalize status (periodic)")
                run_cmd("python3 experiments/analyze_tau1.py", "analyze_tau1 (periodic)")
                run_cmd("python3 experiments/generate_report_tables.py", "generate_report_tables (periodic)")
                update_status_files(lam, can, ds, time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Check if grids complete to stop?
        if lam >= 72 and can >= 540:
            log("GRIDS COMPLETE: lambda 72/72 + canonical 540/540. Stopping refresher loop.")
            break
    log("=== AGENT DATA-REFRESHER FINISHED ===")

if __name__ == "__main__":
    main_loop()
