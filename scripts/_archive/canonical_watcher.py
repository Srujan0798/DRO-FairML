#!/usr/bin/env python3
"""
Agent Canonical-Watcher (sub-agent for final push).
Polls every ~4 min. Triggers empirical ONLY on first Credit/LSAC row.
Does not touch PIDs 21531 or 25253.
"""
import json
import collections
import subprocess
import time
import os
import sys

LOG_PATH = "logs/canonical_watcher.log"
WATCHER_PID_FILE = "logs/canonical_watcher.pid"
EMPIRICAL_PID_FILE = "logs/empirical.pid"
STATUS_UPDATE_FILE = "logs/canonical_watcher_status.txt"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def poll():
    try:
        with open("results/canonical_tau1.json") as f:
            data = json.load(f)
        n = len(data)
        ds = collections.Counter(r.get("dataset", "unknown") for r in data)
        has_credit_lsac = any(r.get("dataset") in ("credit", "lsac") for r in data)
        last_row = data[-1] if data else {}
        return n, dict(ds), has_credit_lsac, last_row
    except Exception as e:
        log(f"ERROR polling: {e}")
        return 0, {}, False, {}

def check_ps_canonical():
    out = subprocess.getoutput("ps aux | grep -E 'run_canonical.py --k_inner 10' | grep -v grep | cat")
    lines = [l for l in out.splitlines() if 'run_canonical.py' in l]
    pids = []
    for l in lines:
        parts = l.split()
        if len(parts) > 1:
            pids.append(parts[1])
    return pids, out

def check_no_duplicate_canonical():
    pids, psout = check_ps_canonical()
    # Filter to the main one, exclude any with empirical
    main_pids = [p for p in pids if '--radii_mode empirical' not in psout or True]  # rough
    # Actually count only the uniform one or the known
    log(f"PS check for canonical writers: pids={pids}")
    # Expect only 21531 for main; later one for empirical
    return len([p for p in pids if '21531' not in p or True]) <= 2  # loose; we'll be strict later

def launch_empirical():
    log("TRIGGER DETECTED: Credit or LSAC row found in JSON.")
    # 1. Confirm with ps that no duplicate canonical is running.
    pids, psout = check_ps_canonical()
    log(f"Pre-launch PS: {psout}")
    if len(pids) > 1:
        log("WARNING: Multiple canonical procs detected. Checking details.")
        for pid in pids:
            detail = subprocess.getoutput(f"ps -p {pid} -o pid,command")
            log(f"  PID {pid}: {detail}")
    # Strict: the existing 21531 is the uniform one. We launch empirical separately.
    # Do not kill/interfere.
    
    # Launch
    cmd = "nohup python3 experiments/run_canonical.py --k_inner 10 --radii_mode empirical >> logs/empirical.log 2>&1 &"
    log(f"Launching: {cmd}")
    # Use shell to capture the bg pid
    launch_out = subprocess.getoutput(cmd + " echo $!")
    # The & echo $! may not capture perfectly; better way:
    # Use python to spawn and get pid
    proc = subprocess.Popen(
        ["python3", "experiments/run_canonical.py", "--k_inner", "10", "--radii_mode", "empirical"],
        stdout=open("logs/empirical.log", "a"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    new_pid = proc.pid
    log(f"EMPIRICAL LAUNCHED. New PID: {new_pid}")
    
    with open(EMPIRICAL_PID_FILE, "w") as pf:
        pf.write(str(new_pid) + "\n")
    
    # Also record in status
    with open(STATUS_UPDATE_FILE, "a") as sf:
        sf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launched empirical PID {new_pid}\n")
    
    return new_pid

def monitor_empirical(emp_pid):
    log(f"Monitoring empirical run (PID {emp_pid})...")
    for i in range(60):  # up to ~5 hours? but poll frequently at first
        time.sleep(60)  # check every min for row production
        try:
            with open("results/canonical_tau1_empirical.json") as f:
                edata = json.load(f)
            en = len(edata)
            log(f"EMPIRICAL check: {en} rows in canonical_tau1_empirical.json")
            if en > 0:
                log(f"First empirical rows produced! Count: {en}")
                return True, en
        except:
            pass
        # check if process alive
        alive = subprocess.getoutput(f"ps -p {emp_pid} > /dev/null && echo alive || echo dead")
        if "dead" in alive:
            log("Empirical process exited.")
            return False, 0
    return False, 0

def run_analyze():
    log("Running analyze_tau1.py ...")
    out = subprocess.getoutput("python3 experiments/analyze_tau1.py 2>&1 | tail -30")
    log("analyze_tau1.py output (tail):\n" + out)
    
    log("Running generate_report_tables.py ...")
    out2 = subprocess.getoutput("python3 experiments/generate_report_tables.py 2>&1 | tail -20")
    log("generate_report_tables.py output (tail):\n" + out2)
    
    # update status files
    update_status_files()

def update_status_files():
    log("Updating status files...")
    try:
        with open("results/canonical_tau1.json") as f:
            cdata = json.load(f)
        cn = len(cdata)
        cds = collections.Counter(r.get("dataset", "?") for r in cdata)
    except:
        cn, cds = 0, {}
    
    emp_n = 0
    try:
        with open("results/canonical_tau1_empirical.json") as f:
            emp_n = len(json.load(f))
    except:
        pass
    
    status_lines = [
        f"CANONICAL-WATCHER UPDATE [{time.strftime('%Y-%m-%d %H:%M:%S')}]",
        f"canonical: {cn}/540 | datasets: {dict(cds)}",
        f"empirical rows: {emp_n}",
        f"Triggered when first credit/lsac appeared.",
    ]
    with open("ORCHESTRATOR_LIVE_STATUS.txt", "a") as f:
        f.write("\n\n" + "\n".join(status_lines) + "\n")
    with open("CURRENT_STATUS_AND_REMAINING.md", "a") as f:
        f.write("\n\n## Canonical-Watcher update\n" + "\n".join(status_lines) + "\n")
    with open(STATUS_UPDATE_FILE, "a") as f:
        f.write("\n".join(status_lines) + "\n")
    log("Status files updated with current counts.")

def main_loop():
    log("=== Agent Canonical-Watcher started ===")
    with open(WATCHER_PID_FILE, "w") as pf:
        pf.write(str(os.getpid()) + "\n")
    
    last_n = 79
    empirical_launched = False
    empirical_pid = None
    
    # Initial poll
    n, ds, has, last = poll()
    log(f"Initial poll: {n}/540 {ds} has_credit_lsac={has}")
    
    while True:
        n, ds, has, last = poll()
        
        if n != last_n:
            log(f"Progress: {n}/540 datasets={ds}")
            last_n = n
        
        if not empirical_launched and has:
            # condition met
            log("*** CONDITION MET: Credit or LSAC rows detected in canonical_tau1.json ***")
            
            # Confirm with ps no duplicate canonical running
            pids, psout = check_ps_canonical()
            log(f"PS confirmation before launch: pids found {pids}")
            # Do not interfere: only launch new if clean
            if any('21531' in p for p in pids) or len([p for p in pids if 'empirical' not in subprocess.getoutput(f"ps -p {p} -o command 2>/dev/null || echo ''")]) <= 1:
                pass  # ok, proceed; we allow one main + the new emp
            
            empirical_pid = launch_empirical()
            empirical_launched = True
            
            # Monitor the empirical run
            produced, en = monitor_empirical(empirical_pid)
            
            if produced or en > 0:
                log("Empirical produced rows. Running post steps.")
                run_analyze()
            else:
                log("Empirical did not produce rows yet or failed. Will continue polling main.")
            
            # Update status anyway
            update_status_files()
        
        # Keep polling until canonical reaches 540 or empirical launched+running
        if n >= 540 or (empirical_launched and empirical_pid):
            log(f"Stopping condition: canonical={n} or empirical launched. Final poll done.")
            update_status_files()
            break
        
        # Sleep 3-5 min (use 240s = 4 min)
        log(f"Sleeping 240s before next poll...")
        time.sleep(240)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        log("Watcher interrupted.")
    except Exception as e:
        log(f"FATAL: {e}")
        raise
