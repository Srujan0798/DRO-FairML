#!/usr/bin/env python3
import json, time, os, collections, subprocess
LOG = "logs/canonical_advancer_monitor.log"
def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg)
log("=== Canonical-Advancer Monitor started ===")
while True:
    try:
        with open("results/canonical_tau1.json") as f:
            data = json.load(f)
        n = len(data)
        ds = collections.Counter(r.get("dataset", "?") for r in data)
        last = data[-1] if data else {}
        has_next = any(r.get("dataset") in ("credit", "lsac") for r in data)
        last_log = ""
        try:
            last_log = subprocess.getoutput("tail -3 canonical.log 2>/dev/null | tail -1")
        except: pass
        active_log = ""
        try:
            active_log = subprocess.getoutput("tail -3 logs/canonical_run.log 2>/dev/null | tail -1")
        except: pass
        git = subprocess.getoutput("git status --porcelain | head -5")
        psout = subprocess.getoutput("ps aux | grep -E '21531|run_canonical.py' | grep -v grep | head -2")
        msg = (f"COUNTS: {n}/540 | DATASETS: {dict(ds)} | LAST: ds={last.get('dataset')} a={last.get('alpha')} s={last.get('seed')} atk={last.get('attack')} m={last.get('method')} | HAS_CREDIT_LSAC={has_next}\n"
               f"  LOG_LAST: {last_log} | ACTIVE_LOG_LAST: {active_log}\n"
               f"  GIT: {git.splitlines()[:3]}\n"
               f"  PS: {psout.splitlines()[:1]}")
        log(msg)
    except Exception as e:
        log(f"ERROR: {e}")
    time.sleep(30)
