#!/usr/bin/env python3
import time, subprocess, json, collections, sys

delay = int(sys.argv[1]) if len(sys.argv)>1 else 180
label = sys.argv[2] if len(sys.argv)>2 else "poll"

print(f"Sleeping {delay}s for {label}...")
time.sleep(delay)

print(f"=== DELAYED {label} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
with open("results/canonical_tau1.json") as f:
    data = json.load(f)
n = len(data)
ds = collections.Counter(r.get("dataset", "?") for r in data)
has = any(r.get("dataset") in ("credit", "lsac") for r in data)
print(f"CANONICAL: {n}/540 | DS: {dict(ds)} | HAS_CREDIT_LSAC={has}")
ps = subprocess.getoutput("ps aux | grep -E '21531|25253|39913|run_canonical' | grep -v grep")
print("KEY_PIDS:\n" + ps)
with open("logs/canonical_watcher.log", "a") as lf:
    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DELAYED_{label}: count={n} has_cl={has} ds={dict(ds)}\n")
print("Logged.")
