#!/usr/bin/env python3
"""Quick progress checker for Fairness-PGD experiments."""
import json
import time
import subprocess

with open('results/fairness_pgd_results.json') as f:
    d = json.load(f)

print(f"[{time.strftime('%H:%M:%S')}] Progress: {len(d)}/270 ({100*len(d)/270:.0f}%)")

from collections import Counter
for k, v in sorted(Counter(r['dataset'] for r in d).items()):
    print(f"  {k}: {v}")

# Check process
result = subprocess.run(['ps', '-p', '81903'], capture_output=True, text=True)
if result.returncode == 0:
    print("Process: PID 81903 RUNNING")
else:
    print("Process: NOT RUNNING")

print(f"ETA: ~{(270-len(d))*50/3600:.1f} hours")
