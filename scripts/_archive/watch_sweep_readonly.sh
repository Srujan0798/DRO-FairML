#!/usr/bin/env bash
# READ-ONLY: poll canonical until IF=180, then exit. No writes, no regen, no commit.
cd /Users/srujansai/Desktop/DRO-FairML
for i in $(seq 1 60); do
  n=$(python3 -c "import json;d=json.load(open('results/canonical_tau1.json'));print(sum(1 for r in d if r['attack']=='if'))" 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S) IF=$n/180"
  [ "$n" -ge 180 ] && { echo "SWEEP COMPLETE"; exit 0; }
  ps -p 10146 >/dev/null 2>&1 || { echo "SWEEP DIED at $n/180"; exit 2; }
  sleep 60
done
echo "timeout after 60min at IF=$n"; exit 3
