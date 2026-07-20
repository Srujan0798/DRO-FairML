#!/bin/bash
echo "=== Quick poll loop started $(date) ==="
for i in 1 2 3; do
  echo "--- Poll $i at $(date +%H:%M:%S) ---"
  python3 logs/canonical_watcher_poll.py
  if [ $i -lt 3 ]; then
    echo "Sleeping 30s..."
    sleep 30
  fi
done
echo "=== Quick poll loop end at $(date) ==="
tail -5 logs/canonical_watcher.log | cat
