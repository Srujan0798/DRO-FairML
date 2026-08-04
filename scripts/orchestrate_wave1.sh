#!/usr/bin/env bash
# Runs every remaining Wave-1/1.5 ablation job SEQUENTIALLY, each at full worker
# budget. Replaces running them all concurrently (which oversubscribed the 14-core
# machine 4x: 9 jobs x ~6 workers = 57 processes, load avg 280+, near-zero real
# throughput). Every underlying job is checkpointed (atomic_save after each result)
# and resume-safe (missing_configs), so killing the old concurrent runs lost
# nothing — this just re-drives the same jobs to completion, one at a time.
set -uo pipefail
cd /Users/srujansai/Desktop/DRO-FairML
LOG=logs/orchestrate_wave1.log
mkdir -p logs
: >"$LOG"
export ABLATION_WORKERS=12

run_job() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$LOG"
  python3 "$@" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)" | tee -a "$LOG"
}

# Ordered by (remaining work / value) — smallest/highest-value first so partial
# progress accumulates fast and the highest-priority advisor asks land soonest.
run_job "A3-Lambda"            experiments/run_a3_lambda.py
run_job "N2-HighAlpha"         experiments/run_n2_high_alpha.py
run_job "A5-Empirical"         experiments/run_a5_empirical.py
run_job "L2-LSAC"              experiments/run_l2_lsac_radii.py
run_job "A4-RvA"               experiments/run_a4_rva.py
run_job "A2-Tau"               experiments/run_a2_tau.py
run_job "N5-Kinner"            experiments/run_n5_kinner.py
run_job "A1-kNN"               experiments/run_a1_knn.py
run_job "N1-AttackStrength-A"  experiments/run_n1_attack_radius.py a
run_job "N1-AttackStrength-B"  experiments/run_n1_attack_radius.py b
run_job "S-N10-Extension"      experiments/run_s_n10.py

echo "[$(date +%H:%M:%S)] ORCHESTRATOR COMPLETE — all Wave-1/1.5 jobs done" | tee -a "$LOG"
