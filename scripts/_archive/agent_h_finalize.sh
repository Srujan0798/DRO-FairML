#!/usr/bin/env bash
# Agent H — one-shot finalization when the canonical grid hits 540 rows.
#
# Preconditions (hard fail if not met):
#   - total rows in results/canonical_tau1.json == 540
#   - attack=='if' rows == 180
#   - max |if_clean| among IF-attack rows >> 1e-6  (non-degenerate metric)
#
# Then runs the deterministic regeneration pipeline:
#   wilcoxon → report tables → make results → deliverables → paper → report
#   + writes results/if_wilcoxon_summary.txt (same logic as monitor_if_then_regen.sh)
#
# Does NOT commit or push (judgment / prose pass is separate).
# Logs every step to results/FINALIZATION_LOG.txt
#
# Usage:
#   ./scripts/agent_h_finalize.sh           # full run (exits 1 if not ready)
#   ./scripts/agent_h_finalize.sh --check   # verification only (dry-run gate)
#   ./scripts/agent_h_finalize.sh --wait    # POLL until gate ready, then finalize
#                                           # NEVER runs experiments / never writes canonical
#
# DATA SAFETY: only the IF sweep may write results/canonical_tau1.json until 540.
# This script is read-only on that file (load + derive tables/figures elsewhere).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CANONICAL="results/canonical_tau1.json"
LOG="results/FINALIZATION_LOG.txt"
CHECK_ONLY=0
WAIT_MODE=0
case "${1:-}" in
  --check|--dry-run) CHECK_ONLY=1 ;;
  --wait|--poll)     WAIT_MODE=1 ;;
esac

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() {
  local msg="$*"
  echo "[$(ts)] $msg"
  mkdir -p results
  echo "[$(ts)] $msg" >>"$LOG"
}

step() {
  # step <name> -- <command...>
  local name="$1"; shift
  local rc=0
  log "STEP START: $name"
  if "$@"; then
    log "STEP OK:    $name"
    return 0
  fi
  rc=$?
  log "STEP FAIL:  $name (exit $rc)"
  return "$rc"
}

# ---------------------------------------------------------------------------
# Verification (must pass before any regeneration)
# ---------------------------------------------------------------------------
verify_gate() {
  python3 - <<'PY'
import json, sys
from collections import Counter

path = "results/canonical_tau1.json"
try:
    d = json.load(open(path))
except Exception as e:
    print(f"VERIFY FAIL: cannot load {path}: {e}")
    sys.exit(2)

total = len(d)
counts = Counter(r.get("attack") for r in d)
n_if = counts.get("if", 0)
n_dp = counts.get("dp", 0)
n_comb = counts.get("combined", 0)

if_rows = [r for r in d if r.get("attack") == "if"]
if if_rows:
    max_if = max(abs(float(r.get("if_clean", 0.0) or 0.0)) for r in if_rows)
else:
    max_if = 0.0

print(f"total={total}  dp={n_dp}  combined={n_comb}  if={n_if}")
print(f"max|if_clean| (attack=if) = {max_if:.6e}")

ok = True
if total != 540:
    print(f"FAIL: total={total} (need 540)")
    ok = False
else:
    print("OK: total==540")

if n_if != 180:
    print(f"FAIL: if={n_if} (need 180)")
    ok = False
else:
    print("OK: if==180")

# Non-degenerate: must be clearly above 1e-6
threshold = 1e-6
if max_if <= threshold:
    print(f"FAIL: max|if_clean|={max_if:.3e} not >> 1e-6 (degenerate or missing)")
    ok = False
else:
    print(f"OK: max|if_clean|={max_if:.6e} >> 1e-6")

if not ok:
    print("GATE: NOT READY")
    sys.exit(1)
print("GATE: READY")
sys.exit(0)
PY
}

# ---------------------------------------------------------------------------
# --wait: poll only (read-only). No experiments. No writes to canonical.
# ---------------------------------------------------------------------------
if [ "$WAIT_MODE" -eq 1 ]; then
  echo "Agent H --wait: polling $CANONICAL until 540/180 (read-only; no data generation)"
  for i in $(seq 1 120); do
    if verify_gate; then
      echo "GATE READY after poll #$i — starting finalize (still no experiment runs)"
      WAIT_MODE=0
      break
    fi
    # sole-writer health (informational)
    if ! ps -p 10146 >/dev/null 2>&1; then
      # sweep may have exited; re-check gate once more after short sleep
      sleep 5
      if verify_gate; then
        echo "Sweep gone and gate READY"
        WAIT_MODE=0
        break
      fi
      echo "WARN: sweep pid 10146 not running and gate not ready — will keep polling (maybe atomic write in flight)"
    fi
    sleep 30
  done
  if [ "$WAIT_MODE" -eq 1 ]; then
    echo "TIMEOUT: gate never passed; NOT running finalize"
    exit 3
  fi
fi

# Start a fresh log for this invocation
mkdir -p results
{
  echo "========================================"
  echo "Agent H finalization — $(ts)"
  echo "cwd=$ROOT  mode=$([ "$CHECK_ONLY" -eq 1 ] && echo check || echo full)"
  echo "DATA SAFETY: this script never writes $CANONICAL"
  echo "========================================"
} >"$LOG"

log "Running verification gate on $CANONICAL"
if ! verify_gate | tee -a "$LOG"; then
  log "VERIFICATION FAILED — aborting (no regeneration)"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "(--check mode: expected failure until grid is complete)"
  fi
  exit 1
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  log "CHECK-ONLY: gate passed; not regenerating"
  exit 0
fi

# ---------------------------------------------------------------------------
# Regeneration pipeline
# Prefer venv python if present
# ---------------------------------------------------------------------------
if [ -x "venv/bin/python3" ]; then
  PY="venv/bin/python3"
else
  PY="python3"
fi
log "Using python: $PY ($($PY --version 2>&1))"

FAIL=0

{
  step "compute_canonical_wilcoxon" $PY experiments/compute_canonical_wilcoxon.py || FAIL=1
  step "generate_report_tables"     $PY experiments/generate_report_tables.py     || FAIL=1
  step "make results"               make results       || FAIL=1
  step "make deliverables"          make deliverables  || FAIL=1
  step "make paper"                 make paper         || FAIL=1
  step "make report"                make report        || FAIL=1
} >>"$LOG" 2>&1

# Mirror last few log lines to console for visibility
tail -n 20 "$LOG" || true

# IF Wilcoxon summary — same logic as scripts/monitor_if_then_regen.sh
log "STEP START: if_wilcoxon_summary"
if $PY - >>"$LOG" 2>&1 <<'PY'
import json, collections
from scipy.stats import wilcoxon
d = json.load(open("results/canonical_tau1.json"))
g = collections.defaultdict(dict)
for r in d:
    g[(r["dataset"], r["attack"], r["alpha"], r["seed"])][r["method"]] = r
out = [
    "Real IF-attack Wilcoxon (DRO better on DP, one-sided), n=6, from canonical_tau1.json",
    "Also shows IF-violation means so the IF-vs-DP tradeoff is visible.",
    "",
]
for ds in ["adult", "credit", "lsac"]:
    for a in [0.0, 0.1, 0.2, 0.3, 0.4]:
        pr = [
            (v["naive"], v["dro"])
            for k, v in g.items()
            if k[:3] == (ds, "if", a) and "naive" in v and "dro" in v
        ]
        if len(pr) < 2:
            continue
        nv = [p[0]["dp_clean"] for p in pr]
        dv = [p[1]["dp_clean"] for p in pr]
        ni = [p[0]["if_clean"] for p in pr]
        di = [p[1]["if_clean"] for p in pr]
        wins = sum(1 for n, x in zip(nv, dv) if n > x)
        try:
            p = wilcoxon(nv, dv, alternative="greater").pvalue
        except Exception:
            p = float("nan")
        out.append(
            f"{ds:6} if a={a}: DP n={sum(nv)/len(nv):.4f} d={sum(dv)/len(dv):.4f} "
            f"wins={wins}/{len(pr)} p={p:.4f} | IF n={sum(ni)/len(ni):.4f} d={sum(di)/len(di):.4f}"
        )
open("results/if_wilcoxon_summary.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
PY
then
  log "STEP OK:    if_wilcoxon_summary → results/if_wilcoxon_summary.txt"
else
  log "STEP FAIL:  if_wilcoxon_summary"
  FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
  log "FINALIZATION COMPLETE — all steps OK (NOT committed; judgment pass is separate)"
  exit 0
else
  log "FINALIZATION FINISHED WITH FAILURES — inspect $LOG (NOT committed)"
  exit 1
fi
