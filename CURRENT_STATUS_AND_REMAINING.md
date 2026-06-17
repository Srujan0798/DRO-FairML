# DRO-FairML — Current Status & Remaining Work
Date: 2026-06-17 (late)

## Live Experiments
- lambda grid: 44/72 (PID 16334 still running)
  - α=0.1: 17/18
  - α=0.2: 18/18 (done)
  - α=0.3: 9/18
- canonical: 73/540 (PID 21531 running, still Adult only)

## Agents Status
1. Agent Lambda-Finisher (019ed657-8b95-72a3-9207-b7a8c4707d0b) — watching lambda, will finalize at 72
2. Agent Canonical-Advancer (019ed657-8b95-72a3-9207-b7b35c547c3b) — watching canonical, will launch empirical when Credit/LSAC appears
3. Agent Final-Polish (019ed657-bb1f-7663-ac56-35d4b810b1b6) — prep complete (plan + status updates done). Will execute full final steps when data ready.

## Remaining Work (to be driven to completion)
1. Finish lambda 44 → 72 (process running)
2. Finish canonical 73 → 540 + Credit + LSAC (process running)
3. When lambda=72: run finalize_experiments.py + commit
4. When canonical has Credit/LSAC: launch empirical
5. Full final deliverables (wilcoxon n=6, final figures, report/PDFs, docs, cleanup, final commit + evidence)
6. UTKFace (low priority, blocked)

All agents are assigned and will react when conditions are met.
