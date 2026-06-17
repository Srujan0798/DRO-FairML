# REMAINING WORK + CURRENT AGENT ASSIGNMENTS
Date: 2026-06-17

## Live Experiments (already launched, do not duplicate)
- lambda grid: 44/72 , PID 16334
- canonical: 73/540 (Adult only so far), PID 21531 (--k_inner 10)

## Assigned Sub-Agents (orchestrator spawned)

1. **Agent Lambda-Finisher** (subagent id: 019ed657-8b95-72a3-9207-b7a8c4707d0b)
   - Watch lambda grid
   - When ==72: run finalize_experiments.py, commit, update status files
   - Evidence required on completion

2. **Agent Canonical-Advancer** (subagent id: 019ed657-8b95-72a3-9207-b7b35c547c3b)
   - Watch canonical
   - Launch empirical companion (with --k_inner 10 --radii_mode empirical) ONLY after first Credit or LSAC row appears
   - Trigger analyze + tables when appropriate

3. **Agent Final-Polish** (subagent id: to be confirmed on spawn)
   - Wait for the above two to signal completion
   - Full final wilcoxon + figures + report/PDF rebuild
   - Docs update + cleanup + final commit + FINAL_EVIDENCE.txt

## Manual / User (you) can also run
- python3 finalize_experiments.py status
- python3 watch_and_finalize.py
- Check counts any time

## Still pending / low priority
- UTKFace (email draft ready, access blocked)

All work must respect MASTER_PLAN §0/§1 (one writer, ps checks, provenance, tau=1/k=10, adversarial only).
