# Remaining Work After Agent Final-Polish Prep (2026-06-17)

## Current Experiment State
- lambda grid: 44/72 (PID 16334 running)
- canonical: 73/540 (PID 21531 running, still Adult only)

## Explicit Remaining Items (not yet executable)
1. **Lambda completion** (44 → 72)
   - Agent Lambda-Finisher (019ed657-8b95-72a3-9207-b7a8c4707d0b) is assigned to watch + run finalize_experiments.py + commit when it hits 72.

2. **Canonical completion** (73 → 540 + Credit + LSAC)
   - Agent Canonical-Advancer (019ed657-8b95-72a3-9207-b7b35c547c3b) is assigned to watch + launch empirical companion (only after Credit/LSAC rows appear).

3. **Post-data execution** (only after 1 and 2 above succeed)
   - Agent Final-Polish (019ed657-bb1f-7663-ac56-35d4b810b1b6) has **completed its preparation phase**:
     - Full plan written: docs/AGENT_FINAL_POLISH_PLAN.md
     - Status files pre-updated
     - Will execute the full final steps (wilcoxon, figures, PDFs, docs, cleanup, FINAL commit + evidence) once the other two agents signal completion.

## Low Priority / Blocked
- UTKFace (email draft ready, access still blocked)

## Ready Commands
- python3 finalize_experiments.py status
- python3 watch_and_finalize.py

All three agents are now live and respecting one-writer + MASTER_PLAN rules.
