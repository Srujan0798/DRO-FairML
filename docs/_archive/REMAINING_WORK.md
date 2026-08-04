# REMAINING WORK (as of 2026-06-17 ~21:xx)

Current:
- lambda: 45/72 (PID 16334 still running, \alpha0.1 at 17/18 + progressing)
- canonical: 73/540 (PID 21531 with --k_inner 10, still finishing Adult)

## High Priority (CPU-bound, processes already running)
1. Finish lambda grid to 72/72
2. Finish canonical through Adult \to Credit + LSAC (6 seeds)

## When lambda hits 72 (do immediately)
- python3 finalize_experiments.py
- Commit results/lambda_lr_grid.json + any new history + updated analysis
- Rebuild figures if needed from full grid

## When canonical has Credit or LSAC rows
- Launch empirical companion:
  nohup python3 experiments/run_canonical.py --k_inner 10 --radii_mode empirical >> logs/empirical.log 2>&1 &
- Then analyze + tables

## Final deliverables (after both grids complete)
- Full n=6 Wilcoxon from complete canonical_tau1.json
- Regenerate all final figures + report/PDFs
- Update KULDEEP_DISCUSSION.md, report, HANDOFF.md
- Cleanup: remove .bak, temp history files if not needed, git clean
- Final commit + evidence package

## Low priority / blocked
- UTKFace: Email draft ready (EMAIL_TO_SUPIN_GOPI_DRAFT.txt). Blocked on flair2 account.

## Active Tooling
- python3 finalize_experiments.py status
- python3 watch_and_finalize.py   (can be left running)
- Clean monitors (if active in session)

## AGENT FINAL-POLISH PREP NOTE (2026-06-17)
Agent Final-Polish (orchestrator cleanup/docs) has read initial files and is monitoring.
- Verified live counts via polls: lambda 45/72, canonical 73/540 (Adult only). 
- Plan prepared per mission order (see ORCHESTRATOR_LIVE_STATUS.txt for full detailed plan).
- Will act ONLY when Lambda-Finisher signals via finalize+commit at 72/72, THEN Canonical-Advancer for Credit/LSAC + empirical.
- Strict: verify counts + git signals before EVERY "complete" claim or edit.
- Will ONLY modify: docs/ root *.md/*.txt (per current hygiene constraints) + remove obvious junk.
- Will run generators only (analyze_tau1.py, compute_canonical_wilcoxon.py, generate_*tables.py, tectonic for PDFs) when conditions met.
- No interference with procs. No empirical launch (reserved).
- Cleanup limited to obvious temp/.bak (report/report.tex.bak already gone; no other bak in allowed areas; python caches + latex build temps removed).
- Final: one commit as specified + FINAL_EVIDENCE.txt .
Current state: lambda 45/72, canonical 73/540. Waiting/polling for finishers. Last hygiene pass: 2026-06-17.
