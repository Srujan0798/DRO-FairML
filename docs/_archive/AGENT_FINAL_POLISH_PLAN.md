# Agent Final-Polish Plan & Prep Log

**Role:** Orchestrator's cleanup & docs agent (id 019ed657-bb1f-7663-ac56-35d4b810b1b6 per ORCHESTRATOR_LIVE_STATUS).

**Mission order (from user brief + REMAINING_WORK.md):**
1. Once lambda hits 72 and Lambda-Finisher has run finalize + committed:
   - Re-run full analysis if needed: `python3 experiments/analyze_tau1.py`
   - Rebuild report tables + PDFs
   - Update KULDEEP_DISCUSSION.md and ORCHESTRATOR_LIVE_STATUS.txt with "lambda 72/72 complete" + date

2. Once canonical has Credit + LSAC and empirical companion is running or done:
   - Run full wilcoxon from complete canonical (use or extend `compute_canonical_wilcoxon.py`)
   - Regenerate final figures using the complete datasets
   - Rebuild both `report.pdf` and `paper/main.pdf` from final data
   - Update all key docs (HANDOFF.md, DELIVERABLES_CHECKLIST, etc.)

3. Final cleanup:
   - Remove or archive unnecessary history files, .bak, old logs **if safe**
   - Ensure git status is clean except for the final results you intentionally commit
   - Make one final commit: "FINAL: lambda + canonical + empirical complete + all analysis + docs"
   - Produce a short "FINAL_EVIDENCE.txt" with last counts, key numbers (Adult α=0.2 DRO vs Naive DP/acc, constant 0.752 bar), commit hashes, and "All agents delivered".

**Hard Rules (enforced):**
- ONLY touch/edit/write/delete in: docs/, report/, paper/, figures/, and status files (root *.md, *.txt for status/HANDOFF/REMAINING etc.).
- NEVER touch/modify: experiments/ (except running analysis gens via cmd), src/, results/ (main json), logs/ main files. Do not rm outside allowed.
- Do not touch the running experiment processes (PIDs: 16334 lambda, 21531 canonical).
- ALWAYS verify counts (via json lens + dataset counters) BEFORE claiming complete in any doc/update.
- No new launches. Only analysis + rebuild + doc updates + final commit.
- Use evidence: spot-check numbers post-rebuild match CSVs (e.g. alpha=0.2 DRO DP ~0.2371).
- Git: stage only allowed changed files for the final commit.

## Prep Actions Completed (2026-06-17)
- Read REMAINING_WORK.md and ORCHESTRATOR_LIVE_STATUS.txt (start requirement).
- Polled counts multiple times (using json + finalize_experiments.py status + ps + log tails read-only): confirmed lambda=44/72, canonical=73/540 (all adult only).
- No Credit/LSAC rows yet, no empirical running.
- Verified git recent commits (no finisher finalize yet).
- Found .bak candidate: only `report/report.tex.bak` (inside allowed dir).
- Read key docs: HANDOFF.md, DELIVERABLES_CHECKLIST.txt, KULDEEP_DISCUSSION.md (full), FINAL_FIGURES_MANIFEST.txt, MASTER_PLAN.md excerpts, agent files.
- Inspected allowed dirs (docs/, report/, paper/, figures/).
- Read key analysis scripts (non-modify): analyze_tau1.py, finalize_experiments.py, compute_canonical_wilcoxon.py, generate_report_tables.py, generate_paper_tables.py, generate_all_figures.py , Makefile.
- Confirmed build tool: `/opt/homebrew/bin/tectonic` available.
- Updated status files with prep notes (ORCHESTRATOR_LIVE_STATUS.txt, REMAINING_WORK.md, DELIVERABLES_CHECKLIST.txt) + this plan doc.
- Created todo list internally (poll done, prepare in progress, waits pending).
- Key numbers extracted for future FINAL_EVIDENCE (from manifest + handoff):
  - Adult α=0.2 (tau=1, DP attack, from tau1_summary / manifest):
    - Naive: acc≈0.7528 , DP≈0.2480
    - DRO: acc≈0.7550 (>0.752 constant bar) , DP≈0.2371
    - Wins: 3/3 seeds
  - High-α: α≥0.3 acc <0.752 constant predictor baseline.
  - constant predictor acc bar = 0.752 (Adult)

## Exact Commands (to use only when conditions verified)
**Lambda 72 phase:**
```bash
python3 -c 'import json; print(len(json.load(open("results/lambda_lr_grid.json"))), "/72")'   # verify 72
python3 experiments/analyze_tau1.py
python3 experiments/generate_report_tables.py
(cd report && /opt/homebrew/bin/tectonic report.tex)
(cd paper && /opt/homebrew/bin/tectonic main.tex)
# then update docs/status only
```

**Canonical complete phase:**
```bash
# verify
python3 -c '
import json
from collections import Counter
c = json.load(open("results/canonical_tau1.json"))
print(len(c), "/540")
print(Counter(r["dataset"] for r in c))
'
python3 experiments/compute_canonical_wilcoxon.py
python3 experiments/analyze_tau1.py
python3 experiments/generate_all_figures.py   # or relevant plot_*.py + generate_*
python3 experiments/generate_report_tables.py
python3 experiments/generate_paper_tables.py
(cd report && /opt/homebrew/bin/tectonic report.tex)
(cd paper && /opt/homebrew/bin/tectonic main.tex)
```

**Final commit (after updates + cleanup limited):**
```bash
git status
# stage only allowed: git add docs/ report/ paper/ figures/ *.md *.txt (status)
git commit -m "FINAL: lambda + canonical + empirical complete + all analysis + docs"
```

**Cleanup (safe only):**
- rm report/report.tex.bak   (or mv docs/_archive/report.tex.bak.2026-06-17 )
- Check no other .bak in report/ paper/ figures/ docs/ (use ls/find restricted)
- Archive old if any in docs/_archive/ only.
- Leave results/ history , logs/ untouched.

**Post rebuild verification:**
- Re-poll counts.
- Spot check e.g. grep in report/report.pdf or use pdftotext for "0.237" "0.752" match to CSVs.
- Update FINAL_FIGURES_MANIFEST.txt if regenerated.
- Append to KULDEEP etc with date + "verified counts: L=72 C=540"

## Wait/ Poll Protocol
- Repeat polls: counts, ps (no kill), git log -5, log tail (non mod), finalize status.
- Signals from others: new commits with "finalize" or "lambda 72", new rows in canonical with Credit, presence of empirical log/json.
- If during session thresholds hit: execute immediately in order, re-poll to confirm.
- Do not claim complete until verified.

## Risks & Mitigations
- Partial data: always filter n_seeds >=2 or full in wilcoxon/analyze (scripts handle).
- Script side effects: analyze may write results/ csvs (unavoidable per mission but figures/report are allowed output).
- Tectonic fails: capture output, ensure .tex clean.
- Git dirty: only commit the intentional final on allowed changes.
- Stale numbers: always re-run generators from complete data; spot-check.

**Status:** Prep complete. Waiting for signals. Will poll and act when lambda==72 (and finisher done) first, then canonical+emp.

Prepared: 2026-06-17. Agent Final-Polish.
All per rules.
