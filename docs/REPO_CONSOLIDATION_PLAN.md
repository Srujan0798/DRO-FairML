# DRO-FairML — Repo Consolidation Plan (Agent J, Phase 2)

**Date:** 2026-08-04  
**Depends on:** `docs/REPO_AUDIT.md` (Phase 1 inventory)  
**Invariants (never violate):**
1. Do **not** modify scientific content of `results/canonical_tau1.json` (or other result numbers/CSVs produced from it by Agent H).
2. `python3 -m pytest tests/ -q` must stay green after **every** stage.
3. No silent fallbacks to stale/contaminated result files.
4. Prefer **archive** (`experiments/_archive/`, `docs/_archive/`, `scripts/_archive/`) over hard delete until Aug 10 ships.
5. Commit in small logical chunks; message style: `chore: ...` / `fix: ...` / `docs: ...`.

---

## Goals (Definition of Done)

- [ ] One canonical **tabular** pipeline documented in README: `run_canonical` / `run_if_parallel` → `loaders` → wilcoxon → tables → figures → paper/report.
- [ ] One **image** runner entry (UTKFace modes); synthetic only if explicitly flagged, never default for reported rows.
- [ ] Single figure/table/stats module set; old generators archived.
- [ ] Makefile targets all real and documented in `make help`.
- [ ] `get_temperature` remains only in `src/temperature.py` (active tree).
- [ ] Root free of stray logs / auto_pipeline noise.
- [ ] Docs: README + STATUS + design/findings stay; agent prompts archived.

---

## Stage 0 — Preflight (read-only + safety)

**Time:** ~15 min  

1. Confirm `pytest tests/ -q` green; record pass count.
2. Confirm `results/canonical_tau1.json` row count and attack histogram (do not edit).
3. Confirm no Agent H regen currently writing figures/tables (or wait for idle).
4. Snapshot: `git status` clean of unrelated WIP, or stash carefully.

**Exit:** green tests; written note of baseline row count; no file moves yet.

---

## Stage 1 — Kill silent fallbacks (code-only, no moves)

**Risk:** Medium (behavior change for broken scripts only)  
**Tests:** run after each file touch.

### 1.1 Fix critical readers

| File | Change |
|------|--------|
| `experiments/meeting_summary.py` | Load **only** `load_canonical_tau1()`; delete K_inner=5 bak and tau_ablation preference; fail if missing. |
| `experiments/generate_all_deliverables.py` | Repoint all inputs to `results/canonical_tau1.json` / live wilcoxon CSV / live lambda only if present; **fail loud** if required file missing. Prefer reusing `generate_final_figures.py` logic over stale_archived paths. |
| `experiments/validate_results.py` | Gate on `load_canonical_tau1()` + Wilcoxon (not nested `all_results.json`). |
| `experiments/generate_results.py` | Either call `canonical_to_all_results` then figures **or** read flat canonical; never return empty list silently without error when used from Makefile. |
| `experiments/compute_canonical_wilcoxon.py` | Remove dead `TAU1_FALLBACK` constant; update docstring. |
| `configs/default.yaml` | Set `tau: 1.0`, align `K_inner`/`k` comments with canonical (do not change trainer defaults that tests pin unless tests still pass). |

### 1.2 UTKFace synthetic

| File | Change |
|------|--------|
| `experiments/run_utkface.py` | Add `--allow-synthetic` flag (default **False**). Without flag or real data → `SystemExit` with clear message. Keep synthetic for smoke tests only when flag set; always tag `dname='UTKFace (synthetic)'`. |

### 1.3 Broken callers that always raise

Either delete call to `load_fairness_pgd_results` and retarget to canonical, **or** archive the whole script in Stage 3:

- `generate_all_figures.py`
- `plot_partial_results.py`
- `generate_meeting_table.py`
- `generate_sensitivity_analysis.py`

**Exit criteria:**
- No active experiment script prefers `tau_ablation*` over canonical for headline numbers.
- `make deliverables` does not read `results/stale_archived/` (or is temporarily disabled with a loud error pointing to new target).
- `pytest` green.

**Commit:** `fix: fail-loud loaders; remove tau_ablation / stale_archived fallbacks`

---

## Stage 2 — Makefile + main.py rewire

**Risk:** Low–medium  

### 2.1 Makefile target matrix

| Target | New behavior |
|--------|----------------|
| `test` | unchanged pytest |
| `validate` | `python3 experiments/validate_results.py` (canonical-based after 1.1) |
| `wilcoxon` | `python3 experiments/compute_canonical_wilcoxon.py` (**new**) |
| `tables` | `python3 experiments/generate_report_tables.py` (**new**) |
| `results` | chain: `canonical_to_all_results` (if still needed) + figure entry **or** single figures module |
| `deliverables` | non-stale figD / final figures only |
| `paper` / `report` | tectonic (unchanged) |
| `theory` | unchanged |
| `experiments` | print deprecation + point to `run_canonical.py` **or** invoke run_canonical with dry-run help — **do not** run 10-seed legacy suite by default |
| `full` | document “regenerate artifacts from existing canonical” (tables+figures+paper+report+validate) — **not** re-train 540 |
| `monitor` | print row counts from `canonical_tau1.json` (attacks, datasets) or remove |
| `review` | remove or point to `docs/REPO_AUDIT.md` + future VERIFICATION_REPORT |
| `help` | update to match |

### 2.2 `main.py`

- Prefer: thin CLI that documents recommended entrypoints.
- `--generate-results` must not silently use incomplete nested JSON.
- `--run-experiments` / `--full-pipeline`: either remove, or hard-error with “use experiments/run_canonical.py”.

**Exit:** `make help`, `make test`, `make validate`, `make paper`, `make report` succeed without touching result JSON content.

**Commit:** `chore: rewire Makefile and main.py to canonical pipeline`

---

## Stage 3 — Archive experiment near-duplicates (moves only)

**Risk:** Low if imports/Makefile already retargeted  
**Order:** archive **after** Stage 1–2 so nothing live imports archived paths.

### 3.1 Create / refresh `experiments/_archive/README.md`

One-line description per archived script (source: REPO_AUDIT §3).

### 3.2 Move in batches (one commit per batch if large)

**Batch A — UTKFace extras → archive** (keep one active runner):
- Keep: `run_utkface.py` (after Stage 1.2) **or** promote `run_utkface_server.py` as the single entry and archive the rest.
- Archive: `run_utkface_extended.py`, `run_utkface_pixel_pgd.py`, `run_utkface_randinit.py` (or fold modes first then archive empty shells).
- Archive: `analyze_utkface.py`, `analyze_utkface_stats.py`, `analyze_dro_failure.py`, `generate_fig10.py`, `setup_celeba.py`, `setup_fairface.py`.

**Batch B — Superseded figure generators:**
- Archive: `generate_all_figures.py`, `generate_pdf_report.py`, `generate_summary_dashboard.py`, `generate_sensitivity_analysis.py`, `generate_meeting_table.py`, `plot_partial_results.py`, older `plot_high_alpha_tau.py` if unused by Makefile, meeting-only plots not referenced by paper.

**Batch C — Ablation / legacy runners:**
- Archive: `run_experiments.py`, `run_robust.py`, `run_ablations.py`, `run_tau_ablation.py`, `run_knn_ablation.py`, `run_lambda_*.py`, `run_random_vs_adversarial.py`, `run_parallel_batch.py`, `run_k10_targeted.py`, `run_single_adult.py`, related summarize/analyze one-offs.

**Batch D — Keep list (do not move):**
- `run_fairness_pgd.py`, `run_canonical.py`, `run_if_parallel.py`, `_run_if_chunk.py`, `loaders.py`, `compute_canonical_wilcoxon.py`, `generate_report_tables.py`, `canonical_to_all_results.py` (until figures merged), `validate_results.py`, `verify_theory.py`, primary figure entry chosen in Stage 4, `run_canonical_empirical.py` if Q5 appendix remains.

After each batch: `pytest`; `make validate` (if applicable).

**Commit:** `chore: archive experiment near-duplicates (batch A/B/C)`

---

## Stage 4 — Merge figure/stats modules

**Risk:** Medium (import paths, figure stem names)

1. Choose **one** implementation spine:
   - Preferred: extend `generate_final_figures.py` + `generate_figures.py` → single `experiments/figures.py` (or package).
   - Stats: `compute_canonical_wilcoxon.py` + summary CSVs → `experiments/stats.py` (optional rename later).
2. Keep **output figure filenames** stable if paper/report include them (grep `figures/` from `.tex`).
3. Update Makefile `results` / `deliverables` to call the merged entry.
4. Archive superseded generators once Makefile no longer references them.
5. Optionally drop `canonical_to_all_results` if no consumer needs nested schema.

**Exit:** regenerating figures from canonical does not change scientific tables; stems required by TeX exist.

**Commit:** `chore: unify figure and stats generators on canonical_tau1`

---

## Stage 5 — Scripts + root hygiene

**Risk:** Low  

1. Create `scripts/_archive/`; move finished watchers/orchestrators (REPO_AUDIT §4).
2. Keep: `extract_utkface_features.py`, `run_if_rerun_cluster.sh`, and at most one finalize/monitor script if still useful.
3. Move root `*.log`, `auto_pipeline*.py` → `logs/` or `scripts/_archive/`.
4. Move `conv.md` → `docs/chat/` or `docs/_archive/`.
5. Untrack or move `paper/ICML_submission.pdf` if still tracked (Agent F recommendation).

**Commit:** `chore: archive one-shot scripts and root strays`

---

## Stage 6 — Docs single source of truth

**Risk:** Low  

1. Archive to `docs/_archive/`: `AGENT_PROMPTS.md`, `AGENT_PROMPTS_REMAINING.md`, `AGENT_FINAL_POLISH_PLAN.md`, outdated checklists as desired.
2. Annotate or archive `FINDING_DRO_FAILS_ON_ADULT.md` with header: superseded by τ=1 canonical finding.
3. Update `README.md`:
   - How to reproduce (download data → tests → validate → tables/figures → paper/report).
   - Point to STATUS, not dead HANDOFF links as primary.
4. Update `docs/PROJECT_FLOW.md` to match new tree.
5. Keep: STATUS, MASTER_DISPATCH, FINAL_COMPLETION_PLAN, LSAC_DEGENERACY, KULDEEP_CORRECTION, design docs, UTKFACE_PIPELINE, SERVER_RUNBOOK.

**Commit:** `docs: single source of truth; archive agent prompt noise`

---

## Stage 7 — Reproducibility polish

**Risk:** Low  

1. Verify `data/download_data.sh` works on clean paths for Adult/Credit/LSAC.
2. Pin `requirements.txt` versions if not already (minimal pins that still pass tests).
3. Confirm `make full` (artifact regen) works without GPU and without re-running 540 trainings.
4. Optional: export `CNNClassifier` from `src/models/__init__.py` if UTKFace kept.

**Commit:** `chore: reproducibility (download, requirements, make full)`

---

## Stage 8 — Final verification gate

1. `pytest tests/ -q`
2. `make validate`
3. `make tables && make wilcoxon` (or equivalent)
4. `make results` / `make deliverables` (no stale_archived reads — grep the invoked scripts)
5. `make paper && make report`
6. Grep active tree:
   - `def get_temperature` → only `src/temperature.py`
   - `stale_archived` → only comments/docs or archive scripts
   - `tau_ablation_tau1_KINNER5` → zero in active experiments
   - `_make_synthetic_utkface` only behind explicit flag
7. Confirm `results/canonical_tau1.json` byte-identical (or row-count + checksum) to pre-consolidation scientific content.

**Commit only if needed:** `chore: post-consolidation verification`

---

## What NOT to do

- Do not re-run the 540-row grid as part of consolidation.
- Do not “fix” LSAC/IF numbers or re-tune λ.
- Do not merge `docs/_archive/submission_may2026` back into live tree.
- Do not delete `results/stale_archived/` until Agent L signs off (space cleanup optional post-submission).
- Do not invent a second temperature helper.

---

## Suggested commit sequence (summary)

1. `fix: fail-loud loaders; remove tau_ablation / stale_archived fallbacks`  
2. `chore: rewire Makefile and main.py to canonical pipeline`  
3. `chore: archive experiment near-duplicates (UTKFace / figures / ablations)`  
4. `chore: unify figure and stats generators on canonical_tau1`  
5. `chore: archive one-shot scripts and root strays`  
6. `docs: single source of truth; archive agent prompt noise`  
7. `chore: reproducibility (download, requirements, make full)`  

---

## Parallelism notes

| Work | Parallel-safe with |
|------|-------------------|
| Stage 1–2 (code fix) | Agent M (UTKFace data) if no shared runner edits |
| Stage 3 archive moves | After 1–2; avoid while H regenerates figures |
| Stage 4 figure merge | Coordinate with Agent K (paper figure includes) |
| Stage 6 docs | Coordinate STATUS updates with H/I |

---

## Rollback

Every stage is git-revertsafe. Prefer archive moves over `git rm` so recovery is `git mv` back. If pytest fails mid-stage: revert the stage commit before continuing.

---

*Phase 2 may begin only after this plan is reviewed against REPO_AUDIT.md and Agent H has finished any in-flight artifact regen, or with explicit coordination on non-overlapping paths.*
