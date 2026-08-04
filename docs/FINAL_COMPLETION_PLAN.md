# DRO-FairML — Final Completion Plan

**Written:** 2026-08-04 (Tue) · **Author:** verified against `results/canonical_tau1.json` on the new machine (Mac16,7, 14 CPU, MPS GPU).

## Two hard deadlines
1. **TODAY, 4:00–5:00 pm** — weekly meeting with Manisha / Kuldeep (`meet.google.com/rop-zyah-wmn`). Goal: present the honest, complete tabular story with **real IF results for the first time**.
2. **Mon Aug 10** — final submission. Goal: the whole project at professional quality — one clean pipeline, every claim traceable, paper/report publication-ready, reproducible from a clean clone. No "vibe-coding" leftovers.

---

## 0. Where we actually are (verified today)

| Thing | State |
|---|---|
| Canonical grid | **383/540 rows** in `results/canonical_tau1.json` — DP 180 ✓, Combined 180 ✓, **IF 23→completing now** |
| IF metric | **FIXED** (cosine-based). Real signal confirmed: IFmax ≈ 0.098, per-config IF ≈ 0.025–0.033. The old ~1e-10 degeneracy is gone. |
| **IF sweep** | **RUNNING NOW** — `experiments/run_if_parallel.py` (pid 10146, 10 workers, ~21 s/config). 157 configs → **ETA ~20–40 min** to full 540. Resume-safe, atomic writes, logs to `logs/if_parallel.log`. |
| Tests | 62 passing (last verified pre-migration) |
| Git | clean tree (only `conv.md` untracked); `main` in sync with `origin/main`; HEAD `835e4b8` |
| Hardware | **The old blocker is gone.** IF is no longer cluster-blocked — it completes locally in minutes. |

### The verified science (unchanged, still true)
- **Adult / DP:** DRO wins every α, p ≤ 0.031 (n=6). **Adult / Combined:** 6/6 every α, p=0.016.
- **Credit:** all attacks, ~14/15 cells p<0.05.
- **LSAC / Combined:** genuine win, p=0.016 at α=0.1/0.3/0.4.
- **LSAC / DP:** honest **negative** — DRO loses 0/6 at every α; the model collapses to the constant predictor (see `docs/LSAC_DEGENERACY.md`).
- **Defensible regime: α ≤ 0.2.** At α ≥ 0.3 both methods fall below the constant-predictor baseline (Adult 0.752 / Credit 0.779 / LSAC 0.902).
- **Central finding:** the earlier "DRO is fragile" was a τ=100 temperature artifact; fixed τ=1 makes DRO robust.
- **New, landing today:** the first real IF-attack results across all three datasets.

### What Kuldeep will check
On Jun 30 he said: *"After drafting the reply, verify all the claims. Sometimes AI makes claims just to make results appear correct."* Every number below must trace to `canonical_tau1.json`. If it can't be traced, it doesn't ship.

---

## MILESTONE 1 — before 4 pm today (~2.5 h)

The IF data completes on its own in ~30 min. Two agents then take it to a meeting-ready state. These are **sequential** (H before I).

### Agent H — Finalize the full 540-row grid and reconcile every claim
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/FINAL_COMPLETION_PLAN.md and
docs/MASTER_DISPATCH.md first.

A parallel job (experiments/run_if_parallel.py, pid 10146) is finishing the IF-attack
third of the canonical grid. WAIT for it to reach 540 rows before starting:
  watch: python3 -c "import json;d=json.load(open('results/canonical_tau1.json'));import collections;print(len(d),dict(collections.Counter(r['attack'] for r in d)))"
  proceed only when total=540 and if=180. Confirm IF is non-degenerate:
  python3 -c "import json;d=json.load(open('results/canonical_tau1.json'));print(max(abs(r['if_clean']) for r in d if r['attack']=='if'))"  # must be >>1e-6

THEN:
1. Regenerate ALL downstream artifacts from the full 540 rows (NOT from any stale
   intermediate): Wilcoxon over all 45 cells (experiments/compute_canonical_wilcoxon.py),
   summary CSVs, tables (experiments/generate_report_tables.py — must read
   canonical_tau1.json directly), figures (make results / make deliverables), and both
   PDFs (make paper && make report).
2. Compute the REAL IF Wilcoxon table and write it to results/if_wilcoxon_summary.txt.
   Report every IF cell honestly, including any where DRO loses. Do NOT tune anything.
3. Reconcile the prose that currently says "IF pending": update STATUS.md §3, the IF
   section of docs/KULDEEP_CORRECTION.md, and the paper/report IF sections
   (paper/sections/conclusion.tex, results.tex; report/report.tex) to the actual IF
   numbers. If IF confirms the DP story (DRO better at α≤0.2), say so; if it shows a
   mixed/negative picture (as the partial data hinted for LSAC/IF and Adult/IF at α=0.3),
   report that plainly.
4. Verify: grep the .tex tables — no "0.0000 ± 0.0000" IF cells remain; every dataset
   present; both PDFs build. Run `python3 -m pytest tests/ -q` (must stay 62 passing).
5. Commit ("data: complete 540-row grid incl. real IF; regenerate all artifacts") and
   push. Report the real IF Wilcoxon numbers in your summary.
```

### Agent I — Produce today's meeting brief (needs H)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/FINAL_COMPLETION_PLAN.md.
Agent H has just regenerated everything from the full 540-row grid.

Produce docs/MEETING_2026-08-04.md — a one-screen, honest update for the 4 pm meeting:
- Headline: fixed τ=1 makes DRO robust on Adult & Credit at α≤0.2 under all three attacks
  (n=6, p<0.05); the earlier "DRO fragile" was a τ=100 artifact.
- The FIRST real IF results (from H) — state exactly what they show, per dataset.
- The honest negatives: LSAC/DP degenerate; α≥0.3 below constant predictor.
- Status: tabular grid 100% complete (540 rows); what remains for Aug 10.
- A short "figures to share" list (absolute paths to the specific PDFs to drop in chat):
  the τ=1 headline, win-curves, Wilcoxon table, and the new IF figures.
Every number must come from results/canonical_tau1.json or the regenerated CSVs. No spin.
Also fold the three corrections from docs/KULDEEP_CORRECTION.md in briefly (IF was
mislabelled before, LSAC was reported pending, α≥0.3 caveat) so nothing is hidden.
Commit it.
```

---

## MILESTONE 2 — by Aug 10 (professional-grade)

Four agents, mostly parallel. J is the big "make it professional, not vibe-coding" job.
K depends on H's final numbers. L is the independent check that gates submission.

### Agent J — Repo professionalization & consolidation (the "not vibe coding" job)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/FINAL_COMPLETION_PLAN.md and
docs/MASTER_DISPATCH.md §1 Agent F (already-done cleanup) first, so you don't redo it.

GOAL: take the repo from research-scratch to professional quality. Read EVERY file
(src/, experiments/, scripts/, tests/, docs/, paper/, report/, Makefile, configs).
Produce docs/REPO_AUDIT.md (a complete inventory: every module, its purpose, keep/merge/
delete) BEFORE changing anything.

Then consolidate:
1. ONE canonical pipeline. experiments/ has ~80 files with heavy duplication (5 UTKFace
   near-duplicates, wrapper runners, one-off analysis scripts, superseded generators).
   Collapse to a small set: one experiment runner, one stats module, one figure module,
   one table module. Move everything else to experiments/_archive/ with a README saying
   what each was. get_temperature is centralized in src/temperature.py — ensure NOTHING
   redefines it (MASTER_DISPATCH flagged 9 copies; verify 0 remain outside src/).
2. Kill dead code and broken fallbacks. Every results loader must fail loudly on missing/
   contaminated data (no silent fallback to stale files). src/corruption/__init__.py must
   export FairnessTargetedPGD. Remove unused: image_pgd.py if still unused, get_run_config,
   duplicate _project_*_weights, etc.
3. Single source of truth for docs. Keep: README.md, STATUS.md, and the canonical
   design/finding docs. Merge/archive the rest under docs/_archive/ (one archive dir, not
   two). Every root-level stray (conv.md, old prompts) → archived or removed.
4. Reproducibility: add data/download_data.sh (the tabular datasets are public — Adult UCI,
   Credit UCI, LSAC); ensure `make full` runs end-to-end from a clean clone; pin
   requirements.txt; add a top-level "How to reproduce" to README.
5. Makefile: remove dead targets, ensure results/deliverables/paper/report/test/validate
   all work and are documented in `make help`.

CONSTRAINTS: touch NO scientific result — data/CSVs/canonical are frozen. `pytest tests/ -q`
must stay green after every stage. Commit in logical chunks. Deliver docs/REPO_AUDIT.md +ff
a clean tree.
```

### Agent K — Paper & report finalization (needs H's final numbers)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/FINAL_COMPLETION_PLAN.md and
the current paper/ and report/ first.

Bring the paper and report to submission quality for Aug 10. The narrative is fixed and
honest — do not invent results:
1. Story: (a) FairnessTargetedPGD attack; (b) τ=1 fix — the central finding, with the
   τ=100 artifact shown as the ablation that motivated it; (c) DRO robust on Adult & Credit
   at α≤0.2 under all three attacks, n=6, Wilcoxon p<0.05; (d) the FIRST real IF results
   (from Agent H); (e) honest negatives: LSAC/DP degeneracy (own subsection, cite
   docs/LSAC_DEGENERACY.md) and the α≥0.3 constant-predictor limit; (f) UTKFace per Agent M.
2. Every table auto-generated from results/canonical_tau1.json — zero hardcoded numbers.
   Hunt and remove any remaining hardcoded τ=100 / 3-seed values (MASTER_DISPATCH lists the
   exact lines: report/report.tex:441,463; paper/sections/results.tex:23-24,54;
   paper/auto_generated/key_findings.tex).
3. Publication-quality figures: consistent style, colour-blind-safe, labelled axes,
   captions that state n=6 and the defensible regime. Reuse the figure generators; don't
   hand-edit PDFs.
4. Abstract + intro + related work + method + results + discussion + limitations +
   conclusion all coherent. Limitations must state LSAC/DP, α≥0.3, and UTKFace honestly.
5. Build clean: make paper && make report. Deliver both PDFs + a one-paragraph
   "what changed" note.
```

### Agent L — Independent verification & QA (gates submission)
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/FINAL_COMPLETION_PLAN.md.
You are the adversarial checker — assume claims are wrong until proven from data. This is
the "verify all claims" step Kuldeep explicitly asked for.

1. Trace EVERY numeric claim in README.md, STATUS.md, docs/KULDEEP_CORRECTION.md,
   docs/MEETING_2026-08-04.md, and the paper/report back to results/canonical_tau1.json.
   Recompute means, deltas, win-counts, and Wilcoxon p-values yourself. List any mismatch.
2. Scan for fabricated/stale/hardcoded values across all .tex, .md, and figure-generator
   .py files (grep for suspicious constants like 0.752, 0.0195, tau=100, n=3). Report each.
3. Confirm provenance uniformity: all 540 canonical rows have tau=1.0, k_inner=10,
   epochs=60, and IF is non-degenerate. Confirm radii_mode label is truthful (MASTER_DISPATCH
   Agent B flagged the uniform-vs-empirical dead-branch — verify it was resolved).
4. Run `pytest tests/ -q` and `make validate`. Run the security-review skill / semgrep over
   the code. Report anything.
5. Deliver docs/VERIFICATION_REPORT.md: a table of every claim → source → verified?/mismatch.
   Nothing ships to Aug 10 with an unresolved mismatch.
```

### Agent M — UTKFace: obtain the real dataset and run it, or scope out honestly
```
Working dir: /Users/srujansai/Desktop/DRO-FairML. Read docs/FINAL_COMPLETION_PLAN.md and
docs/UTKFACE_PIPELINE.md.

The last "blocked" item. flair2 GPU access never came, but this machine has MPS and
UTKFace is a PUBLIC dataset. Try to actually complete it:
1. Attempt to obtain real UTKFace images from a public mirror (Kaggle / GitHub / academic
   torrent). There is already results-adjacent data/raw/utkface_features_smoke.npz (a smoke
   subset of ResNet18 features) — check whether a full feature set is obtainable or
   extractable.
2. If obtainable: extract ResNet18 features (fast on MPS), then run the SAME canonical
   protocol (tau=1, k_inner=10, 6 seeds, the three attacks) via the UTKFace runner, writing
   real (not synthetic) rows. Regenerate the UTKFace figures/tables. This turns a withdrawn
   claim into a real fourth result.
3. If genuinely not obtainable in the time window: leave UTKFace as honest future work —
   keep docs/_archive/UTKFACE_RESULTS_SYNTHETIC_SMOKE_ONLY.md, ensure no paper/report claim
   depends on synthetic data (Agent K), and state the blocker in limitations.

CRITICAL: never let synthetic Gaussian features (run_utkface.py:_make_synthetic_utkface)
be reported as real results. Tag provenance clearly. Report which path (2 or 3) you took.
```

---

## Sequencing

```
NOW (done)      IF parallel sweep running (pid 10146) → 540 rows in ~30 min
TODAY ≤4pm      H (finalize+reconcile) → I (meeting brief)          [sequential]
                → present at 4pm meeting
Aug 5–9         J (professionalize) ∥ K (paper, after H) ∥ M (UTKFace)   [parallel]
Aug 9–10        L (verify everything) → fix mismatches → final commit + push
Aug 10          submit
```

## Definition of "100% done" (Aug 10)
- [ ] Canonical 540 rows, all three attacks, IF non-degenerate, provenance uniform
- [ ] Every table/figure/PDF regenerated from canonical; zero hardcoded numbers
- [ ] Every claim in every doc traced to data by Agent L; zero unresolved mismatches
- [ ] LSAC/DP degeneracy, α≥0.3 limit, and UTKFace status all stated honestly
- [ ] One clean pipeline; dead code and duplicates archived; reproducible from clean clone
- [ ] `pytest` green, `make full` works, security/QA pass
- [ ] Paper + report submission-ready; meeting brief delivered
- [ ] Kuldeep correction sent; UTKFace either done (real) or honestly scoped out
```
```
