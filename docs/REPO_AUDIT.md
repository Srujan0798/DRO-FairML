# DRO-FairML — Repository Audit (Agent J, Phase 1)

**Date:** 2026-08-04  
**Mode:** AUDIT ONLY — no files deleted, moved, or scientific results modified.  
**Scope:** `src/`, `experiments/`, `scripts/`, `tests/`, `docs/`, `paper/`, `report/`, `configs/`, root entry points, Makefile.  
**Do not touch:** `results/canonical_tau1.json` scientific content (or any committed result numbers).

This inventory is the prerequisite for Phase 2 consolidation. Recommendations are **keep / merge / archive / delete**.

---

## 0. Executive summary

| Area | Active files (approx) | Assessment |
|------|----------------------|------------|
| `src/` | 16 `.py` | **Healthy core.** τ centralized; FairnessTargetedPGD exported; trainers/metrics/attacks coherent. |
| `experiments/` | 72 `.py` + 21 already archived | **Heavy duplication.** 5 UTKFace runners, ~12 figure generators, ~10 analysis one-offs, dual pipelines (`all_results` vs `canonical_tau1`). |
| `scripts/` | 19 | Mostly one-shot watchers/orchestrators; keep 2–4 operational, archive rest. |
| `tests/` | 10 | Keep all; green gate for consolidation. |
| `docs/` | 19 active + 103 archive | Keep design/findings/status; archive agent-prompt noise and superseded status docs. |
| `paper/` + `report/` | publication tree | Keep; regenerate from canonical only (Agent K). |
| Root | entry points + strays | `main.py` / Makefile need rewiring; root logs/auto_pipeline → archive. |

### Special checks (requested)

| Check | Result |
|-------|--------|
| `def get_temperature` only in `src/temperature.py` | **PASS** for active tree. Only other definition: `docs/_archive/submission_may2026/run_experiments.py` (archived τ=100 fork). |
| `FairnessTargetedPGD` in `src/corruption/__init__.py` | **PASS** — exported with `AdversarialCorruptor`, `RandomCorruptor`. |
| `image_pgd.py` | **Already gone** (Agent F). |
| `get_run_config` in `src/training/__init__.py` | **Already gone**. |
| Silent stale fallbacks | **Several remain** (see §7). Highest risk: `meeting_summary.py` (K_inner=5 bak + tau_ablation), `generate_all_deliverables.py` (reads `results/stale_archived/`), `make results` → `all_results.json` (90 nested rows, not flat 540). |
| Makefile dead / misleading targets | `monitor` (cosmetic only), `review` (echo archived paths), `experiments` (old `run_experiments.py` / n_seeds=10), `full` → legacy pipeline not canonical. |

### Recommendation counts (active inventory, excluding `docs/_archive/**` and `experiments/_archive/**`)

| Recommendation | Count (approx) |
|----------------|----------------|
| **keep** | ~55 |
| **merge** | ~35 |
| **archive** | ~55 |
| **delete** | ~15 (root logs, byte-duplicate strays, broken callers that only raise) |

---

## 1. Already completed (Agent F) — do not redo

From `docs/MASTER_DISPATCH.md` §1 Agent F / progress table:

- τ zombies consolidated into `src/temperature.py` (constant 1.0).
- `FairnessTargetedPGD` export fixed.
- Contaminated loaders: `load_fairness_pgd_results()` now **raises**.
- `load_canonical_tau1()` fail-loud on missing / wrong `k_inner`.
- Stale results moved under `results/stale_archived/`.
- Obsolete orchestrators moved to `experiments/_archive/`.
- `docs/_archive/` single archive tree (includes `submission_may2026/`).
- Dead `image_pgd.py` / `get_run_config` removed.

Phase 2 should **build on** this, not re-delete already-gone paths.

---

## 2. `src/` inventory

| Path | Purpose (1 line) | Rec | Dependencies / notes |
|------|------------------|-----|----------------------|
| `src/temperature.py` | Single `get_temperature` → τ=1.0 | **keep** | Import from all runners |
| `src/corruption/__init__.py` | Public exports for corruptors + PGD | **keep** | Exports FairnessTargetedPGD ✓ |
| `src/corruption/adversarial.py` | `AdversarialCorruptor`, `FairnessTargetedPGD`, `RandomCorruptor` | **keep** | Core attack; large (~670 LOC). Optional dead methods still present (legacy FGSM helpers if any remain unused — verify before delete). |
| `src/data/__init__.py` | Dataset package exports | **keep** | |
| `src/data/datasets.py` | Adult / Credit / LSAC / UTKFace loaders; fail-loud tabular | **keep** | UTKFace path used by image runners |
| `src/evaluation/__init__.py` | Metrics exports | **keep** | |
| `src/evaluation/metrics.py` | Acc, DP, IF (cosine IF fix) | **keep** | Tests depend |
| `src/models/__init__.py` | Exports MLP only | **keep** | CNN not exported (optional polish) |
| `src/models/classifier.py` | MLP binary classifier | **keep** | |
| `src/models/cnn_classifier.py` | ResNet18 CNN for pixel UTKFace | **keep** (if UTKFace real) / else **archive** after M | Used by pixel/randinit paths |
| `src/training/__init__.py` | Trainer exports | **keep** | No `get_run_config` |
| `src/training/dro_fair.py` | Algorithm 1 DRO trainer | **keep** | |
| `src/training/naive_fair.py` | Naive-FAIR baseline | **keep** | |
| `src/training/standard_ml.py` | Unconstrained ML baseline | **keep** (tests / tradeoff) | Low use in canonical path |
| `src/utils/__init__.py` | Projection export | **keep** | |
| `src/utils/projections.py` | Simplex + L1-ball / Dykstra | **keep** | |

**src verdict:** no structural consolidation needed beyond optional CNN export and dead-method cleanup inside `adversarial.py` (only after grep proves zero callers).

---

## 3. `experiments/` inventory

### 3.1 Canonical pipeline (keep as anchors)

| Path | Purpose | Rec | Notes |
|------|---------|-----|-------|
| `run_fairness_pgd.py` | Tabular FairnessTargetedPGD single-config + batch driver | **keep** | Shared primitive for wrappers |
| `run_canonical.py` | Canonical grid driver (τ=1, K=10, provenance) | **keep** | Primary long-run driver |
| `run_if_parallel.py` | Parallel IF-attack completion → merge into canonical | **keep** | Resume-safe; meeting-critical |
| `_run_if_chunk.py` | Worker chunk for IF parallel | **keep** (with parallel) or **merge** into package | Implementation detail |
| `loaders.py` | Fail-loud `load_canonical_tau1` / raise on contaminated | **keep** | Extend as sole loader API |
| `canonical_to_all_results.py` | Flat canonical → nested `all_results.json` for legacy fig gens | **merge** | Temporary bridge; delete after figure gens read canonical |
| `compute_canonical_wilcoxon.py` | Wilcoxon over canonical cells | **keep** | Docstring still mentions tau_ablation fallback; **code now uses loaders only** — clean dead constants |
| `generate_report_tables.py` | LaTeX auto tables from canonical | **keep** | |
| `validate_results.py` | Gate for DP wins / Wilcoxon | **merge** | Currently reads `all_results.json` — must retarget canonical |
| `verify_theory.py` | Radius / projection theory checks | **keep** | Makefile `theory` |

### 3.2 UTKFace near-duplicates (highest risk cluster)

| Path | Purpose | Rec | Duplicates |
|------|---------|-----|------------|
| `run_utkface.py` | Feature-space UTKFace; **silent synthetic** if data missing | **merge** → one runner | Defines `_make_synthetic_utkface`; tags `UTKFace (synthetic)` |
| `run_utkface_server.py` | GPU batch server script (canonical protocol) | **merge** | Overlaps server flags with extended |
| `run_utkface_extended.py` | alpha_sweep / fairness_pgd / lambda_max modes | **merge** | Modes should be CLI flags on one runner |
| `run_utkface_pixel_pgd.py` | Pixel-space PGD (H2) | **merge** as mode | Own image loading loop |
| `run_utkface_randinit.py` | Random-init backbone (H1) | **merge** as mode | Own image loading loop |
| `analyze_utkface.py` | Plot UTKFace results | **merge** | Overlaps fig10 |
| `analyze_utkface_stats.py` | Stats on UTKFace json | **merge** | |
| `analyze_dro_failure.py` | Hypothesis writeup for image DRO fail | **archive** | Synthetic-era analysis |
| `generate_fig10.py` | UTKFace curves | **merge** into figure module | Reads `utkface_results.json` |
| `setup_celeba.py` / `setup_fairface.py` | Out-of-scope dataset prep | **archive** | Not in submission claims |

**Consolidation target:** one `experiments/run_utkface.py` (or `image_runner.py`) with modes; **fail loud** if real data missing (Agent M). No silent synthetic for reported rows.

### 3.3 Figure / deliverable generators (near-duplicates)

| Path | Purpose | Rec | Risk |
|------|---------|-----|------|
| `generate_figures.py` | fig1–fig7 from nested `all_results.json` | **merge** | Depends on bridge schema |
| `generate_all_figures.py` | “ALL” figs incl. UTKFace; calls **raising** loader | **archive** or fix | Broken if invoked |
| `generate_final_figures.py` | fig_final_* from canonical | **merge** | Best modern source for constant-predictor / wilcoxon plots |
| `generate_all_deliverables.py` | figD1–D10; **reads `results/stale_archived/`** | **merge/fix** | **Stale landmine** — Makefile `deliverables` target |
| `generate_results.py` | Tables/plots for `main.py --generate-results` | **merge** | Loads `all_results.json` only; empty list if missing |
| `generate_pdf_report.py` | Standalone PDF report (matplotlib) | **archive** | Superseded by paper/report TeX |
| `generate_summary_dashboard.py` | Meeting dashboard | **archive** | One-off |
| `generate_sensitivity_analysis.py` | Sensitivity fig; uses raising loader | **archive/fix** | Broken |
| `generate_fig8_matrix.py` | Attack–defense matrix | **merge** | Reads wilcoxon CSV |
| `generate_meeting_table.py` | Meeting table; raising loader | **archive/fix** | Broken |
| `generate_latex_extras.py` | Runtime/ablation LaTeX | **archive** or merge | |
| `generate_high_alpha_summary.py` | high_alpha CSV from tau ablations | **archive** | Ablation scoped dropped |
| `plot_tau1_headline.py` | Headline τ=1 figure | **merge** | |
| `plot_win_curves_tau1.py` / `plot_acc_win_curves.py` | Win curves | **merge** | Near-duplicates |
| `plot_acc_by_attack.py` / `plot_if_by_attack.py` | Per-attack panels | **merge** | |
| `plot_high_alpha_tau.py` | High-α + τ curves | **archive** | |
| `plot_lambda_*` (4 files) | Lambda diagnostic heatmaps | **merge** → one lambda plot module | |
| `plot_convergence.py` | Training curves | **merge** | |
| `plot_partial_results.py` | Partial PGD plots; **raises** | **delete/archive** | Dead |
| `plot_random_vs_adversarial.py` / `plot_uniform_vs_empirical.py` | Ablation plots | **archive** | Ablations dropped/partial |

**Target shape:**  
`experiments/figures.py` (or package `experiments/viz/`) with entrypoints used by `make results` / `make deliverables`, all reading **only** `load_canonical_tau1()` (+ optional ablation paths that fail loud).

### 3.4 Analysis / summary one-offs

| Path | Purpose | Rec |
|------|---------|-----|
| `analyze_tau1.py` | Master τ story + many figures | **merge** into stats+figures |
| `analyze_results.py` | Generic advanced analysis | **archive** |
| `analyze_partial_results.py` | Partial PGD | **archive** |
| `analyze_lsac_complete.py` | LSAC 90-row analysis; wrong path | **archive** |
| `analyze_high_alpha.py` | High-α conclusion from ablations | **archive** |
| `analyze_lambda_grid.py` | Lambda heatmap analysis | **merge** with lambda plots |
| `analyze_dro_failure.py` | Image failure narrative | **archive** |
| `summarize_tau1.py` / `summarize_random_vs_adv.py` | Print markdown tables | **archive** |
| `meeting_summary.py` | Meeting numbers | **fix then keep/merge** | **BROKEN FALLBACK** (§7) |
| `diagnostics.py` / `dro_radii_diagnostic.py` / `demonstrate_radii_mismatch.py` | Radii diagnostics | **archive** (findings already in LSAC_DEGENERACY) |

### 3.5 Ablation / alternate runners

| Path | Purpose | Rec |
|------|---------|-----|
| `run_tau_ablation.py` | Fixed-τ grid | **archive** (ablation dropped; keep code for reproducibility) |
| `run_knn_ablation.py` | k-NN ablation | **archive** |
| `run_lambda_lr_grid.py` / `run_lambda_grid_comprehensive.py` | λ grids | **archive** (partial / dropped) |
| `run_lambda_diagnostic.py` / `run_lambda_diagnostic_full.py` | λ diagnostics | **archive** |
| `run_random_vs_adversarial.py` | Random vs PGD | **archive** |
| `run_canonical_empirical.py` | Empirical radii companion | **keep or archive** | Keep if Q5 appendix stays; else archive |
| `run_ablations.py` | Generic ablations | **archive** |
| `run_experiments.py` | Pre-canonical nested-schema runner | **archive** | Still wired by `main.py` / `make experiments` |
| `run_robust.py` | Merge individual → all_results | **archive** |
| `run_parallel_batch.py` / `run_k10_targeted.py` / `run_single_adult.py` | Ad-hoc batch helpers | **archive** |

### 3.6 Already under `experiments/_archive/` (21 files)

Leave as-is; add/refresh `experiments/_archive/README.md` in Phase 2 listing each script’s historical role. No further action in Phase 1.

---

## 4. `scripts/` inventory

| Path | Purpose | Rec |
|------|---------|-----|
| `run_if_rerun_cluster.sh` | Cluster IF re-run | **keep** |
| `monitor_if_then_regen.sh` | Watch IF sweep → regen | **keep** (until post-meeting) then **archive** |
| `finalize_if_sweep.sh` | Finalize IF + commit | **keep** short-term / **archive** after |
| `extract_utkface_features.py` | ResNet feature extraction | **keep** (Agent M) |
| `finalize_experiments.py` | Regen artifacts after runs | **merge** into one finalize entry | Overlaps watchers |
| `canonical_watcher.py` / `canonical_watcher_poll.py` / `canonical_advancer_monitor.py` | Canonical progress watchers | **archive** (superseded) |
| `lambda_watcher.py` / `watch_lambda.sh` | Lambda grid watchers | **archive** |
| `agent_data_refresher.py` / `data_refresher_loop.sh` | Periodic refresh | **archive** |
| `auto_complete.sh` / `finish_everything_when_ready.sh` / `final_delivery_orchestrator.sh` | Full auto orchestrators | **archive** |
| `grok_final_delivery_orchestrator.py` | Grok-side poller | **archive** |
| `watch_and_finalize.py` / `delayed_poll.py` / `quick_poll_loop.sh` | Misc pollers | **archive** |

---

## 5. `tests/` inventory

| Path | Purpose | Rec |
|------|---------|-----|
| `conftest.py` | Shared fixtures | **keep** |
| `test_metrics.py` | IF non-degenerate etc. | **keep** |
| `test_fairness_pgd.py` | Attack unit tests | **keep** |
| `test_corruption.py` | Corruptor tests | **keep** |
| `test_projections.py` | Projection tests | **keep** |
| `test_radii_calibration.py` | Radii + attack | **keep** |
| `test_greedy_attack_superiority.py` | Greedy > random | **keep** |
| `test_end_to_end.py` | Smoke e2e + real data | **keep** |
| `test_cnn_classifier.py` | CNN smoke | **keep** |
| `__init__.py` | Package marker | **keep** |

**Constraint:** `pytest tests/ -q` must stay green after every Phase 2 stage.

---

## 6. Docs, paper, report, configs, root

### 6.1 Active docs (`docs/`) — stay vs archive

| Path | Rec | Why |
|------|-----|-----|
| `FINAL_COMPLETION_PLAN.md` | **keep** | Live plan through Aug 10 |
| `MASTER_DISPATCH.md` | **keep** | Historical blockers + Agent F baseline |
| `PROJECT_FLOW.md` | **keep** (update after consolidation) | Flow map |
| `KULDEEP_CORRECTION.md` | **keep** | Honest corrections |
| `LSAC_DEGENERACY.md` | **keep** | Scientific finding |
| `FAIRNESS_PGD_DESIGN.md` | **keep** | Attack design |
| `FAIRNESS_PGD_RESULTS.md` | **merge/update** or **archive** if superseded by STATUS | Check freshness |
| `KEY_FORMULAS.md` | **keep** | |
| `Q5_derivation.md` | **keep** | Empirical radii math |
| `TAU1_ABLATION_SUMMARY.md` | **keep** (historical finding) | |
| `FINDING_DRO_FAILS_ON_ADULT.md` | **archive** or annotate “superseded by τ=1” | Pre-fix narrative risk |
| `ABLATION_STATUS_REPORT.md` | **keep** | Adjudication of dropped ablations |
| `UTKFACE_PIPELINE.md` | **keep** | Agent M |
| `SERVER_RUNBOOK.md` | **keep** | flair2 |
| `EMAIL_TO_SUPIN_GOPI_DRAFT.txt` | **keep** until sent / **archive** after | |
| `AGENT_*.md`, `AGENT_PROMPTS*.md` | **archive** | Orchestration noise |
| `DELIVERABLES_CHECKLIST.txt` | **archive** after Aug 10 | Transient |
| `chat/gchat_raw_export.md` | **keep** | Primary conversation archive |
| `_archive/**` | **keep as archive** | Do not re-fragment |

Root docs:

| Path | Rec |
|------|-----|
| `README.md` | **keep** — rewrite “How to reproduce” in Phase 2 |
| `STATUS.md` | **keep** — single status SSOT (update with IF when H lands) |
| `KULDEEP_DISCUSSION.md` | **keep** or fold into docs/ | Working brief |
| `conv.md` | **archive** → `docs/chat/` or `_archive` | Raw chat dump |
| `docs/REPO_AUDIT.md` (this file) | **keep** | |
| `docs/REPO_CONSOLIDATION_PLAN.md` | **keep** | Phase 2 plan |

### 6.2 Paper / report

| Path | Rec |
|------|-----|
| `paper/main.tex` + `sections/*` | **keep** (Agent K prose) |
| `paper/auto_generated/*` | **keep** — must be regenerated only from canonical |
| `paper/references.bib` | **keep** |
| `paper/main.pdf` | **keep** (built artifact) |
| `paper/ICML_submission.pdf` | **untrack/archive** | May 4 retracted conclusion (Agent F flagged) |
| `report/report.tex` + `sections/*` | **keep** |
| `report/report.pdf` | **keep** |

### 6.3 Configs / data

| Path | Purpose | Rec |
|------|---------|-----|
| `configs/default.yaml` | Default hparams | **fix** | Still has **`tau: 100.0`** — misleading if anything reads it |
| `data/download_data.sh` | Tabular download | **keep** |
| `data/raw/*` | Cached datasets | **keep** (git policy as-is) |

### 6.4 Root strays

| Path | Rec |
|------|-----|
| `main.py` | **merge/rewrite** to call canonical pipeline |
| `Makefile` | **fix** (§8) |
| `setup.py`, `requirements.txt`, `LICENSE` | **keep** |
| `auto_pipeline.py`, `auto_pipeline_v2.py` + `.log` | **archive** |
| Root `*.log` (`credit_alpha04*.log`, `knn_ablation_k10.log`, `lambda_comprehensive.log`, `lsac_canonical.log`) | **delete or move to `logs/`** |

---

## 7. Broken / silent fallbacks (must fix in Phase 2)

These are the **scientific integrity** risks if someone re-runs analysis without reading this audit.

| Location | Behavior | Severity |
|----------|----------|----------|
| `experiments/meeting_summary.py` `_load_tau1` | Prefers `tau_ablation_tau1.json`; if short, **`tau_ablation_tau1_KINNER5_BAK.json`** | **CRITICAL** — wrong K_inner / non-canonical numbers |
| `experiments/generate_all_deliverables.py` | Hardcoded paths under **`results/stale_archived/`** for tau, wilcoxon, lambda, individual runs | **CRITICAL** — Makefile `deliverables` can regenerate figD* from stale data |
| `experiments/generate_results.py` / `main.py --generate-results` | Loads `results/all_results.json` (currently **90** nested rows) or returns `[]` silently | **HIGH** — wrong schema / incomplete grid |
| `experiments/validate_results.py` | Uses `all_results.json` after optional merge | **HIGH** — not the 540-row gate |
| `experiments/generate_all_figures.py`, `plot_partial_results.py`, `generate_meeting_table.py`, `generate_sensitivity_analysis.py` | Call `load_fairness_pgd_results()` → **always raises** | **MEDIUM** (loud fail — good) but dead Makefile/scripts paths |
| `experiments/run_utkface.py` | On missing data: **`_make_synthetic_utkface`** with only a print | **CRITICAL** for Agent M / any UTKFace claim |
| `experiments/loaders.constant_predictor_acc` | Falls back to hardcoded majority rates if data unloadable | **LOW** (documented; values correct per dataset) |
| `compute_canonical_wilcoxon.py` | Dead `TAU1_FALLBACK` constant + outdated docstring | **LOW** (code path fixed) |
| `configs/default.yaml` | `tau: 100.0` | **MEDIUM** if any loader trusts yaml |

**Note:** `loaders.load_canonical_tau1` and `load_fairness_pgd_results` (raise) are the **correct** pattern — migrate all readers to them.

---

## 8. Makefile audit

| Target | Current action | Verdict |
|--------|----------------|---------|
| `help` | Lists targets | **keep** — update text |
| `install` | pip requirements | **keep** |
| `test` | pytest | **keep** |
| `monitor` | Glob count + echo watcher | **dead-ish** — replace with real status or remove |
| `validate` | `validate_results.py` | **fix** path → canonical Wilcoxon |
| `theory` | `verify_theory.py` | **keep** |
| `experiments` | `run_experiments.py --n_seeds 10` | **dead for science** — should point to `run_canonical.py` or remove |
| `results` | `main.py --generate-results` → nested `all_results` | **rewire** to canonical tables/figures |
| `deliverables` | `generate_all_deliverables.py` (stale_archived) | **rewire** to non-stale generator |
| `review` | Echo archived checklists | **remove or retarget** VERIFICATION |
| `paper` / `report` | tectonic | **keep** |
| `full` | `main.py --full-pipeline` | **rewire** or document as non-canonical |
| `clean` | caches | **keep** |

Missing useful targets (recommend in Phase 2): `wilcoxon`, `tables`, `canonical` (document only — long compute).

---

## 9. Top 10 highest-risk duplicates / landmines

1. **`generate_all_deliverables.py` + Makefile `deliverables`** reading `results/stale_archived/` while claiming “canonical 540”.  
2. **`meeting_summary.py` K_inner=5 / tau_ablation preference** vs `canonical_tau1.json`.  
3. **Dual result schemas:** flat `canonical_tau1.json` vs nested `all_results.json` (90 rows) driving `make results` / `generate_figures.py`.  
4. **Five UTKFace runners** + shared synthetic fallback → risk of synthetic rows treated as real.  
5. **Figure generator swarm** (`generate_figures` / `generate_final_figures` / `generate_all_figures` / `generate_all_deliverables` / many `plot_*`) producing overlapping stems with different sources.  
6. **`main.py` + `run_experiments.py`** pre-provenance pipeline still advertised by Makefile `full` / `experiments`.  
7. **`configs/default.yaml` tau: 100.0** vs code τ=1.0.  
8. **`paper/ICML_submission.pdf`** retracted conclusion if opened first.  
9. **Watchers/orchestrators** (`scripts/*`, root `auto_pipeline*`) that can re-trigger regen from wrong scripts if left running.  
10. **Docs narrative fork:** `FINDING_DRO_FAILS_ON_ADULT.md` / old handoffs vs `STATUS.md` / τ=1 story.

---

## 10. Recommended consolidation plan (overview for Phase 2)

See **`docs/REPO_CONSOLIDATION_PLAN.md`** for ordered safe steps. Summary target tree:

```
src/                    # unchanged core
experiments/
  run_fairness_pgd.py   # primitive
  run_canonical.py      # tabular grid
  run_if_parallel.py    # IF completion
  run_utkface.py        # ONE image runner (modes)
  loaders.py            # sole result I/O
  stats.py              # wilcoxon + summaries (merge)
  figures.py            # all plots (merge)
  tables.py             # report/paper tex (merge generate_report_tables)
  validate_results.py
  verify_theory.py
  _archive/             # everything else + README
scripts/
  extract_utkface_features.py
  run_if_rerun_cluster.sh
  finalize_artifacts.py # one regen entry
  _archive/
docs/                   # design + STATUS + plans; prompts archived
```

**Invariants:** never rewrite `results/canonical_tau1.json` numbers; pytest green after each stage; fail loud, no silent fallbacks.

---

## 11. Blockers for Phase 2

| Blocker | Impact | Owner |
|---------|--------|-------|
| Agent H still regenerating from full IF grid (if in flight) | Avoid concurrent figure/table commits | Sequence after H or only touch code paths H does not write |
| Frozen scientific results | Consolidation must not re-run experiments or edit JSONs | Agent J constraint |
| `make deliverables` currently stale-backed | Cannot claim deliverables green until rewired | Phase 2 early |
| UTKFace real data not yet present | Runner consolidation OK; real runs are Agent M | M parallel |
| Large figure set with mixed provenance | Need single regen script + manifest | J + K |

**Non-blockers (already fixed):** multi-`get_temperature`, missing FairnessTargetedPGD export, contaminated loader silent success.

---

## 12. File counts snapshot (Phase 1)

| Tree | Count |
|------|------:|
| `experiments/*.py` (active) | 72 |
| `experiments/_archive/*` | 21 |
| `scripts/*` | 19 |
| `src/**/*.py` | 16 |
| `tests/*.py` | 10 |
| `docs/*` active md/txt (excl. archive) | ~19 |
| `docs/_archive` files | 103 |
| `paper` tex/bib | ~13 |
| `report` tex | 4 |
| `configs` | 1 |

---

*End of Phase 1 audit. No destructive changes performed.*
