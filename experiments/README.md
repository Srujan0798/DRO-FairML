# experiments/

Active drivers for the **canonical** DRO-FairML pipeline. One-offs live in `_archive/`.

## Makefile critical path (no retrain)

| Script | Role | Make target |
|--------|------|-------------|
| `validate_results.py` | Consistency + Wilcoxon gate on `canonical_tau1.json` | `make validate` |
| `compute_canonical_wilcoxon.py` | Write `results/canonical_wilcoxon.*` | `make wilcoxon` |
| `generate_report_tables.py` | Auto LaTeX tables for paper/report | `make tables` |
| `generate_results.py` | Nested tables + basic plots via `main.py` | `make results` |
| `generate_all_deliverables.py` | Full deliverable figure pack | `make deliverables` |
| `canonical_to_all_results.py` | Flat → nested bridge for legacy plots | (called by generate_results) |
| `verify_theory.py` | Numeric theory checks | `make theory` |
| `loaders.py` | Fail-loud load of `canonical_tau1.json` only | (library) |

## Training / runners

| Script | Role |
|--------|------|
| `run_canonical.py` | **Canonical** tabular grid (τ=1, K=10, n=6) — optional retrain |
| `run_if_parallel.py` | IF-attack third (already complete in committed JSON) |
| `run_fairness_pgd.py` | FairnessTargetedPGD entry (+ Wave-1 kwargs: attack_k, lr_lambda, corruptor_type) |
| `run_ablation_parallel.py` | Shared ablation driver (**refuses** writing locked 540/UTKFace) |
| `run_a1_knn.py` … `run_a5_empirical.py`, `run_n5_kinner.py` | Wave-1 advisor ablations → separate `results/*_*.json` |
| `_launch_wave1.py` | Sequential A3→A4→A5→N5→A1→A2 launcher (workers=1) |
| `summarize_rva.py` | Live summary of `random_vs_adversarial.json` (partial OK) |
| `agent_n4_if_wilcoxon.py` | Analysis-only IF-metric Wilcoxon → `if_wilcoxon_n4_summary.md` |
| `run_canonical_empirical.py` | Q5 empirical radii mode (optional) |
| `run_utkface.py` / `run_utkface_server.py` | REAL UTKFace (default no synthetic) |
| `run_experiments.py` | **DEPRECATED** legacy n_seeds=10; needs `FORCE_LEGACY=1` |
| `meeting_summary.py` | CLI summary for meetings (canonical-only loaders) |

## Invariants

1. Never rewrite science rows of `results/canonical_tau1.json` casually.
2. Never load `results/stale_archived/` for claims.
3. IF story is **mixed**; Adult/DP α=0.1 is **5/6**; LSAC/DP degenerate.
4. UTKFace: **90/90 REAL** locked in `results/utkface_canonical.json` — paper/report state **mixed clean-test pilot** only (not Adult copy). Do not dual-write or invent synthetic rows.

| Script | Role |
|--------|------|
| `plot_meeting_figs_540.py` | Meeting headline figs from canonical 540 |
