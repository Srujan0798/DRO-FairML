# DRO-FairML — Distributionally Robust Optimization for Fairness

Implements Algorithm 1 (min-max Lagrangian DRO-FAIR with corruption-calibrated TV
uncertainty sets) vs a Naive-FAIR baseline, under **adversarial** fairness-targeted
PGD attacks (DP / IF / combined). Datasets: Adult, Credit, LSAC (tabular), UTKFace (image).

## Start here
- **[HANDOFF.md](docs/_archive/HANDOFF.md)** — full project state, history, every decision, and constraints. Read this first.
- **[MASTER_PLAN.md](docs/_archive/MASTER_PLAN.md)** — remaining work split into agent briefs (file-owned, parallel-safe).
- **[KULDEEP_DISCUSSION.md](KULDEEP_DISCUSSION.md)** — concise technical brief for Kuldeep working session (tau=1 Adult table from CSVs, ablations, LSAC framing, live status + asks).
- **[SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)** — flair2 GPU setup for UTKFace + exact server commands (credentials NOT stored here; see your password manager / email supin.gopi for the flair2 account).

All other historical meeting prep, one-pagers, timelines, launch snapshots and audits are consolidated in `docs/_archive/` (see june-root-cleanup/ and previous-root-archive/ subdirs) so the root stays minimal and scannable.

## Key code
- `src/training/dro_fair.py` — DRO-FAIR trainer (Algorithm 1).
- `src/training/naive_fair.py` — Naive-FAIR baseline.
- `src/corruption/adversarial.py` — `FairnessTargetedPGD` (the attack) + `RandomCorruptor` (baseline only).
- `experiments/run_fairness_pgd.py` — main tabular experiment driver.
- `experiments/run_tau_ablation.py`, `run_knn_ablation.py`, `run_lambda_lr_grid.py` — ablations.

## Headline finding (scoped — see docs/MASTER_DISPATCH.md BLOCKER 3)
At **α ≤ 0.2**, DRO-FAIR achieves lower DP than Naive-FAIR on **Adult and Credit** under
all three attacks (p<0.05, n=6). **LSAC/DP is degenerate** (model collapses to the
majority-class predictor) and is reported separately, not as a DRO win. At **α ≥ 0.3**
both methods fall below the constant-predictor baseline, so **no method claim is made**
there. The earlier "DRO is fragile" result was a **tau=100 temperature artifact**, fixed
by the canonical tau=1. Current tables: KULDEEP_DISCUSSION.md (sourced from
`results/canonical_tau1.json`).

## Run (local CPU)
```bash
python3 experiments/run_fairness_pgd.py --datasets adult credit lsac --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 3
```

## Hard constraints (do not violate)
Corruption is always adversarial (never RandomCorruptor as the method); `epochs=60`,
`K_inner=10`; step order θ→λ→p; dual λ init 0.0; `lambda_max=1.5` all datasets; no oracle
corruption rates to DRO. Full rationale in docs/_archive/HANDOFF.md.

## Project structure (clear + minimal root)
Root now contains only the 5 persistent key docs above + standard entry points (main.py, Makefile, setup.py, requirements.txt, LICENSE) and the main directories.

**Core dirs (original project):**
- `src/` — implementation (Algorithm 1 trainer, FairnessTargetedPGD attack, radii, etc.)
- `experiments/` — runners (run_canonical.py with K=10/tau=1/provenance, ablations, plot generators, UTK server script). Old one-offs in experiments/_archive/.
- `results/` + `figures/` — all committed deliverables (json with full provenance rows, meeting-ready plots with CM serif fonts, error bars, absolute DP/IF values, no shading).
- `docs/` — design notes (FAIRNESS_PGD_DESIGN, KEY_FORMULAS, UTKFACE_*, TAU1_ABLATION etc.) + `_archive/` (everything historical/one-off consolidated here for clarity — no more root clutter) + `chat/gchat_raw_export.md` (the *entire* conversation with Madam + Kuldeep stored very clearly: full timeline + all threads/Qs + decisions + links to results/plots).
- `paper/`, `report/` — paper .tex + built PDFs. (The old `submission/` fork was retracted and moved to `docs/_archive/submission_may2026/`; do not use it.)
- `data/`, `configs/`, `tests/`, `scripts/`

**Intentionally local / generated (gitignored, never in clones or "original project structure"):**
- `FRIEND/` — slim duplicate copies **only for your laptop comfort** during Kuldeep (and friend) chats. Do not put originals or full project source inside them. The real files live in the tracked locations above.
- `logs/`, `packages/` (vendored wheels for offline), `venv/`

See .gitignore for exact excludes. After any big run or meeting prep, archive transients under docs/_archive/ so the tree stays findable.

## Project Flow & Structure (added for clarity)
See **[docs/PROJECT_FLOW.md](docs/PROJECT_FLOW.md)** for the complete end-to-end flow:
- Launch (run_canonical with K=10/tau=1 + lambda grid + tau ablations)
- Analysis (analyze_tau1 + wilcoxon + tables)
- Viz (meeting-format plots + final figures, high-α tau first per Kuldeep)
- Report (paper/report + auto sections)
- Automation: `make results` / `make deliverables` / `make validate` / `make paper` / `make report` regenerate all artifacts from `results/canonical_tau1.json` (360 rows, DP+Combined; IF pending cluster).

**Clean structure (post-cleanup):**
- Root: only key persistent docs (HANDOFF, KULDEEP, MASTER_PLAN, SERVER, README) + entrypoints.
- `scripts/`: all orchestration (finalize_experiments.py, finish_..., watchers).
- `docs/_archive/`: full history incl. prior status/orchestrator MDs (never pollute root).
- `logs/`: everything log-related.
- Comfort dups only in FRIEND/ (laptop-only, no full source).

After runs/meetings: move transients to _archive/. Root stays minimal + scannable.
