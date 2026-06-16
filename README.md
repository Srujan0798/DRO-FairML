# DRO-FairML — Distributionally Robust Optimization for Fairness

Implements Algorithm 1 (min-max Lagrangian DRO-FAIR with corruption-calibrated TV
uncertainty sets) vs a Naive-FAIR baseline, under **adversarial** fairness-targeted
PGD attacks (DP / IF / combined). Datasets: Adult, Credit, LSAC (tabular), UTKFace (image).

## Start here
- **[HANDOFF.md](HANDOFF.md)** — full project state, history, every decision, and constraints. Read this first.
- **[MASTER_PLAN.md](MASTER_PLAN.md)** — remaining work split into agent briefs (file-owned, parallel-safe).
- **[KULDEEP_DISCUSSION.md](KULDEEP_DISCUSSION.md)** — concise technical brief for Kuldeep working session (tau=1 Adult table from CSVs, ablations, LSAC framing, live status + asks).
- **[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)** — flair2 GPU setup for UTKFace + exact server commands (credentials NOT stored here; see your password manager / email supin.gopi for the flair2 account).

All other historical meeting prep, one-pagers, timelines, launch snapshots and audits are consolidated in `docs/_archive/` (see june-root-cleanup/ and previous-root-archive/ subdirs) so the root stays minimal and scannable.

## Key code
- `src/training/dro_fair.py` — DRO-FAIR trainer (Algorithm 1).
- `src/training/naive_fair.py` — Naive-FAIR baseline.
- `src/corruption/adversarial.py` — `FairnessTargetedPGD` (the attack) + `RandomCorruptor` (baseline only).
- `experiments/run_fairness_pgd.py` — main tabular experiment driver.
- `experiments/run_tau_ablation.py`, `run_knn_ablation.py`, `run_lambda_lr_grid.py` — ablations.

## Headline finding
Fixing **tau=1** (vs the old stepped tau=100 schedule) makes DRO beat Naive on DP at
every corruption level α on Adult, with the advantage growing in α. The earlier
"DRO is fragile" result was a high-temperature artifact. See KULDEEP_DISCUSSION.md for the current tables (sourced from committed results/*.csv + canonical_tau1.json).

## Run (local CPU)
```bash
python3 experiments/run_fairness_pgd.py --datasets adult credit lsac --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 3
```

## Hard constraints (do not violate)
Corruption is always adversarial (never RandomCorruptor as the method); `epochs=60`,
`K_inner=10`; step order θ→λ→p; dual λ init 0.0; `lambda_max=1.5` all datasets; no oracle
corruption rates to DRO. Full rationale in HANDOFF.md.

## Project structure (clear + minimal root)
Root now contains only the 5 persistent key docs above + standard entry points (main.py, Makefile, setup.py, requirements.txt, LICENSE) and the main directories.

**Core dirs (original project):**
- `src/` — implementation (Algorithm 1 trainer, FairnessTargetedPGD attack, radii, etc.)
- `experiments/` — runners (run_canonical.py with K=10/tau=1/provenance, ablations, plot generators, UTK server script). Old one-offs in experiments/_archive/.
- `results/` + `figures/` — all committed deliverables (json with full provenance rows, meeting-ready plots with CM serif fonts, error bars, absolute DP/IF values, no shading).
- `docs/` — design notes (FAIRNESS_PGD_DESIGN, KEY_FORMULAS, UTKFACE_*, TAU1_ABLATION etc.) + `_archive/` (everything historical/one-off consolidated here for clarity — no more root clutter) + **CHAT_HISTORY_MAY_JUNE.md** (the *entire* conversation with Madam + Kuldeep stored very clearly: full timeline + all threads/Qs + decisions + links to results/plots).
- `paper/`, `report/`, `submission/` — paper .tex + built PDFs.
- `data/`, `configs/`, `tests/`, `scripts/`

**Intentionally local / generated (gitignored, never in clones or "original project structure"):**
- `kuldeep_meeting/` and `FRIEND/` — slim duplicate copies (meeting plots + KULDEEP_DISCUSSION snapshot + ready_chat_message.txt + conversation_key.txt + CHAT_HISTORY_MAY_JUNE.md copy) **only for your laptop comfort** during Kuldeep (and friend) chats. Do not put originals or full project source inside them. The real files live in the tracked locations above.
- `logs/`, `packages/` (vendored wheels for offline), `venv/`

See .gitignore for exact excludes. After any big run or meeting prep, archive transients under docs/_archive/ so the tree stays findable.
