# DRO-FairML — Distributionally Robust Optimization for Fairness

Implements Algorithm 1 (min-max Lagrangian DRO-FAIR with corruption-calibrated TV
uncertainty sets) vs a Naive-FAIR baseline, under **adversarial** fairness-targeted
PGD attacks (DP / IF / combined). Datasets: Adult, Credit, LSAC (tabular), UTKFace (image).

## Start here
- **[HANDOFF.md](HANDOFF.md)** — full project state, history, every decision, and constraints. Read this first.
- **[MASTER_PLAN.md](MASTER_PLAN.md)** — remaining work split into agent briefs (file-owned, parallel-safe).
- **[KULDEEP_DISCUSSION.md](KULDEEP_DISCUSSION.md)** — concise technical brief for Kuldeep working session (tau=1 Adult table from CSVs, ablations, LSAC framing, status).
- **[MEETING_TODAY.md](MEETING_TODAY.md)** — meeting notes pointer (see KULDEEP_DISCUSSION.md for current story).
- **[ADULT_RESULTS_FOR_KULDEEP.md](docs/_archive/ADULT_RESULTS_FOR_KULDEEP.md)** — archived (historical; numbers superseded by tau=1 CSVs).
- **[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)** — flair2 GPU setup for UTKFace (credentials NOT stored here; see your password manager / email supin.gopi for the flair2 account).

## Key code
- `src/training/dro_fair.py` — DRO-FAIR trainer (Algorithm 1).
- `src/training/naive_fair.py` — Naive-FAIR baseline.
- `src/corruption/adversarial.py` — `FairnessTargetedPGD` (the attack) + `RandomCorruptor` (baseline only).
- `experiments/run_fairness_pgd.py` — main tabular experiment driver.
- `experiments/run_tau_ablation.py`, `run_knn_ablation.py`, `run_lambda_lr_grid.py` — ablations.

## Headline finding
Fixing **tau=1** (vs the old stepped tau=100 schedule) makes DRO beat Naive on DP at
every corruption level α on Adult, with the advantage growing in α. The earlier
"DRO is fragile" result was a high-temperature artifact. See MEETING_TODAY.md.

## Run (local CPU)
```bash
python3 experiments/run_fairness_pgd.py --datasets adult credit lsac --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 3
```

## Hard constraints (do not violate)
Corruption is always adversarial (never RandomCorruptor as the method); `epochs=60`,
`K_inner=10`; step order θ→λ→p; dual λ init 0.0; `lambda_max=1.5` all datasets; no oracle
corruption rates to DRO. Full rationale in HANDOFF.md.
