# DRO-FairML — Distributionally Robust Optimization for Fairness

Implements Algorithm 1 (min-max Lagrangian DRO-FAIR with corruption-calibrated TV
uncertainty sets) vs a Naive-FAIR baseline, under **adversarial** fairness-targeted
PGD attacks (DP / IF / combined). Datasets: Adult, Credit, LSAC (tabular), UTKFace (image).

## Start here
- **[STATUS.md](STATUS.md)** — single source of truth for completion state (IF sweep, deliverables).
- **[HANDOFF.md](docs/_archive/HANDOFF.md)** — full project state, history, every decision, and constraints.
- **[MASTER_PLAN.md](docs/_archive/MASTER_PLAN.md)** — remaining work split into agent briefs (file-owned, parallel-safe).
- **[KULDEEP_DISCUSSION.md](KULDEEP_DISCUSSION.md)** — concise technical brief for Kuldeep working session (tau=1 Adult table from CSVs, ablations, LSAC framing, live status + asks).
- **[SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)** — flair2 GPU setup for UTKFace + exact server commands (credentials NOT stored here; see your password manager / email supin.gopi for the flair2 account).

All other historical meeting prep, one-pagers, timelines, launch snapshots and audits are consolidated in `docs/_archive/` (see june-root-cleanup/ and previous-root-archive/ subdirs) so the root stays minimal and scannable.

## How to reproduce

End-to-end path from a clean clone. Prefer using the **committed** canonical JSON rather than re-running the full grid unless you need a full recompute.

### 1. Install

```bash
python3 -m pip install -r requirements.txt
# optional, for PDFs:
#   brew install tectonic   # or install tectonic from upstream
make install   # same as pip install -r requirements.txt
```

Python **≥ 3.10**. Core deps (torch, numpy, pandas, scikit-learn, scipy, matplotlib,
seaborn, pytest, xlrd/openpyxl for Credit xls) are listed with compatible version
ranges in `requirements.txt`.

### 2. Download data (tabular)

```bash
bash data/download_data.sh            # Adult (UCI), Credit (UCI), LSAC (public mirror)
bash data/download_data.sh --verify   # same + SHA-256 check against known pins
# make data   # equivalent to --verify
```

Provenance is documented in the script header:

| Dataset | Source | Local path |
|---------|--------|------------|
| Adult | UCI ML Repository | `data/raw/adult.data`, `adult.test` |
| Credit | UCI “default of credit card clients” | `data/raw/default_of_credit_card_clients.xls` |
| LSAC | Public research mirror ([damtharvey/law-school-dataset](https://github.com/damtharvey/law-school-dataset)) | `data/raw/lsac.csv` |

UTKFace images are **not** auto-downloaded (large). Use `bash data/download_data.sh --utkface`
for manual steps, then `docs/UTKFACE_PIPELINE.md`. Do not treat
`data/raw/utkface_features_smoke.npz` as a full real-image result.

### 3. Run tests

```bash
make test
# or: python3 -m pytest tests/ -v
```

### 4. Canonical results (tabular)

**Already committed:** `results/canonical_tau1.json` holds the DP + Combined grid
(360 rows: 3 datasets × 5 α × {DP, Combined} × 6 seeds × 2 methods). The IF-attack
third (target +180 → **540** total) may still be filling on a local parallel sweep —
check live status in **STATUS.md** (do not assume IF is complete).

```bash
# Progress check
python3 -c "import json,collections; d=json.load(open('results/canonical_tau1.json')); print(len(d), dict(collections.Counter(r['attack'] for r in d)))"

# Full recompute (CPU OK; multi-hour). Resume-safe.
python3 experiments/run_canonical.py

# IF third only (parallel workers; resume-safe)
python3 experiments/run_if_parallel.py 10
```

Locked config: **τ=1.0**, **K_inner=10**, **epochs=60**, **pgd_steps=20**, **n_seeds=6**,
`lambda_init=0.0`, `radii_mode=uniform`, `coordinated=False`. See STATUS.md §2.

Quick smoke (not for claims):

```bash
python3 experiments/run_canonical.py --smoke
```

### 5. Validate, tables, figures, PDFs

```bash
make validate          # Wilcoxon / consistency checks on current results
make results           # tables + plots from existing results
make deliverables      # broader deliverable pack
make paper             # paper/main.pdf via tectonic
make report            # report/report.pdf via tectonic
```

`make full` runs the **legacy** `main.py --full-pipeline` (older experiment driver with
default `n_seeds=10`), not the canonical τ=1 grid. Prefer `run_canonical.py` /
committed `canonical_tau1.json` for paper claims.

### 6. Hardware notes

| Workload | Hardware |
|----------|----------|
| Tabular (Adult / Credit / LSAC) | **CPU is fine.** Full 540-config grid is multi-hour on laptop CPU; parallel IF workers help. |
| Unit tests | CPU, minutes. |
| UTKFace feature extract + image pipeline | **MPS (Apple) or CUDA GPU** recommended; CPU-only is possible but slow. Real images required for any image claim (see `docs/UTKFACE_PIPELINE.md`, `docs/SERVER_RUNBOOK.md`). |

### Makefile targets (`make help`)

| Target | What it does |
|--------|----------------|
| `install` | `pip install -r requirements.txt` |
| `data` | Download + SHA-256 verify tabular data |
| `test` | `pytest tests/ -v` |
| `monitor` | Count `results/*.json`; print watcher / IF progress hints |
| `validate` | `experiments/validate_results.py` |
| `theory` | `experiments/verify_theory.py` |
| `experiments` | Legacy `run_experiments.py` (not canonical) |
| `results` | `main.py --generate-results` |
| `deliverables` | `generate_all_deliverables.py` |
| `review` | Paths to archived review checklists |
| `paper` / `report` | Build PDFs with tectonic |
| `full` | Legacy full pipeline |
| `clean` | Caches and checkpoint junk |

## Key code
- `src/training/dro_fair.py` — DRO-FAIR trainer (Algorithm 1).
- `src/training/naive_fair.py` — Naive-FAIR baseline.
- `src/corruption/adversarial.py` — `FairnessTargetedPGD` (the attack) + `RandomCorruptor` (baseline only).
- `experiments/run_fairness_pgd.py` — main tabular experiment driver.
- `experiments/run_canonical.py` — canonical τ=1 grid writer (`results/canonical_tau1.json`).
- `experiments/run_tau_ablation.py`, `run_knn_ablation.py`, `run_lambda_lr_grid.py` — ablations.

## Headline finding (scoped — see docs/MASTER_DISPATCH.md BLOCKER 3)
At **α ≤ 0.2**, DRO-FAIR achieves lower DP than Naive-FAIR on **Adult and Credit** under
the **DP and Combined** attacks (p<0.05, n=6; Adult/DP wins every α at p≤0.031 with α=0.1
at **5/6**, others **6/6**). Do **not** claim “all three attacks” until the IF-attack third
is complete (180 rows) and STATUS.md says so. **LSAC/DP is degenerate** (model collapses to
the majority-class predictor) and is reported separately, not as a DRO win. At **α ≥ 0.3**
both methods fall below the constant-predictor baseline on **Adult and Credit** (LSAC under
DP stays pinned *at* majority accuracy, not below), so **no method claim is made** there.
The earlier "DRO is fragile" result was a **tau=100 temperature artifact**, fixed by the
canonical tau=1. Current tables: KULDEEP_DISCUSSION.md (sourced from
`results/canonical_tau1.json`). **IF-attack results:** metric fixed; local sweep may be
partial — use only non-degenerate IF-attack rows and treat the full IF third as complete
only when STATUS.md says so (target 540 rows). A small Adult IF PoC is in
`results/if_poc_adult.json`.

## Run (local CPU, non-canonical quick path)
```bash
python3 experiments/run_fairness_pgd.py --datasets adult credit lsac --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 3
```

## Hard constraints (do not violate)
Corruption is always adversarial (never RandomCorruptor as the method); `epochs=60`,
`K_inner=10`; step order θ→λ→p; dual λ init 0.0; `lambda_max=1.5` all datasets; no oracle
corruption rates to DRO. Full rationale in docs/_archive/HANDOFF.md.

## Project structure (clear + minimal root)
Root contains key docs (README, STATUS, KULDEEP_DISCUSSION, …) + standard entry points
(main.py, Makefile, setup.py, requirements.txt, LICENSE) and the main directories.

**Core dirs:**
- `src/` — implementation (Algorithm 1 trainer, FairnessTargetedPGD attack, radii, etc.)
- `experiments/` — runners (`run_canonical.py` with K=10/tau=1/provenance, ablations, plot generators, UTK server script). Old one-offs in `experiments/_archive/`.
- `results/` + `figures/` — deliverables (json with provenance rows, meeting-ready plots).
- `docs/` — design notes + `_archive/` + `chat/gchat_raw_export.md`.
- `paper/`, `report/` — paper .tex + built PDFs.
- `data/`, `configs/`, `tests/`, `scripts/`

**Intentionally local / generated (gitignored):**
- `FRIEND/` — slim laptop comfort copies only.
- `logs/`, `packages/`, `venv/`

See .gitignore for exact excludes. After any big run or meeting prep, archive transients under docs/_archive/ so the tree stays findable.

## Project Flow & Structure
See **[docs/PROJECT_FLOW.md](docs/PROJECT_FLOW.md)** for the end-to-end flow:
- Launch (run_canonical with K=10/tau=1 + lambda grid + tau ablations)
- Analysis (analyze_tau1 + wilcoxon + tables)
- Viz (meeting-format plots + final figures)
- Report (paper/report + auto sections)
- Automation: `make results` / `make deliverables` / `make validate` / `make paper` / `make report` regenerate artifacts from `results/canonical_tau1.json`.

**Clean structure (post-cleanup):**
- Root: key persistent docs + entrypoints.
- `scripts/`: orchestration (finalize_experiments.py, finish_..., watchers).
- `docs/_archive/`: full history.
- `logs/`: log files.

After runs/meetings: move transients to `_archive/`. Root stays minimal + scannable.
