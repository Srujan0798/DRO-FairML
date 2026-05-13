# DRO-FAIR: Clean Project Structure

> This document describes the organized, cleaned-up project layout.
> Every file has a purpose. Nothing is dead code.

---

## Root Directory

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point. Run experiments, generate results, or full pipeline. |
| `Makefile` | Common commands: `make test`, `make experiments`, `make results` |
| `setup.py` | Python package setup |
| `requirements.txt` | Dependencies |
| `.gitignore` | Git ignore rules |
| `ICML_submission.pdf` | The original research paper |

---

## Source Code (`src/`)

| File | Purpose | Lines |
|------|---------|-------|
| `src/data/datasets.py` | Load Adult, Credit, LSAC datasets. No synthetic fallbacks. | ~200 |
| `src/models/classifier.py` | MLP with dropout, squeezed output, temperature-scaled predictions. | ~35 |
| `src/corruption/adversarial.py` | AdversarialCorruptor (PGD/FGSM) + RandomCorruptor (Gaussian). | ~280 |
| `src/training/naive_fair.py` | Naive-FAIR baseline. Full-batch, BCE loss, dual ascent. | ~180 |
| `src/training/dro_fair.py` | DRO-FAIR (Algorithm 1). Tilted risk, inner max on p-weights, Dykstra projection. | ~260 |
| `src/training/standard_ml.py` | Standard ML pretrainer (no fairness). | ~50 |
| `src/evaluation/metrics.py` | Accuracy, DP violation, IF violation (k-NN). | ~135 |
| `src/utils/projections.py` | Simplex + L1-ball projection via Dykstra. | ~120 |

**Total source code:** ~1,260 lines (clean, documented, tested)

---

## Experiments (`experiments/`)

| File | Purpose | When to run |
|------|---------|-------------|
| `run_experiments.py` | Main experiment runner. 150 experiments (3×5×10). Checkpointing. | After hyperparam fix |
| `run_ablations.py` | Ablation studies: DP-only, IF-only, joint, standard ML. | After main experiments |
| `run_random_vs_adversarial.py` | Compare random vs adversarial corruption strength. | Optional |
| `generate_results.py` | Generate Table 1 (CSV + LaTeX), plots, summary stats. | After experiments |
| `verify_theory.py` | Verify Theorem 4.2, 4.3, 6.1 formulas computationally. | Anytime |
| `hyperparam_sweep.py` | Test 6 hyperparameter configs. **ALREADY RAN.** | Done |
| `professor_review_simulator.py` | Self-grade against all 15 professor checks. | Before submission |
| `fix_stale_files.py` | Fix configs/default.yaml + validate imports. | Done |
| `analyze_results.py` | Advanced result analysis. | After experiments |
| `plot_convergence.py` | Plot training convergence curves. | After experiments |

---

## Tests (`tests/`)

| File | Tests |
|------|-------|
| `test_end_to_end.py` | 14 tests: DRO/Naive run, p-weights on simplex, λ clamping, tau multiply, gradients, reproducibility, etc. |
| `test_corruption.py` | 5 tests: zero α, nonzero α, coordinated targeting, random vs adversarial difference. |
| `test_metrics.py` | 4 tests: accuracy, DP zero/nonzero, IF zero/nonzero. |
| `test_projections.py` | 7 tests: simplex, L1-ball, Dykstra, random inputs, training regime. |

**Total: 30 tests** (all passing)

---

## Documentation (Root `.md` files)

| File | Audience | Purpose |
|------|----------|---------|
| `README.md` | Everyone | Project overview, installation, quick start. **NEEDS RESULTS UPDATE.** |
| `EXPLANATION_FOR_YOU.md` | You (non-technical) | Simple explanation of what the project does and what was fixed. |
| `PROFESSOR_FAQ.md` | You | 15 Q&A for professor meetings. |
| `PRESENTATION_TALKING_POINTS.md` | You | 10-slide presentation script with timing. |
| `MASTER_PROTOCOL.md` | Your agents | Full technical spec for implementation. |
| `FINAL_AGENT_BRIEFING.md` | Your agents | Exact instructions for running full experiments. |
| `AGENT_BRIEFING.md` | Your agents | Instructions for hyperparameter sweep. **Done.** |
| `AGENT_BRIEFING_2.md` | Your agents | Instructions for parallel cleanup. **Done.** |
| `FINAL_REPORT.md` | Everyone | Bug-fixing report with all 11 issues documented. |
| `analysis[12 May 9:49pm].md` | You | Original analysis from our discussion. |
| `PROF_PROMPT.md` | You | Professor's review protocol (what they'll check). |
| `AGENT_PROMPT.md` | You | Original agent instructions (historical). |
| `AGENTS.md` | You | Agent configuration. |
| `PROJECT_STRUCTURE.md` | Everyone | This file. |

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Hyperparameters. **Updated to epochs=60.** |

---

## Data

| Directory | Contents |
|-----------|----------|
| `data/raw/` | Adult (UCI), Credit (UCI), LSAC (real law school data). No synthetic data. |

---n## Results & Figures

| Directory | Contents |
|-----------|----------|
| `results/` | Experiment outputs. **Currently empty** — will be populated after full run. |
| `figures/` | Generated plots. **Currently empty** — will be populated after `generate_results.py`. |

---

## Scripts

| Directory | Contents |
|-----------|----------|
| `scripts/` | **Empty.** All ad-hoc debug scripts removed. |
| `notebooks/` | **Empty.** Stale notebook removed. |

---

## What Was Removed

**Old log files (root):**
- `ablation_run.log`, `clean_run.log`, `experiment_full.log`, `experiment_log.txt`
- `experiment_new.log`, `experiment_run.log`, `experiment_warmup.log`
- `full_run.log`, `hyperparam_sweep.log`

**Old log files (experiments/):**
- `run.log`, `ablation.log`, `ablation_corrected.log`, `run_corrected.log`

**Stale results:**
- `results/checkpoint.pkl`, `results/cycle3_*.log`, `results/fast_verify.*`
- `results/full_experiments.log`, `results/lambda_trace_*.csv`

**Ad-hoc debug scripts (scripts/):**
- `check_progress.py`, `compare_tau.py`, `demo.py`, `diagnose_lambda.py`
- `fast_verify.py`, `minimal_verify.py`, `project_status.py`
- `quick_verify.py`, `run_quick_exp.py`, `verify_cycle3.py`
- `sweep_lambda_max.py`

**Stale notebook:**
- `notebooks/analysis.ipynb`

**Stale figure:**
- `figures/lambda_trace_lsac_0.2_0.png`

---

## Verification Checklist

After cleanup, verify:

```bash
# 1. All imports work
python3 -c "from experiments.run_experiments import run_all_experiments"

# 2. All tests pass
python3 -m pytest tests/ -v

# 3. Main entry point works
python3 main.py --help

# 4. Theory verification passes
python3 experiments/verify_theory.py

# 5. End-to-end smoke test
python3 -c "from experiments.run_experiments import run_single_experiment; \
  r = run_single_experiment('adult', 0.1, seed=0); \
  print(f'Naive Acc={r[\"naive\"][\"clean\"][\"accuracy\"]:.3f}'); \
  print(f'DRO Acc={r[\"dro\"][\"clean\"][\"accuracy\"]:.3f}')"
```

All of the above ✅ verified after cleanup.
