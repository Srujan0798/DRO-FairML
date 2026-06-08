# DRO-FairML Experiment Status

> Auto-updated dashboard. Last refresh: see git timestamp.

## 🔄 Active Runs

### Tabular Re-run (270 experiments)
- **Status:** RUNNING (no timeout)
- **Progress:** See `logs/tabular_rerun_270.log`
- **Results:** `results/fairness_pgd_results.json`
- **Config:** 3 datasets × 3 alphas × 5 seeds × 3 attacks × 2 methods = 270
- **Features:** Incremental save + resume support

### Lambda Diagnostic (12 runs)
- **Status:** RUNNING
- **Progress:** See `logs/lambda_diagnostic_full.log`
- **Results:** `results/lambda_diagnostic_full.json`
- **Config:** adult lmax=1.5 (3 seeds), adult lmax=0.5 (3 seeds), credit lmax=1.5 (3 seeds), lsac lmax=1.5 (3 seeds)

## ✅ Completed Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Audit `src/corruption/adversarial.py` + fix 4 critical bugs | ✅ Done |
| 2 | Audit `src/corruption/image_pgd.py` + fix device mismatch | ✅ Done |
| 3 | Add stronger tests (`test_greedy_attack_superiority.py`) | ✅ Done |
| 4 | All 42 tests passing | ✅ Done |
| 5 | Commit + push all fixes | ✅ Done |
| 6 | Restart tabular re-run (incremental save, no timeout) | ✅ Done |
| 7 | Restart lambda diagnostic | ✅ Done |
| 8 | Create `auto_finalize.py` post-processing pipeline | ✅ Done |
| 9 | Create `generate_paper_tables.py` | ✅ Done |
| 10 | Create `generate_report_tables.py` | ✅ Done |

## 🐛 Bugs Fixed

1. **DP attack used batched PGD on discrete flips** → Replaced with greedy algorithm
2. **DP gradient used uniform ±1 magnitude** → Now uses exact marginal gain `±1/count_g`
3. **Combined attack: IF dominated DP 1000×** → Normalized both to [-1,1] before mixing
4. **Baseline heuristic: stale group rates** → Recomputed after each flip
5. **Coordinated targeting NaN bug** → Uses `np.full` masked arrays
6. **Image PGD device mismatch** → Fixed `X_max`/`X_min` tensor handling
7. **JSON serialization** → All history values cast to `float()`
8. **Tabular runner: no incremental save** → Saves after every experiment
9. **Tabular runner: no resume support** → Skips already-completed experiments
10. **Analyze script: missing `makedirs`** → Added to `plot_alpha_curves`
11. **Generate figures: KeyError risk** → Added `.get()` guard
12. **Data splits: label-only stratification** → Joint label+group stratification
13. **UTKFace: `a=race` (5-class) breaks binary trainer** → Override to `a=gender`
14. **IF attack: recomputed k-NN 3000× per run** → Precompute once, reuse (CRITICAL speedup)

## 📊 Preliminary Result (Fixed Attack)

```
adult α=0.2 λ_max=1.5 seed=0:
  acc=0.823, dp=0.1389, lambda_dp_final=0.050
```

Old buggy result: `dp=0.047` (attack was too weak). New attack is ~3× stronger.

## ⏱️ Estimated Timeline

- Lambda diagnostic: ~1-2 hours
- Tabular re-run: ~12-18 hours
- Auto-finalize (figures + tables + push): ~5 minutes after completion

## 🚀 Next Steps After Completion

1. `venv/bin/python3 experiments/auto_finalize.py`
2. Review `RESULTS_SUMMARY.md`
3. Check `figures/` for new plots
4. Check `paper/auto_generated/` for updated tables
5. Check `report/sections/` for updated tables

## 📁 Key Files

| File | Purpose |
|------|---------|
| `logs/tabular_rerun_270.log` | Tabular experiment log |
| `logs/lambda_diagnostic_full.log` | Lambda diagnostic log |
| `results/fairness_pgd_results.json` | Tabular results (incremental) |
| `results/lambda_diagnostic_full.json` | Lambda diagnostic results |
| `experiments/auto_finalize.py` | One-command post-processing |
| `experiments/generate_paper_tables.py` | Paper table generation |
| `experiments/generate_report_tables.py` | Report table generation |
