# EXECUTE_NOW — Week 2 Status
**Updated:** May 27, 2026 4:30 PM
**Next meeting:** Tue June 2, 2026

---

## ✅ COMPLETED TODAY

| Task | Commit | Notes |
|------|--------|-------|
| Fix IF gradient (`compute_if_gradient`) | `90ece42` | k-NN based within-group label agreement |
| Write tests (`tests/test_fairness_pgd.py`) | `90ece42` | 6 tests: DP, IF, combined, budget, minority, API |
| Write experiment driver (`run_fairness_pgd.py`) | `90ece42` | Smoke: 6 rows in ~2.5 min |
| Smoke test PASSED | `results/fairness_pgd_results.json` | Valid JSON, 6 rows |
| Launch full run | PID 64923 | 162 experiments (3 datasets × 3 alphas × 3 attacks × 2 methods × 3 seeds) |
| UTKFace pipeline | `a31d43f` | CNNClassifier + ImagePGD + run_utkface.py |

---

## 🧨 What MUST be true by Tuesday 4 PM

1. **Task 1 (Fairness-PGD):** Results table with DP/IF under each attack, Naive vs DRO comparison
2. **Task 2 (UTKFace):** Pipeline working, baseline results if GPU secured
3. **GitHub:** v1.1 tagged, README updated
4. **Slides:** 5-minute deck ready (Sunday night)

---

## 📊 Current Run Status

**Full PGD run:** `PID 64923` — running in background
```
Expected: 162 experiments
At: ~experiment 2/162 (adult α=0.1 seed=0)
Est. time: ~90 min at ~30s/exp
```

---

## 📋 Full Experiment Grid

```
3 datasets:     adult, credit, lsac
3 alphas:       0.1, 0.2, 0.3
3 attacks:      dp, if, combined
2 methods:      naive, dro
3 seeds:        0, 1, 2
─────────────────────────
Total:          162 experiments
```

---

## 🔀 Agent Status

### Agent A (Fairness-PGD) — ✅ DONE, commit `90ece42`
- IF gradient: k-NN based, works correctly
- Tests: 6/6 passing
- Driver: smoke passes, full run launched

### Agent B (UTKFace) — ✅ DONE, commit `a31d43f`
- CNNClassifier: committed
- ImagePGD: committed
- run_utkface.py: committed (synthetic fallback works)
- GPU: still blocked (hostname not resolvable)

---

## 📆 Remaining Days

| Day | Date | Task |
|-----|------|------|
| Thu | May 28 | Full run completes → aggregate + figures |
| Fri | May 29 | Writeups + final commits |
| Sat | May 30 | Integrate into report.tex |
| Sun | May 31 | Section 13 + recompile PDF |
| Mon | Jun 1 | 5-slide deck + dry run |
| Tue | Jun 2 | **4 PM Meeting** |

---

## 🚨 If Time Runs Out — Cut In This Order

1. Drop IF-only + combined attacks → keep only DP-only
2. Drop credit + lsac → keep only Adult (prof asked about Adult)
3. Drop UTKFace GPU → CPU synthetic baseline only
4. The ONE thing you must have: DP attack on Adult showing DRO < Naive

---

## 📁 Key Files

```
COMMITTED:
  a31d43f - UTKFace pipeline (CNN, ImagePGD, run_utkface)
  90ece42 - FairnessTargetedPGD (IF gradient fix, tests, driver)

RESULTS:
  results/fairness_pgd_results.json - smoke test (6 rows)
  results/utkface_results.json - UTKFace smoke (1 row)
  logs/full_fpgd.log - running experiment output
```