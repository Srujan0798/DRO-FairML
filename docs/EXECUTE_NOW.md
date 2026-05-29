# EXECUTE_NOW — Week 2 Final Status
**Updated:** May 27, 2026 6:30 PM
**Next meeting:** Tue June 2, 2026

---

## ✅ ALL TASKS COMPLETED

### Fairness-PGD ✅ DONE & PUSHED
| Item | Commit | Status |
|------|--------|--------|
| IF gradient fix | `90ece42` | k-NN based, working |
| Tests (6/6) | `90ece42` | All passing |
| Experiment driver | `90ece42` | Working |
| Full run (270 exp) | `fd545d1` | 3 datasets × 3 alphas × 5 seeds × 3 attacks × 2 methods |
| Analysis + figures | `fd545d1` | Wilcoxon, fig8, fig9 |
| Writeup | `6112906` | FAIRNESS_PGD_RESULTS.md + ADVERSARIAL_FAIRNESS_REPORT.md |

### UTKFace 🔄 GPU BLOCKED
| Item | Status |
|------|--------|
| CNNClassifier | ✅ Committed |
| ImagePGD | ✅ Committed |
| run_utkface.py | ✅ Committed (synthetic fallback works) |
| GPU server | ❌ `10.0.62.234` unreachable from laptop |
| Real UTKFace | 🔄 Pending GPU access |

---

## 📊 Key Results (270 Experiments)

**Most significant finding:** IF-attack is the strongest attack. DRO significantly outperforms Naive under IF attacks:

| Dataset | Attack | α | DP Reduction | p-value |
|---------|--------|---|-------------|---------|
| Credit | IF | 0.2 | **+64.5%** | 0.031 *** |
| Credit | IF | 0.3 | **+97.5%** | 0.031 *** |
| LSAC | IF | 0.3 | **+96.2%** | 0.031 *** |
| Adult | IF | 0.3 | +19.3% | 0.062 (marginal) |

---

## 🗣️ What to Tell Madam (Tuesday June 2)

1. **"Did you modify the paper?"** — "No, Madam. Frozen at v1.0. This is a separate direction."

2. **"What is the main finding?"** — "The IF-attack (gradient-based individual fairness attack) is the most effective attack. DRO-FAIR reduces DP violation by 64-97% under IF attacks on Credit and LSAC."

3. **"Is DRO good on large data?"** — "GPU server is blocked. UTKFace pipeline works on synthetic data. Full results pending server access."

4. **"What is novel?"** — "Prior work flips labels heuristically. We compute exact gradient of fairness metric and flip the exact worst samples."

---

## 📁 Key Files (on GitHub)

```
COMMIT: 6f11f2f "Update ADVERSARIAL_FAIRNESS_REPORT: 270 experiments complete, GPU blocked"

FIGURES:
  figures/fig8_fairness_pgd_comparison.png
  figures/fig9_fairness_pgd_curves.png

RESULTS:
  results/fairness_pgd_results.json (270 rows)
  results/fairness_pgd_summary.csv (36 rows)
  results/fairness_pgd_wilcoxon.csv (27 rows)

DOCS:
  docs/FAIRNESS_PGD_RESULTS.md
  docs/ADVERSARIAL_FAIRNESS_REPORT.md

CODE:
  src/corruption/adversarial.py (FairnessTargetedPGD)
  src/models/cnn_classifier.py (CNNClassifier)
  src/corruption/image_pgd.py (ImagePGD)
  experiments/run_fairness_pgd.py
  experiments/analyze_fairness_pgd.py
  experiments/run_utkface.py
  tests/test_fairness_pgd.py
```

---

## 📆 Remaining Days

| Day | Task |
|-----|------|
| Thu 28 | UTKFace GPU resolution (email sysadmin) |
| Fri 29 | Writeups + report.tex Section 13 |
| Sat 30 | Integrate into report.tex |
| Sun 31 | Section 13 + recompile PDF |
| Mon Jun 1 | 5-slide deck + dry run |
| Tue 2 | **4 PM Meeting** |

---

## 🚨 Remaining Blockers

1. **GPU server** — `10.0.62.234` unreachable. Need sysadmin confirmation.
2. **UTKFace** — Can't run 200K image CNN without GPU.
3. **report.tex Section 13** — Need to add Week 2 results.

---

## 🚨 If Time Runs Out

1. Drop UTKFace → present pipeline only, say "results next week"
2. Drop combined attack → keep only IF-attack (strongest result)
3. Drop credit + lsac → keep only Adult IF-attack result (p=0.062)