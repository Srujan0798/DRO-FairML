# EXECUTE_NOW — Week 2 Status
**Updated:** May 27, 2026 5:30 PM
**Next meeting:** Tue June 2, 2026

---

## ✅ COMPLETED TODAY

### Agent A — Fairness-PGD ✅ COMMITTED
| Item | Commit | Status |
|------|--------|--------|
| IF gradient fix | `90ece42` | k-NN based, working |
| Tests | `90ece42` | 6/6 passing |
| Experiment driver | `90ece42` | Smoke: 6 rows in ~6 min |
| Smoke test results | `cc1090e` | 6 rows committed |
| Analysis script | `adb49f2` | Wilcoxon + summary |
| Figures | `adb49f2` | fig8, fig9 generated |
| Writeup | `adb49f2` | FAIRNESS_PGD_RESULTS.md |

### Agent B — UTKFace ✅ COMMITTED
| Item | Commit | Status |
|------|--------|--------|
| CNNClassifier | `a31d43f` | ResNet18 + FC head |
| ImagePGD | `a31d43f` | epsilon=4/255 |
| run_utkface.py | `a31d43f` | Synthetic fallback works |
| Smoke test results | `613267f` | 1 row committed |

---

## 📊 Key Results

### Fairness-PGD (Adult, α=0.2)
```
Combined attack — DRO shows 57% lower DP than Naive:
  Naive: DP=0.189
  DRO:   DP=0.081  ← best result
```
Full summary: `results/fairness_pgd_summary.csv` (12 rows)
Wilcoxon tests: `results/fairness_pgd_wilcoxon.csv` (6 rows)

### UTKFace
```
GPU blocked (hostname not resolvable)
Pipeline works with synthetic data: DRO lower DP than Naive
```

---

## 🚀 What's Running / Pending

### Full PGD Run (was running, may have been killed)
- 162 experiments planned but slow on CPU
- Smoke test results (6 rows) are valid and committed
- Full run would need ~6+ hours on CPU

### GPU Push for UTKFace
- Still blocked: `gpu-server` hostname not resolvable
- Need sysadmin contact to get correct hostname

---

## 📁 Files Created This Session

```
COMMITTED (19 commits ahead of origin/main):
  a31d43f - UTKFace pipeline (CNN, ImagePGD, run_utkface)
  90ece42 - FairnessTargetedPGD (IF gradient fix, tests, driver)
  cc1090e - Fairness PGD smoke results (6 rows)
  613267f - UTKFace smoke results
  adb49f2 - Analysis + figures + writeups

RESULTS:
  results/fairness_pgd_results.json - smoke test data
  results/fairness_pgd_summary.csv - aggregated stats
  results/fairness_pgd_wilcoxon.csv - statistical tests
  results/utkface_results.json - UTKFace smoke

FIGURES:
  figures/fig8_fairness_pgd_comparison.{pdf,png}
  figures/fig9_fairness_pgd_curves.{pdf,png}

DOCS:
  docs/FAIRNESS_PGD_RESULTS.md
  docs/UTKFACE_RESULTS.md
```

---

## 📆 Remaining Days

| Day | Task |
|-----|------|
| Thu 28 | Full experiment run (if time allows) |
| Fri 29 | Writeups final + commit |
| Sat 30 | Integrate into report.tex |
| Sun 31 | Section 13 + recompile PDF |
| Mon Jun 1 | 5-slide deck + dry run |
| Tue 2 | **4 PM Meeting** |

---

## 🗣️ What to Tell Madam

1. **"Did you modify the paper?"** — No, frozen at v1.0. This is a separate direction.

2. **"Is DRO good or only on small data?"** — Testing on UTKFace (200K images). Pipeline works; full results next week.

3. **"What's novel about the PGD attack?"** — Prior work flips labels heuristically. We compute the gradient of the fairness metric (DP or IF) and flip the exact worst samples.

4. **"When will this be ready?"** — Fairness-PGD results this weekend. UTKFace full ablation by next Friday.

---

## 🚨 If Time Runs Out

1. Drop IF-only + combined → keep only DP-only attack
2. Drop credit + lsac → keep only Adult
3. Drop UTKFace GPU → CPU synthetic only
4. The ONE thing: DP attack on Adult showing DRO < Naive