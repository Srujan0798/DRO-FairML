# Submission Package Manifest

**Project:** DRO-FAIR: Robust Individual and Group Fair Classification  
**Version:** v1.0.1  
**Date:** 2026-05-18  
**Author:** Srujan Sai  
**Repository:** https://github.com/Srujan0798/DRO-FairML  
**Release:** https://github.com/Srujan0798/DRO-FairML/releases/tag/v1.0

---

## Contents

### Report
| File | Size | Description |
|------|------|-------------|
| `report.pdf` | 315 KB | Main LaTeX report (compiled from `report/report.tex`) |

### Figures (7 + 2 extras)
| File | Description |
|------|-------------|
| `fig1_main_results.png` | DRO-FAIR vs Naive-FAIR across all datasets/metrics |
| `fig2_dp_reduction_heatmap.png` | DP reduction percentage heatmap |
| `fig3_robustness_clean_vs_corrupted.png` | Clean vs corrupted test robustness |
| `fig4_significance_matrix.png` | Wilcoxon significance matrix (DP + IF) |
| `fig5_accuracy_fairness_tradeoff.png` | Accuracy vs DP Pareto frontier |
| `fig6_seed_stability.png` | Per-seed accuracy boxplots |
| `fig7_summary_win_rates.png` | Win-rate summary (6/9 DP, 5/9 IF) |
| `main_results.png` | Alternative main results view |
| `test_time_eval.png` | Test-time evaluation comparison |

### Data
| File | Description |
|------|-------------|
| `all_results.json` | **Authoritative ground truth:** 150 experiments |
| `table1.csv` | Aggregated mean ± std table |

### Code
| Path | Description |
|------|-------------|
| `src/` | Full source code (training, corruption, evaluation, data, models, utils) |
| `run_experiments.py` | Main experiment runner |
| `default.yaml` | Hyperparameter configuration |

### Documentation
| File | Description |
|------|-------------|
| `ORC_PROMPT.md` | Omnipotent Review Catalyst prompt |
| `VERIFICATION_PROMPT.md` | Full verification protocol |
| `PROFESSOR_REVIEW_PROMPT.md` | Professor review rubric |

---

## Verification Status

- [x] 150 experiments complete
- [x] 0 NaN/Inf values
- [x] DP wins: 6/9 (Wilcoxon p<0.05)
- [x] IF wins: 5/9 (Wilcoxon p<0.05)
- [x] Credit α=0.3: DP −91.8%, IF −96%, acc drop −1.9 pp
- [x] LSAC α=0.3: DP −99.6%, IF −100%, acc drop −0.1 pp
- [x] All theoretical formulas verified (8/8 checks)
- [x] Report PDF compiled from current LaTeX source
- [x] Submission package internally consistent

## Known Issues (Documented)

1. **Adult α≥0.3:** DRO-FAIR collapses (accuracy ~25–40% on clean test) due to adversarial feedback loop. This is an honest empirical limitation documented in report Section 5.
2. **Runtime overhead:** ~37.5× on CPU vs paper's ~12× on GPU.
