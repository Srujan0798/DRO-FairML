# TUESDAY MEETING AGENDA — Jun 2, 2026

---

## 1. Recap of Last Week (2 min)

**What was completed:**
- Fairness-Targeted PGD (DP/IF/Combined attacks) — 270 experiments on Adult/Credit/LSAC
- UTKFace pipeline —15 experiments on GPU server (5 seeds × 3 alphas)
- Statistical analysis with Wilcoxon tests

**What was delayed:**
- GPU access issues (now resolved — server had existing PyTorch environment)
- Meeting rescheduled from Friday to Tuesday

---

## 2. Main Results (10 min)

### Tabular: Fairness-PGD Attacks

| Attack | Dataset | Alpha | Result | p-value |
|--------|---------|-------|--------|---------|
| IF | Credit | 0.2 | DRO wins:64.5% DP reduction | 0.031 |
| IF | Credit | 0.3 | DRO wins: 97.5% DP reduction | 0.031 |
| IF | LSAC | 0.3 | DRO wins: 96.2% DP reduction | 0.031 |
| DP | Adult | any | DRO loses (feedback loop) | — |

**Key message:** DRO defends well under IF attacks at high corruption levels. DRO loses under DP attacks on Adult (same feedback loop as Week 1 random corruption).

### UTKFace: Image Data

| Alpha | Condition | Result |
|-------|-----------|--------|
| 0.0 | Clean | DRO slightly better (0.023 vs 0.029) |
| 0.1 | Corrupted | Naive better (0.116 vs 0.141) |
| 0.2 | Corrupted | Naive better (0.080 vs 0.092) |

**Key message:** On image features, DRO actually makes things **worse** under corruption — opposite of tabular. With only 5 seeds, not statistically significant, but trend is consistent.

---

## 3. New Finding to Discuss (3 min)

**DRO's robustness is modality-dependent:**
- **Tabular:** DRO wins under IF attacks (features carry demographic signal)
- **Image:** DRO loses under corruption (ResNet18 features are fairness-agnostic)

**Hypothesis:** ResNet18 pretrained features don't encode demographic information. DRO's worst-case reweighting over-corrects when labels are corrupted because there's no demographic signal in features to anchor on.

---

## 4. Next Steps (3 min)

1. **More UTKFace seeds** (10+) to get statistical significance
2. **Demographic-aware features** — train fairness-aware encoder instead of pretrained ResNet
3. **CelebA/FairFace** — test on larger image datasets
4. **Theoretical analysis** — why does DRO invert on image features?

---

## 5. Questions for Madam (2 min)

1. Should we focus on tabular (where DRO works) or investigate the image feature issue?
2. Any recommended fairness-aware feature extractors?
3. Should we try different DRO formulations (CVaR, LGMs) for image data?

---

## Files to Have Ready

- `docs/ADVERSARIAL_FAIRNESS_REPORT.md`
- `figures/fig8_fairness_pgd_comparison.png`
- `figures/fig9_fairness_pgd_curves.png`
- `figures/fig_utkface_dp_comparison.png`
- `results/fairness_pgd_wilcoxon.csv`
- `results/utkface_results.json`
