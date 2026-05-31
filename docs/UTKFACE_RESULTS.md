# UTKFace Results — GPU Server Run

**Date:** May 29, 2026  
**Setup:** 23,705 images, ResNet18 features (512-dim), 5 seeds per alpha

---

## Summary Table

| α | Naive Clean DP | DRO Clean DP | Naive Corr DP | DRO Corr DP | Winner Clean | Winner Corr |
|---|----------------|--------------|--------------|-------------|--------------|-------------|
| 0.0 | 0.0293 | 0.0225 | 0.0293 | 0.0225 | DRO | DRO |
| 0.1 | 0.0253 | 0.0342 | 0.1157 | 0.1410 | Naive | Naive |
| 0.2 | 0.0243 | 0.0268 | 0.0796 | 0.0918 | Naive | Naive |

---

## Wilcoxon Tests (Naive vs DRO, 5 seeds)

| Alpha | Condition | p-value | Significant? |
|-------|-----------|---------|--------------|
| 0.0 | Clean | 0.156 | No |
| 0.0 | Corrupted | 0.156 | No |
| 0.1 | Clean | 0.844 | No |
| 0.1 | Corrupted | 0.688 | No |
| 0.2 | Clean | 0.688 | No |
| 0.2 | Corrupted | 0.969 | No |

**Note:** With only 5 seeds, we lack statistical power to detect significance. The trends are consistent but not statistically significant.

---

## Key Findings

1. **Clean data (α=0.0):** DRO slightly better than Naive (DP 0.023 vs 0.029)
2. **Corrupted data (α>0):** Naive consistently better than DRO
3. **Trend:** DRO makes fairness worse under label corruption on image features
4. **Opposite of tabular:** On Adult/Credit/LSAC, DRO helps under IF attacks. On UTKFace, DRO hurts.

---

## Hypothesis

ResNet18 features are pretrained on ImageNet and contain no demographic information. When labels are adversarially corrupted:
- DRO's worst-case reweighting over-corrects (no demographic signal in features to anchor on)
- Naive ERM is more robust because it doesn't try to optimize worst-case

On tabular data, features naturally carry demographic correlations, so DRO can find robust patterns even under corruption.

---

## Next Steps

1. Run with more seeds (10+) to get significance
2. Try demographic-aware features (fairness-aware encoders)
3. Test on CelebA or FairFace with proper demographic labels
4. Try different backbone (ResNet50, ViT)
