# DRO-FAIR — Adversarial Fairness Attacks
## Tuesday Meeting | Jun 2, 2026

---

## Slide 1: Recap — What We Did

- **Fairness-Targeted PGD:** Implemented DP/IF/Combined gradient attacks (270 experiments)
- **UTKFace Pipeline:** Built and ran on GPU server (15 experiments, 5 seeds)
- **Key finding:** DRO works on tabular, but **fails on image features** under corruption

---

## Slide 2: Tabular Results — IF Attacks

**DRO significantly outperforms Naive under IF attacks at high α:**

| Dataset | Alpha | DRO Improvement | p-value |
|---------|-------|----------------|---------|
| Credit | 0.2 | +64.5% DP reduction | 0.031 |
| Credit | 0.3 | +97.5% DP reduction | 0.031 |
| LSAC | 0.3 | +96.2% DP reduction | 0.031 |

**At α=0.1:** No significant difference (attack too weak)

---

## Slide 3: Tabular Results — DP Attacks

**DRO collapses under DP-targeted attacks on Adult:**

- Same feedback loop as Week 1 random corruption
- Adversary is strong enough to break DRO's worst-case reweighting
- Adult's demographic correlations allow adversary to flip labels that maximize DP

**This is expected** — Adult has known feedback loop issue.

---

## Slide 4: UTKFace — New Finding

**On image features, DRO makes things WORSE under corruption:**

| Alpha | Naive DP | DRO DP | Winner |
|-------|----------|--------|--------|
| 0.0 (clean) | 0.029 | 0.023 | DRO |
| 0.1 (corrupt) | 0.116 | 0.141 | **Naive** |
| 0.2 (corrupt) | 0.080 | 0.092 | **Naive** |

**Consistent trend across 5 seeds, but not statistically significant (p > 0.05)**

---

## Slide 5: Why Does DRO Fail on Images?

**Hypothesis:**

ResNet18 features are pretrained on ImageNet — they contain **no demographic information**.

When labels are corrupted:
- DRO's worst-case reweighting has no demographic signal to anchor on
- DRO over-corrects, making fairness worse
- Naive ERM is more robust because it doesn't try to optimize worst-case

On tabular data, features naturally carry demographic correlations → DRO can find robust patterns.

---

## Slide 6: Next Steps

1. **Run more UTKFace seeds (10+)** to get statistical significance
2. **Demographic-aware features** — train fairness-aware encoder instead of pretrained ResNet
3. **CelebA/FairFace** — test on larger image datasets
4. **Theoretical analysis** — why does DRO invert on image features?

---

## Slide 7: Questions for Madam

1. Should we focus on tabular (where DRO works) or investigate the image feature issue?
2. Any recommended fairness-aware feature extractors?
3. Should we try different DRO formulations (CVaR, LGMs) for image data?

---

## Key Takeaway

**DRO's robustness is modality-dependent:**
- Tabular: DRO wins under IF attacks ✓
- Image: DRO loses under corruption ✗

**This is a new finding worth investigating — could become a paper.**
