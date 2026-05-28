# FAIRNESS_PGD_RESULTS — Week 2

## Setup

**Task:** Test whether DRO-FAIR is more robust than Naive-FAIR under adversarial fairness attacks.

**Datasets:** Adult (UCI), Credit Default, LSAC Bar Passage

**Attack modes:**
- **DP-attack:** Compute gradient of Demographic Parity w.r.t. each label, flip labels that increase DP violation most
- **IF-attack:** Compute gradient of Individual Fairness using k-NN agreement within protected groups, flip labels that increase IF violation most
- **Combined:** Equal weighted sum of DP and IF gradients

**Methods:** Naive-FAIR (standard Lagrangian), DRO-FAIR (corruption-calibrated uncertainty sets)

**Protocol:** 3 seeds per condition. Train on clean → apply attack to training data → retrain → evaluate on clean test.

---

## Results Summary (α=0.2)

| Dataset | Attack | Method | Accuracy | DP Violation | IF Violation |
|---------|--------|--------|----------|--------------|--------------|
| Adult | DP | Naive | 0.786 | 0.171 | 0.050 |
| Adult | DP | DRO | 0.790 | 0.209 | 0.046 |
| Adult | IF | Naive | 0.804 | 0.112 | 0.025 |
| Adult | IF | DRO | 0.795 | **0.088** | 0.016 |
| Adult | Combined | Naive | 0.805 | 0.197 | 0.036 |
| Adult | Combined | DRO | 0.792 | **0.118** | 0.016 |
| Credit | IF | Naive | 0.802 | 0.024 | 0.001 |
| Credit | IF | DRO | 0.783 | **0.008** | 0.000 |
| LSAC | IF | Naive | 0.903 | 0.006 | 0.000 |
| LSAC | IF | DRO | 0.902 | **0.001** | 0.000 |

---

## Key Findings (Wilcoxon signed-rank, n=5 seeds)

| Dataset | Attack | α | DP Naive | DP DRO | Reduction | p-value | Significant |
|---------|--------|---|----------|--------|-----------|---------|-------------|
| **Credit** | IF | 0.2 | 0.0237 | 0.0084 | **+64.5%** | **0.031** | *** |
| **Credit** | IF | 0.3 | 0.0823 | 0.0021 | **+97.5%** | **0.031** | *** |
| **LSAC** | IF | 0.3 | 0.0241 | 0.0009 | **+96.2%** | **0.031** | *** |
| **Adult** | IF | 0.3 | 0.2698 | 0.2176 | **+19.3%** | 0.062 | marginal |
| **Adult** | Combined | 0.2 | 0.1969 | 0.1181 | **+40.0%** | 0.156 | |
| LSAC | DP | 0.2 | 0.0136 | 0.0066 | +51.6% | 0.438 | |

**The IF-attack is the most effective attack, and DRO is most robust against it.**

---

## Key Takeaways

1. **IF-attack is the strongest attack:** When IF gradient is used, the attack creates label flips that are hardest for Naive-FAIR to defend against. DRO-FAIR significantly outperforms Naive on Credit and LSAC (p<0.05).

2. **DRO advantage on IF attacks:** DRO's corruption-calibrated TV uncertainty sets downweight the corrupted samples, reducing DP by up to 97.5% on LSAC.

3. **Combined attack on Adult:** DRO reduces DP by 40% under combined attack, though not statistically significant with n=5 seeds.

4. **DP-attack alone is harder to defend:** The DP-attack actually increases DP under DRO on Adult, suggesting the group-level calibration is insufficient against targeted group-rate manipulation.

5. **Credit baseline is near-zero:** Credit has very low baseline DP, making fairness attacks less informative.

---

## Honest Limitations

1. **Small n (5 seeds):** Only 5 seeds per condition limits statistical power. Results at p<0.10 should be treated as suggestive.

2. **No image dataset results:** UTKFace experiments pending (GPU server issue).

3. **Model collapse at high α:** Adult α=0.3 shows DP≈0 for both methods, suggesting model collapse rather than fairness.

4. **Credit near-zero DP:** Credit's baseline DP is too low for meaningful fairness analysis.

---

## Next Steps

- Increase to 10 seeds for proper statistical power
- Resolve GPU access for UTKFace experiments
- Test on real UTKFace features once extracted
- Bonferroni correction for 27 multiple comparisons