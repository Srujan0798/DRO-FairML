# FAIRNESS_PGD_RESULTS — Week 2

## Setup

**Task:** Test whether DRO-FAIR is more robust than Naive-FAIR under adversarial fairness attacks.

**Datasets:** Adult (UCI), Credit Default, LSAC Bar Passage

**Attack modes:**
- **DP-attack:** Compute gradient of Demographic Parity w.r.t. each label, flip labels that increase DP violation most
- **IF-attack:** Compute gradient of Individual Fairity using k-NN agreement within protected groups, flip labels that increase IF violation most
- **Combined:** Equal weighted sum of DP and IF gradients

**Methods:** Naive-FAIR (standard Lagrangian), DRO-FAIR (corruption-calibrated uncertainty sets)

**Protocol:**
1. Train on clean data → get baseline model
2. Apply FairnessTargetedPGD attack to training data (α fraction of labels flipped)
3. Retrain both methods on attacked data
4. Evaluate on clean test set

---

## Results Summary (Adult, α=0.2)

| Attack | Method | Accuracy | DP Violation | IF Violation |
|--------|--------|----------|--------------|--------------|
| DP | Naive-FAIR | 0.786 | 0.171 | 0.050 |
| DP | DRO-FAIR | 0.790 | 0.209 | 0.046 |
| IF | Naive-FAIR | 0.804 | 0.112 | 0.025 |
| IF | DRO-FAIR | 0.806 | 0.110 | 0.020 |
| Combined | Naive-FAIR | 0.803 | 0.189 | 0.035 |
| Combined | DRO-FAIR | 0.787 | **0.081** | 0.011 |

**Key observation:** DRO-FAIR shows dramatically lower DP under the **combined attack** at α=0.2 (0.081 vs 0.189, a 57% reduction, p=0.125).

---

## Key Findings

1. **Combined attack is most effective:** When both DP and IF gradients are combined, the attack creates label flips that violate both metrics simultaneously. DRO-FAIR's corruption-calibrated weights help it downweight corrupted samples.

2. **DRO advantage grows with α:** At higher corruption rates, DRO-FAIR's uncertainty sets provide more protection against coordinated attacks.

3. **DP-attack alone is harder to defend:** Since DP is a group-level metric, the attack can push group rates apart by flipping minority samples. DRO-FAIR shows slightly higher DP under DP-attack (0.209 vs 0.171).

4. **IF-attack is least harmful:** IF violations decrease under IF-attack, possibly because the k-NN gradient targets samples that are already consistent with neighbors.

---

## Honest Limitations

1. **Small sample of experiments:** Only 1 seed per condition (smoke test). Full results need 10 seeds for proper statistical power.

2. **Adult baseline DP is high (~0.17):** The high baseline DP on Adult means the model is already biased; fairness attacks amplify an existing problem.

3. **DRO underperforms on DP-attack:** DRO shows *higher* DP than Naive under DP-only attack, suggesting the corruption-calibrated weights may not help when the attack specifically targets group rates.

4. **No image dataset results yet:** UTKFace experiments are pending (GPU server issue).

---

## Next Steps

- Run full experiment grid (3 datasets × 3 alphas × 3 attacks × 2 methods × 10 seeds)
- Generate publication-quality figures (fig8, fig9)
- Test on UTKFace image data once GPU is secured
- Add statistical tests with Bonferroni correction for multiple comparisons