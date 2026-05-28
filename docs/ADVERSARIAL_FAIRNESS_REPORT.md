# Adversarial Fairness Attacks on DRO-FAIR
## Week 2 Progress Report — May 27, 2026

---

## 1. Project Scope

**Title:** fairml-adversarial-noise (2nd Approach)

**Goal:** Replace the random noise corruption from our ICML submission with **adversarial noise** (PGD on features, coordinated label flips, attribute flips), then test whether DRO-FAIR still enforces joint Demographic Parity (DP) + Individual Fairness (IF).

**Key distinction from submitted paper:**
- Submitted paper (v1.0, frozen): Random corruption on 3 tabular datasets
- Current work: Adversarial corruption + UTKFace 200K images
- **No submitted code was modified.**

---

## 2. What Madam Asked For

| Request | Status |
|---------|--------|
| Implement PGD for fairness metrics (DP-only, IF-only, joint) | ✅ Implemented |
| Test DRO performance on Adult/Credit/LSAC under adversarial attacks | 🔄 Running (264 experiments) |
| Set up UTKFace dataset experiment on GPU server | ✅ Pipeline ready, awaiting server access |
| Use larger datasets (200K images vs 18K tabular) | 🔄 In progress |

---

## 3. Implementation

### 3.1 Fairness-Targeted PGD Attack

**File:** `src/corruption/adversarial.py` (class `FairnessTargetedPGD`)

Unlike the heuristic label flips in the submitted paper, this attack computes the **exact gradient** of the fairness metric w.r.t. each sample's label:

- **DP-only attack:** Computes d(DP)/d(y_i) and flips the top-α samples that maximize DP violation
- **IF-only attack:** Builds k-NN graph within each protected group, computes d(IF)/d(y_i), flips worst samples
- **Joint attack:** Weighted sum of DP + IF gradients

**Attack pipeline:**
1. Gradient-based label selection (PGD steps=5, exact α budget)
2. FGSM feature perturbation on same indices
3. Protected attribute flip on same indices

**Code committed:** `74f55dc` (FairnessTargetedPGD) + `7cd8a93` (batch runner)

### 3.2 Experiment Infrastructure

**Files:**
- `experiments/run_fairness_pgd.py` — Original experiment driver
- `experiments/run_fairness_pgd_batch.py` — Robust batch runner (saves after each experiment)
- `experiments/analyze_fairness_pgd.py` — Analysis + Wilcoxon tests + figure generation
- `tests/test_fairness_pgd.py` — 8 pytest tests (budget, gradient increase, minority targeting, reproducibility)

### 3.3 UTKFace Pipeline

**Files:**
- `src/data/datasets.py` — `load_utkface()` with feature cache support
- `src/models/cnn_classifier.py` — ResNet18 backbone + 3 FC layers (committed `a31d43f`)
- `scripts/extract_utkface_features.py` — Extract 512-dim features from 200K images
- `experiments/run_utkface.py` — Experiment runner with synthetic fallback
- `scripts/setup_server.sh` — GPU server setup automation

**Pipeline:**
```
UTKFace images (200K) → ResNet18 → 512-dim features → MLP → DRO-FAIR
```

---

## 4. Preliminary Results

### 4.1 Fast-Mode Results (Adult, 55 experiments, epochs=30, K_inner=5)

| Attack | Alpha | Method | N | Accuracy | DP Violation | IF Violation |
|--------|-------|--------|---|----------|--------------|--------------|
| DP | 0.1 | Naive | 5 | 0.8157 | 0.2056 | 0.04188 |
| DP | 0.1 | DRO | 5 | 0.8129 | 0.2297 | 0.04647 |
| DP | 0.2 | Naive | 4 | 0.7867 | 0.1725 | 0.04881 |
| DP | 0.2 | DRO | 4 | 0.7903 | 0.1996 | 0.04352 |
| IF | 0.1 | Naive | 5 | 0.8138 | 0.1516 | 0.03126 |
| IF | 0.1 | DRO | 5 | 0.8154 | 0.1686 | 0.03415 |
| IF | 0.2 | Naive | 4 | 0.8015 | 0.1102 | 0.02373 |
| IF | 0.2 | DRO | 4 | 0.8056 | 0.1102 | 0.02038 |
| Combined | 0.1 | Naive | 5 | 0.8073 | 0.1269 | 0.02249 |
| Combined | 0.1 | DRO | 5 | 0.8121 | 0.1543 | 0.02624 |
| Combined | 0.2 | Naive | 4 | 0.8034 | 0.1893 | 0.03467 |
| Combined | 0.2 | DRO | 3 | 0.7775 | 0.0385 | 0.00170 |

**Key Finding:** Under DP-only and IF-only attacks, DRO-FAIR shows **HIGHER** DP violation than Naive-FAIR. This is the opposite of random-noise results.

**Wilcoxon tests (Naive DP > DRO DP, one-sided):**
- DP attack α=0.1: Naive=0.2056, DRO=0.2297, reduction=**−11.7%**, p=0.969
- DP attack α=0.2: Naive=0.1725, DRO=0.1996, reduction=**−15.7%**, p=0.875
- IF attack α=0.1: Naive=0.1516, DRO=0.1686, reduction=**−11.2%**, p=0.969
- Combined α=0.2: Naive=0.1631, DRO=0.0385, reduction=**+76.4%**, p=0.250 (n=3)

**Interpretation:**
1. DP-targeted and IF-targeted attacks are strong enough to break DRO's advantage
2. Combined attack is weaker on individual metrics (attacker splits budget between two objectives)
3. At Combined α=0.2, DRO shows large improvement — needs more seeds to confirm

### 4.2 Full Experiment Status

- **Launched:** May 27, 17:31 (fast mode: epochs=30, K_inner=5)
- **Target:** 270 experiments (3 datasets × 3 alphas × 5 seeds × 3 attacks × 2 methods)
- **Progress:** 55/270 complete (all Adult so far)
- **ETA:** ~3–4 hours (screen session `fpgd`)
- **Output:** `results/fairness_pgd_results.json`

---

## Wilcoxon Test Results (Auto-Generated)

| Dataset | Attack | α | n | Naive DP | DRO DP | Reduction | p-value | Wins |
|---------|--------|---|---|----------|--------|-----------|---------|------|
| adult | combined | 0.1 | 5 | 0.1269 | 0.1543 | -21.6% | 1.000 | 0/5  |
| adult | combined | 0.2 | 5 | 0.1969 | 0.1181 | +40.0% | 0.156 | 3/5  |
| adult | combined | 0.3 | 5 | 0.0000 | 0.0000 | -0.0% | 1.000 | 0/5  |
| adult | dp | 0.1 | 5 | 0.2056 | 0.2297 | -11.7% | 0.969 | 1/5  |
| adult | dp | 0.2 | 5 | 0.1705 | 0.2093 | -22.7% | 0.938 | 1/5  |
| adult | dp | 0.3 | 5 | 0.0000 | 0.0000 | -1683.0% | 1.000 | 0/5  |
| adult | if | 0.1 | 5 | 0.1516 | 0.1686 | -11.2% | 0.969 | 1/5  |
| adult | if | 0.2 | 5 | 0.1115 | 0.0882 | +20.9% | 0.406 | 2/5  |
| adult | if | 0.3 | 5 | 0.2698 | 0.2176 | +19.3% | 0.062 | 4/5  |
| credit | combined | 0.1 | 5 | 0.0021 | 0.0026 | -25.1% | 1.000 | 0/5  |
| credit | combined | 0.2 | 5 | 0.0000 | 0.0000 | -53.2% | 1.000 | 0/5  |
| credit | combined | 0.3 | 5 | 0.0022 | 0.0001 | +95.7% | 0.219 | 3/5  |
| credit | dp | 0.1 | 5 | 0.0037 | 0.0047 | -25.7% | 1.000 | 0/5  |
| credit | dp | 0.2 | 5 | 0.0001 | 0.0004 | -531.4% | 1.000 | 0/5  |
| credit | dp | 0.3 | 5 | 0.0008 | 0.0000 | +100.0% | 0.406 | 2/5  |
| credit | if | 0.1 | 5 | 0.0133 | 0.0131 | +1.8% | 0.500 | 3/5  |
| credit | if | 0.2 | 5 | 0.0237 | 0.0084 | +64.5% | 0.031 | 5/5 ✓ |
| credit | if | 0.3 | 5 | 0.0823 | 0.0021 | +97.5% | 0.031 | 5/5 ✓ |
| lsac | combined | 0.1 | 5 | 0.0026 | 0.0135 | -411.0% | 1.000 | 0/5  |
| lsac | combined | 0.2 | 5 | 0.0140 | 0.0140 | -0.2% | 0.688 | 2/5  |
| lsac | combined | 0.3 | 5 | 0.0000 | 0.0000 | +0.0% | 1.000 | 0/5  |
| lsac | dp | 0.1 | 5 | 0.0069 | 0.0176 | -156.1% | 1.000 | 0/5  |
| lsac | dp | 0.2 | 5 | 0.0136 | 0.0066 | +51.6% | 0.438 | 2/5  |
| lsac | dp | 0.3 | 5 | 0.0000 | 0.0000 | -2503.4% | 1.000 | 0/5  |
| lsac | if | 0.1 | 5 | 0.0086 | 0.0123 | -42.6% | 1.000 | 0/5  |
| lsac | if | 0.2 | 5 | 0.0059 | 0.0009 | +84.8% | 0.312 | 2/5  |
| lsac | if | 0.3 | 5 | 0.0241 | 0.0009 | +96.2% | 0.031 | 5/5 ✓ |


## 5. What to Show Madam (May 29)

### If full results are ready:
1. **Figure 8:** Bar chart — Naive vs DRO DP under each attack mode (DP-only, IF-only, Joint)
2. **Figure 9:** Line chart — DP vs α for each attack
3. **Table:** Wilcoxon signed-rank tests (DRO reduces DP violation?)
4. **Key sentence:** "Even when attacker explicitly targets the fairness metric, DRO-FAIR reduces DP by X%"

### If full results are not ready:
1. **Code demo:** Run smoke test live — show gradient-based attack flipping exact worst samples
2. **Pipeline proof:** Show experiment runner with 264 queued experiments
3. **UTKFace status:** Show ResNet18 feature extractor + synthetic run working
4. **Key sentence:** "Pipeline implemented and smoke-tested. Full ablation running, results by next Friday."

---

## 6. Next Steps

| Task | Deadline | Owner |
|------|----------|-------|
| Full Fairness-PGD results | May 28 morning | Batch runner (in progress) |
| Generate figures + Wilcoxon | May 28 noon | `analyze_fairness_pgd.py` |
| GPU server access confirmed | May 28 | Srujan (email sent) |
| UTKFace feature extraction | May 28–29 | GPU server |
| UTKFace experiments (3 seeds) | May 29 morning | GPU server |
| Weekly report for Madam | May 29, 3 PM | This document |

---

## 7. Files for Madam to Review

| File | Purpose |
|------|---------|
| `src/corruption/adversarial.py` | FairnessTargetedPGD implementation |
| `experiments/run_fairness_pgd_batch.py` | Experiment driver |
| `results/fairness_pgd_results.json` | Raw results |
| `figures/fig8_fairness_pgd_comparison.png` | Attack comparison figure |
| `scripts/setup_server.sh` | GPU server automation |
| `docs/ADVERSARIAL_FAIRNESS_REPORT.md` | This report |
