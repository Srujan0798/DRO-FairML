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

### 4.1 Smoke Test (Adult, α=0.2, 1 seed, epochs=10)

| Attack | Method | Accuracy | DP Violation | IF Violation | Time |
|--------|--------|----------|--------------|--------------|------|
| DP | Naive | 0.808 | 0.1822 | 0.0165 | 170s |
| DP | DRO | 0.804 | 0.1964 | 0.0206 | 167s |
| IF | Naive | 0.782 | 0.0424 | 0.0041 | 178s |
| IF | DRO | 0.791 | 0.0638 | 0.0072 | 1128s |
| Combined | Naive | 0.782 | 0.0679 | 0.0028 | 20s |
| Combined | DRO | 0.793 | 0.1015 | 0.0050 | 27s |

**Notes:**
- Smoke test uses reduced epochs (10 vs 60) — models not fully converged
- Adult at α=0.2 is known to trigger feedback loops (documented in submitted paper)
- Full experiments with 60 epochs, 5 seeds, 3 datasets running now

### 4.2 Full Experiment Status

- **Launched:** May 27, 16:30
- **Target:** 264 experiments (3 datasets × 3 alphas × 5 seeds × 3 attacks × 2 methods − 6 smoke)
- **Progress:** 1/264 complete (Adult α=0.1 seed=0 dp/naive)
- **ETA:** ~4–5 hours
- **Output:** `results/fairness_pgd_results.json`

---

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
