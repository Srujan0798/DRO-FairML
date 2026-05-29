# Adversarial Fairness Attacks on DRO-FAIR
## Weekly Report #1 — May 29, 2026

---

## 1. Project Scope

**Title:** Adversarial Fairness Attacks on Distributionally Robust Fair ML

**Goal:** Test whether DRO-FAIR remains robust when adversaries explicitly target fairness metrics (DP/IF) via gradient-based attacks, on both tabular and image data.

**Key distinction from submitted paper:**
- Submitted paper (v1.0, frozen): Random corruption on 3 tabular datasets (Adult, Credit, LSAC)
- Current work: Adversarial corruption + UTKFace images (23,705 samples)
- **No submitted code was modified.**

---

## 2. Madam's Tasks & Status

| # | Task | Status | Details |
|---|------|--------|---------|
| 1 | Implement PGD for fairness metrics (DP-only, IF-only, Combined) | ✅ **COMPLETE** | `FairnessTargetedPGD` class in `src/corruption/adversarial.py` |
| 2 | Test DRO on Adult/Credit/LSAC under adversarial attacks | ✅ **COMPLETE** | 270 experiments (3 datasets × 3 alphas × 5 seeds × 3 attacks × 2 methods) |
| 3 | Set up UTKFace on GPU server | ✅ **COMPLETE** | 23,705 images, ResNet18 features, 9 experiments done |

---

## 3. Implementation

### 3.1 Fairness-Targeted PGD Attack

**File:** `src/corruption/adversarial.py` (class `FairnessTargetedPGD`, lines ~204-350)

Unlike heuristic flips, this attack computes the **exact gradient** of the fairness metric w.r.t. each sample's label:

- **DP-only attack:** `compute_dp_gradient(y, a)` — analytical gradient d(DP)/d(y_i)
- **IF-only attack:** `compute_if_gradient(y, a, X)` — k-NN graph gradient d(IF)/d(y_i)
- **Combined attack:** Weighted sum of DP + IF gradients

**Attack pipeline:**
1. PGD iterative label selection (5 steps, exact α budget enforcement)
2. FGSM feature perturbation on selected indices
3. Protected attribute flip on same indices

**Code committed:** `977422d` (main branch)

### 3.2 Experiment Infrastructure

**Files:**
- `experiments/run_fairness_pgd_fast.py` — Fast batch runner (epochs=30, K_inner=5)
- `experiments/analyze_fairness_pgd.py` — Analysis + Wilcoxon tests + figure generation
- `experiments/analyze_utkface.py` — UTKFace analysis + figure generation

### 3.3 UTKFace Pipeline

**Files:**
- `src/data/datasets.py` — `load_utkface()` with feature cache support
- `scripts/extract_utkface_features.py` — ResNet18 feature extractor
- `experiments/run_utkface.py` — Experiment runner

**Pipeline:**
```
UTKFace images (23,705) → ResNet18 → 512-dim features → MLP → DRO-FAIR
```

**GPU Server:** `flair2.iitgn.ac.in` (2× NVIDIA L40S 48GB)

---

## 4. Results

### 4.1 Task 1: Tabular Data (270 experiments)

**Key Finding:** DRO-FAIR significantly outperforms Naive-FAIR under **IF-targeted attacks** on Credit and LSAC **at high corruption levels (α=0.3)**.

| Attack | Dataset | α | DRO DP Reduction | p-value | Significant? |
|--------|---------|---|-----------------|---------|--------------|
| IF | Credit | 0.2 | 64.5% DP reduction | 0.031 | ✅ Yes |
| IF | Credit | 0.3 | 97.5% DP reduction | 0.031 | ✅ Yes |
| IF | LSAC | 0.2 | 84.8% DP reduction | 0.312 | ❌ ns |
| IF | LSAC | 0.3 | 96.2% DP reduction | 0.031 | ✅ Yes |

**At lower corruption levels (α=0.1):** DRO does not significantly outperform Naive — the attack is too weak to differentiate.

**Under DP-targeted attacks on Adult:** DRO shows HIGHER DP than Naive. The DP-targeted adversary is strong enough to break DRO's advantage — this is consistent with Week 1's feedback loop finding.

### 4.2 Task 2: UTKFace Image Data (9 experiments)

**Setup:** 23,705 images, ResNet18 features (512-dim), 3 seeds, α ∈ {0.0, 0.1, 0.2}

| α | Naive DP | DRO DP | Winner | Interpretation |
|---|----------|--------|--------|----------------|
| 0.0 (clean) | 0.0363 | **0.0272** | **DRO** | 25% better on clean data |
| 0.1 (corrupt) | **0.0937** | 0.1304 | **Naive** | DRO is 39% **worse** |
| 0.2 (corrupt) | **0.1031** | 0.1104 | **Naive** | DRO is 7% **worse** |

**🔍 NEW FINDING:** On image data with ResNet18 features, DRO-FAIR actually **increases** DP violation under label corruption, compared to Naive-FAIR. This is the **opposite** of tabular data results.

**Hypothesis:** ResNet18 features are fairness-agnostic (no demographic information encoded). When labels are adversarially corrupted, DRO overfits to the corrupted fairness signal. On tabular data, features naturally carry demographic correlations, so DRO can find robust patterns even under corruption.

---

## 5. Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig 8 | `figures/fig8_fairness_pgd_comparison.png` | Tabular: Naive vs DRO under 3 attack modes |
| Fig 9 | `figures/fig9_fairness_pgd_curves.png` | Tabular: DP vs α with error bars |
| Fig U1 | `figures/fig_utkface_dp_comparison.png` | UTKFace: Clean vs Corrupted DP comparison |
| Fig U2 | `figures/fig_utkface_tradeoff.png` | UTKFace: Accuracy vs Fairness tradeoff |

---

## 6. What to Show Madam (May 29, 3 PM)

1. **Task 1 results:** Figure 8 + Wilcoxon table → "DRO wins on Credit/LSAC under IF attacks"
2. **Task 2 results:** Figure U1 → "UTKFace shows opposite pattern — DRO hurts under corruption"
3. **Code demo:** `src/corruption/adversarial.py` → "Exact gradient computation, not heuristic"
4. **Key message:** "Adversarial fairness attacks reveal DRO's robustness is **metric-dependent** (IF attacks: DRO wins; DP attacks: DRO loses) and **modality-dependent** (tabular: DRO wins; image features: DRO loses)"

---

## 7. Next Steps (Post-Meeting)

| Task | Priority | Details |
|------|----------|---------|
| More UTKFace seeds | High | Run 5+ seeds to confirm DRO-worse finding |
| Larger image datasets | High | CelebA, FairFace (need more GPU access) |
| Deeper networks | Medium | ResNet50, ViT features |
| Theoretical analysis | Medium | Why does DRO fail on image features? |
| Draft paper | Medium | Target: NeurIPS/ICLR deadline |

---

## 8. Files for Madam to Review

| File | Purpose |
|------|---------|
| `src/corruption/adversarial.py` | FairnessTargetedPGD implementation |
| `results/fairness_pgd_wilcoxon.csv` | Statistical tests (tabular) |
| `results/utkface_results.json` | UTKFace raw results |
| `figures/fig8_fairness_pgd_comparison.png` | Tabular attack comparison |
| `figures/fig_utkface_dp_comparison.png` | UTKFace comparison |
| `docs/MEETING_CHEAT_SHEET.md` | Talking points for meeting |
