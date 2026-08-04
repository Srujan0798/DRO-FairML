# DRO-FairML — Progress Update  
**Date:** 4 August 2026  
**Project:** Distributionally Robust Optimization for Fairness under Adversarial Corruption  
**Meeting:** weekly check-in (Manisha / Kuldeep)

---

## 1. Headline result

**Fixing prediction temperature to τ = 1** makes **DRO-FAIR** more robust than **Naive-FAIR** on **Adult** and **Credit** at moderate corruption (**α ≤ 0.2**), under Fairness-Targeted PGD attacks (DP and Combined).

The earlier impression that “DRO is fragile” came from a **stepped temperature schedule (τ = 100 for α ≤ 0.3)**. That was an experimental artifact, not a failure of the method. With **fixed τ = 1** the conclusion reverses on the main tabular setting.

**Protocol (locked):** τ = 1, K_inner = 10, 6 seeds, 60 epochs, 20 PGD steps.  
**Full grid complete:** 540 runs = 3 datasets × 5 α × 3 attacks × 6 seeds × 2 methods.

---

## 2. Main results (DP violation ↓ better; n = 6 seeds)

Wilcoxon one-sided: H₁ = Naive DP > DRO DP. Wins = seeds where DRO is better.

### Adult — DP attack

| α | Naive DP | DRO DP | Wins | p |
|---|----------|--------|------|---|
| 0.0 | 0.1491 | 0.1426 | 6/6 | 0.016 |
| 0.1 | 0.2026 | 0.1999 | **5/6** | 0.031 |
| 0.2 | 0.2452 | 0.2334 | 6/6 | 0.016 |
| 0.3 | 0.2848 | 0.2614 | 6/6 | 0.016 |
| 0.4 | 0.3140 | 0.2855 | 6/6 | 0.016 |

Note: at **α = 0.1** the win count is **5/6** (not 6/6); still significant (p = 0.031).

### Adult — Combined attack

| α | Naive DP | DRO DP | Wins | p |
|---|----------|--------|------|---|
| 0.0 | 0.1491 | 0.1426 | 6/6 | 0.016 |
| 0.1 | 0.1509 | 0.1432 | 6/6 | 0.016 |
| 0.2 | 0.1963 | 0.1784 | 6/6 | 0.016 |
| 0.3 | 0.2176 | 0.1922 | 6/6 | 0.016 |
| 0.4 | 0.2110 | 0.1815 | 6/6 | 0.016 |

### Credit — DP attack

All α: **6/6** wins, p = 0.016 (DRO lower DP).

### Credit — Combined attack

| α | Wins | p |
|---|------|---|
| 0.0 | 6/6 | 0.016 |
| 0.1 | **5/6** | 0.031 |
| 0.2–0.4 | 6/6 | 0.016 |

### LSAC — Combined attack (positive)

| α | Wins | p |
|---|------|---|
| 0.0 | 0/6 | — |
| 0.1 | 6/6 | 0.016 |
| 0.2 | 5/6 | 0.031 |
| 0.3–0.4 | 6/6 | 0.016 |

### LSAC — DP attack (negative / degenerate)

DRO loses **0/6** at every α. Accuracy is pinned near the majority-class baseline (~0.90); Naive DP freezes for α ≥ 0.2. We report this as a **degenerate diagnostic**, not a clean method comparison.

---

## 3. First real Individual Fairness (IF) attack results

IF metric was previously broken (near-zero). It is now cosine-based and non-degenerate.

**Summary: mixed — not a clean “third attack” sweep.**

| Dataset | Main message |
|---------|----------------|
| **Adult** | At **α = 0.1 and 0.2**, DRO is better on **both** IF and DP under IF attack (6/6). At α = 0.0, DP wins but IF is not significant (4/6). At **α = 0.3**, IF improves but **DP loses** (1/6). |
| **Credit** | IF improves for α ≥ 0.1 (6/6); DP under IF is mostly favorable (α = 0.1 is weaker, 4/6, not significant). |
| **LSAC** | Does not support the low-α story; DP under IF attack loses for α ≤ 0.3. |

---

## 4. Defensible claim scope (accuracy)

Constant-predictor (majority class) accuracy:

| Dataset | Baseline accuracy |
|---------|-------------------|
| Adult | 0.752 |
| Credit | 0.779 |
| LSAC | 0.902 |

At **α ≥ 0.3**, both DRO and Naive fall **below** the constant predictor on **Adult and Credit**. We therefore **do not claim robustness** in that regime.  
**LSAC** stays **pinned at** ~0.90 (degeneracy), not “below baseline.”

**Defensible regime for method claims: α ≤ 0.2 on Adult and Credit.**

---

## 5. Attack & method (reminder)

- **Attack:** Fairness-Targeted PGD — DP only, IF only, or Combined.  
- **Methods:** Naive-FAIR vs DRO-FAIR (TV uncertainty set).  
- **Datasets (tabular):** Adult, Credit, LSAC.  
- **UTKFace:** real image features extracted; multi-seed grid still running. **No image claim in this update.**

---

## 6. Figures (attach with this note)

1. **τ = 1 headline** — Adult DP vs α (Naive vs DRO), seed win counts marked  
2. **Win / significance summary** — DP-attack wins across Adult, Credit, LSAC and α  

---

## 7. Next steps

1. Complete the UTKFace multi-seed multi-attack runs on **real** image features (or list as future work if incomplete).  
2. Final paper/report write-up with the **mixed IF** story and honest LSAC/DP section.

---

## 8. Bottom line (30 seconds)

1. **τ = 1** fixed the false “DRO fragile” story.  
2. **Adult & Credit, α ≤ 0.2, DP + Combined:** DRO better on DP (n = 6), with honest **5/6** cells noted.  
3. **IF attack:** real for the first time; **mixed** across datasets/α.  
4. **LSAC/DP:** degenerate collapse — reported honestly.  
5. **α ≥ 0.3:** out of claim scope on Adult/Credit (below constant predictor).

---

*All numbers recomputed from the locked canonical experiment grid (540 runs). No hidden negatives.*
