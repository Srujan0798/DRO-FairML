# Adversarial Fairness Attacks on Distributionally Robust Fair ML

**Srujan Sai**  
*Indian Institute of Technology Gandhinagar*  
*May 31, 2026*

---

## Abstract

We introduce gradient-based adversarial attacks that explicitly target fairness metrics — Demographic Parity (DP) and Individual Fairness (IF) — and evaluate the robustness of DRO-FAIR, a distributionally robust fair classification method, under these attacks. Unlike prior work that tests only random label noise, our attacks compute the exact gradient of the fairness metric with respect to each training label and flip the optimal subset to maximize unfairness. We conduct 270 experiments on three tabular datasets (Adult, Credit, LSAC) and 15 experiments on UTKFace image data with ResNet18 features. Our key finding is that DRO-FAIR's robustness is **not universal**: it strongly defends against IF-targeted attacks on tabular data (64–97% DP reduction, p < 0.05), but **inverts** on image features — producing worse fairness than naive training under corruption. This modality-dependent behavior reveals a fundamental limitation: DRO requires feature-demography correlation to be effective. Without it, worst-case reweighting amplifies noise rather than signal.

---

## 1. Introduction

### 1.1 Background

Algorithmic fairness has become a critical concern as machine learning systems are deployed in high-stakes domains such as lending, hiring, and criminal justice. A common approach is to enforce fairness constraints during training, such as Demographic Parity (DP) — requiring equal positive prediction rates across protected groups — and Individual Fairness (IF) — requiring similar individuals to receive similar predictions.

Distributionally Robust Optimization (DRO) has emerged as a principled framework for training models that are robust to distributional shifts. In the context of fair ML, DRO-FAIR formulates fairness as a constraint within a DRO framework, learning model parameters that satisfy fairness constraints under the worst-case distribution within an uncertainty set.

### 1.2 The Gap: Random vs. Adversarial Corruption

Existing evaluations of fair ML methods under data corruption typically use **random label noise** — uniformly flipping a fraction of training labels. While this provides a baseline stress test, it does not reflect realistic adversarial scenarios where an attacker with knowledge of the fairness metric can strategically corrupt the most impactful samples.

Recent work by Solans, Biggio, and Castillo (ECML 2021) introduced poisoning attacks on algorithmic fairness, but these attacks use heuristic strategies rather than gradient-based optimization. To our knowledge, no prior work has computed the **exact gradient** of fairness metrics with respect to individual labels and used it to construct targeted adversarial examples.

### 1.3 Our Contributions

1. **FairnessTargetedPGD**: A novel gradient-based attack that computes ∂(fairness)/∂y_i for each sample and flips the top-α samples to maximize DP or IF violation.
2. **Comprehensive Evaluation**: 270 experiments on tabular data and 15 experiments on image data (UTKFace with ResNet18 features).
3. **Key Finding**: DRO-FAIR's robustness is **metric-dependent** (wins under IF attacks, loses under DP attacks) and **modality-dependent** (wins on tabular data, inverts on image features).

---

## 2. Attack Design

### 2.1 Threat Model

We consider an attacker who can modify a fraction α of training labels, features, and protected attributes. The attacker has white-box access to the fairness metric and seeks to maximize the unfairness of the trained model.

### 2.2 FairnessTargetedPGD

Our attack, implemented in `src/corruption/adversarial.py`, operates in three modes:

**DP-only Attack.** Computes the analytical gradient of the Demographic Parity gap with respect to each label:

```
∂(DP_gap)/∂y_i = sign(P(y=1|a=0) - P(y=1|a=1)) × (1{a_i=0}/n_0 - 1{a_i=1}/n_1)
```

This gradient indicates whether flipping y_i increases or decreases the DP gap. We select the top-α samples with the largest positive gradient and flip them.

**IF-only Attack.** Computes the gradient of Individual Fairness violation using a k-NN graph within each protected group:

```
∂(IF)/∂y_i = (agreeing_neighbors - disagreeing_neighbors) / k_eff
```

Samples with many disagreeing neighbors are flipped to increase IF violation.

**Combined Attack.** Uses a weighted sum of DP and IF gradients.

### 2.3 PGD Optimization

The attack runs 5 iterative steps:
1. Compute fairness gradient on current labels
2. Select top-α targets
3. Flip selected labels
4. Repeat

After all steps, we enforce the exact α budget by keeping only the top-α flips ranked by final gradient magnitude.

---

## 3. Experimental Setup

### 3.1 Datasets

| Dataset | Samples | Features | Protected | Task |
|---------|---------|----------|-----------|------|
| Adult | 48,842 | 14 | Sex | Income >50K |
| Credit | 30,000 | 23 | Sex | Default prediction |
| LSAC | 26,184 | 11 | Race | Bar passage |
| UTKFace | 23,705 | 512 (ResNet18) | Race | Gender classification |

### 3.2 Methods

- **Naive-FAIR**: Standard fairness-constrained training with fixed Lagrange multiplier.
- **DRO-FAIR**: Distributionally robust fairness training with adaptive worst-case reweighting.

### 3.3 Attack Configurations

- α ∈ {0.1, 0.2, 0.3} for tabular; α ∈ {0.0, 0.1, 0.2} for UTKFace
- 5 random seeds per configuration
- 3 attack modes: DP-only, IF-only, Combined

### 3.4 Evaluation Metrics

- Accuracy
- DP Violation: |P(ŷ=1|a=0) - P(ŷ=1|a=1)|
- IF Violation: Average k-NN label disagreement within groups
- Statistical significance: Wilcoxon signed-rank test (one-sided), n=5

---

## 4. Results — Tabular Data

### 4.1 Main Results

| Dataset | Attack | α | DRO Reduction | p-value | Significant? |
|---------|--------|---|---------------|---------|--------------|
| Credit | IF | 0.2 | **64.5%** | 0.031 | ✓ |
| Credit | IF | 0.3 | **97.5%** | 0.031 | ✓ |
| LSAC | IF | 0.3 | **96.2%** | 0.031 | ✓ |
| Adult | IF | 0.3 | 19.3% | 0.062 | (trend) |
| Adult | Combined | 0.2 | 40.0% | 0.156 | n.s. |

**Key Finding:** Under IF-targeted attacks, DRO-FAIR significantly reduces DP violation by 64–97% on Credit and LSAC (p < 0.05). This is our strongest positive result.

### 4.2 Attack-Dependent Behavior

![Attack-Defense Matrix](fig8_attack_defense_matrix.png)

The heatmap reveals a clear pattern:
- **IF attacks** (middle row): DRO wins on Credit and LSAC (green cells)
- **DP attacks** (top row): DRO loses across all datasets (red cells)
- **Combined attacks** (bottom row): Mixed, mostly not significant

**Why does DP attack break DRO?** The DP-targeted adversary directly optimizes the same metric that DRO is trying to protect. This creates a stronger feedback loop: the adversary flips labels to maximize group rate disparity, and DRO's worst-case reweighting struggles to counteract this explicit targeting.

### 4.3 Effect of Corruption Level

At α = 0.1, attacks are too weak for DRO's advantage to manifest — no significant differences. At α = 0.2–0.3, the adversary has sufficient budget to create measurable effects.

---

## 5. Results — Image Data (UTKFace)

### 5.1 Experimental Setup

We extract 512-dimensional features from 23,705 UTKFace images using ResNet18 (pre-trained on ImageNet). Gender is the classification target; race (binarized: White vs. non-White) is the protected attribute.

### 5.2 Results

| α | Naive DP | DRO DP | Winner | Wilcoxon p |
|---|----------|--------|--------|------------|
| 0.0 (clean) | 0.029 ± 0.012 | 0.023 ± 0.006 | **DRO** | 0.156 |
| 0.1 (corrupt) | 0.116 ± 0.092 | 0.141 ± 0.067 | **Naive** | 0.688 |
| 0.2 (corrupt) | 0.080 ± 0.034 | 0.092 ± 0.033 | **Naive** | 0.969 |

![UTKFace Curves](fig10_utkface_curves.png)

### 5.3 The UTKFace Surprise

**On clean data (α = 0.0):** DRO improves fairness by 23% — consistent with tabular results.

**Under corruption (α = 0.1, 0.2):** DRO **inverts** — producing 15–22% higher DP violation than Naive-FAIR. This is opposite to tabular results.

Statistical significance is not achieved with n = 5 seeds (high variance), but the directional pattern is consistent across all corruption levels.

---

## 6. Discussion — Why DRO Fails on Images

### 6.1 The Feature-Demography Correlation Hypothesis

Our central hypothesis is that **DRO-FAIR requires features to naturally encode demographic information** in order to be effective under corruption.

**Tabular data:** Features like income, credit score, and education level naturally correlate with protected attributes (sex, race). Even when labels are corrupted, DRO can identify robust patterns in the feature space that preserve fairness.

**Image features (ResNet18):** Trained on ImageNet, ResNet18 extracts visual features (edges, textures, shapes) that are **demographic-agnostic**. The model does not encode race or gender information in its feature representation. Under label corruption, DRO has no robust signal to latch onto — its worst-case reweighting instead amplifies the spurious fairness signal introduced by the corrupted labels.

### 6.2 Mechanism

1. Corrupted labels introduce a spurious fairness constraint
2. DRO's Lagrange multiplier λ increases to enforce this constraint
3. Without feature-demography correlation, λ over-corrects
4. Result: DRO achieves worse fairness than Naive training

### 6.3 Implications

This finding has important implications for deploying DRO-based fair ML:
- **Tabular data with natural demographic correlations:** DRO is effective
- **Transfer-learned image features:** DRO may be harmful under adversarial corruption
- **Recommendation:** For image data, consider adversarial training or feature-space fairness regularization instead of DRO

---

## 7. Related Work

**Poisoning Attacks on Fairness.** Solans, Biggio, and Castillo (ECML 2021) introduced the first poisoning attacks targeting algorithmic fairness. Their attacks use heuristic strategies (e.g., flipping labels in the minority group). Our work extends this by computing exact gradients and using PGD optimization.

**DRO for Fairness.** Hashimoto et al. (2018) and subsequent work formulated fairness within the DRO framework. Our evaluation reveals that DRO's theoretical guarantees may not translate to all data modalities under adversarial corruption.

**Adversarial Robustness in ML.** PGD attacks (Madry et al., 2018) are the standard for adversarial robustness in standard classification. We adapt this framework to fairness metrics.

---

## 8. Limitations & Future Work

### 8.1 Limitations

1. **Sample size:** UTKFace experiments use only 5 seeds. High variance means statistical significance is not achieved. We need 10+ seeds.
2. **Single protected attribute:** UTKFace uses only race as protected. Other attributes (age) may show different patterns.
3. **Feature-space attacks only:** We corrupt labels and features, but do not attack raw image pixels. Image-space PGD may reveal different vulnerabilities.
4. **Single network architecture:** Only ResNet18 features tested. Deeper networks (ResNet50, ViT) may encode more demographic information.

### 8.2 Future Work

1. **Larger-scale validation:** CelebA (200K images) and FairFace (100K images)
2. **Deeper features:** ResNet50, ViT, CLIP embeddings
3. **Image-space attacks:** PGD on raw pixels rather than pre-extracted features
4. **Theoretical analysis:** Formal characterization of when DRO helps vs. hurts
5. **Alternative defenses:** Adversarial training, input validation, certified robustness

---

## 9. Conclusion

We introduced FairnessTargetedPGD, the first gradient-based adversarial attack explicitly targeting fairness metrics. Our evaluation of DRO-FAIR on 285 experiments reveals that:

1. DRO significantly improves fairness under IF-targeted attacks on tabular data (64–97% reduction, p < 0.05)
2. DRO loses under DP-targeted attacks — the adversary is too strong
3. Most surprisingly, **DRO inverts on image features** — producing worse fairness than naive training under corruption

This modality-dependent behavior reveals a fundamental requirement for DRO: feature-demography correlation. Without it, worst-case reweighting amplifies noise. This finding cautions against deploying DRO-based fair ML on transfer-learned image features without careful validation.

---

## References

1. Solans, D., Biggio, B., & Castillo, C. (2021). Poisoning Attacks on Algorithmic Fairness. *ECML*.
2. Hashimoto, T. B., Srivastava, M., Namkoong, H., & Liang, P. (2018). Fairness Without Demographics in Repeated Loss Minimization. *ICML*.
3. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR*.
4. Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating Unwanted Biases with Adversarial Learning. *AIES*.

---

## Appendix: Reproducibility

**Code:** `github.com/Srujan0798/DRO-FairML`  
**Key files:**
- `src/corruption/adversarial.py` — FairnessTargetedPGD implementation
- `experiments/run_fairness_pgd_fast.py` — Experiment runner
- `experiments/analyze_fairness_pgd.py` — Analysis and figure generation

**Hardware:** GPU server with 2× NVIDIA L40S 48GB  
**Runtime:** ~4 hours for 270 tabular experiments; ~30 minutes for 15 UTKFace experiments
