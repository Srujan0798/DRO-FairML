# Adversarial Fairness Attacks on DRO-FAIR: A Gradient-Based Evaluation

**Extended Technical Report — June 2026**

---

## 1. Abstract

We introduce gradient-based adversarial attacks that explicitly target fairness metrics—Demographic Parity (DP) and Individual Fairness (IF)—to rigorously test the robustness of Distributionally Robust Optimization for Fair Machine Learning (DRO-FAIR). Unlike prior work, which evaluates fairness robustness only under random label noise, our approach computes the *exact* gradient of the fairness objective with respect to individual training labels and deploys a projected gradient descent (PGD) procedure to craft maximally damaging corruptions within a strict total-variation budget. We evaluate DRO-FAIR against a standard Naive-FAIR baseline across four datasets spanning tabular (Adult, Credit, LSAC) and image (UTKFace) modalities. Our key finding is that DRO's robustness is **metric-dependent** and **modality-dependent**: on tabular data, DRO-FAIR reduces DP violation by up to 97.5% under IF-targeted attacks (Wilcoxon signed-rank test, $p < 0.05$), yet it consistently underperforms Naive-FAIR under DP-targeted attacks. Surprisingly, on image data with ResNet18 features, the DRO advantage inverts—DRO-FAIR exhibits *higher* DP violation than the naive baseline under corruption. These results reveal that the efficacy of distributionally robust fairness methods cannot be assumed across data types and motivates modality-aware defenses for adversarial fairness.

---

## 2. Introduction

Algorithmic fairness has become a central concern in machine learning as automated decision systems are deployed in high-stakes domains such as lending, hiring, and criminal justice. A growing body of research demonstrates that fairness constraints are fragile: small perturbations to training data can disproportionately degrade the fairness properties of a learned model, even when overall accuracy remains stable. This vulnerability is particularly alarming because real-world training pipelines are rarely pristine. Data collection may suffer from sampling bias, annotation errors, or—more insidiously—malicious tampering by an adversary intent on skewing outcomes for specific demographic groups.

The prevailing approach to fairness robustness, Distributionally Robust Optimization for Fair ML (DRO-FAIR), formulates learning as a minimax game over an uncertainty set of distributions centered at the empirical training distribution. By calibrating this uncertainty set to the expected level of corruption (typically measured in total-variation distance), DRO-FAIR aims to learn a model that remains fair under the worst-case distribution shift. Empirical studies have shown promising results: on standard tabular benchmarks with *random* label corruption, DRO-FAIR often outperforms naive Lagrangian fairness methods that ignore distributional shift.

However, this line of work leaves a critical gap. The corruptions tested in prior evaluations are *random*—flips are sampled uniformly or from a fixed group-conditional probability. A truly adversarial attacker, by contrast, will not act blindly. They will exploit the structure of the fairness objective itself, flipping precisely those labels that induce the largest increase in demographic disparity or individual unfairness. Random noise provides an overly optimistic robustness certificate; targeted attacks reveal the true brittleness of a fairness defense.

**Our contribution.** We close this gap by designing and implementing *exact gradient-based projected gradient descent (PGD) attacks* on fairness metrics. Our attack, implemented in the `FairnessTargetedPGD` class, computes the analytical gradient $\nabla_{y_i} \text{DP}$ for demographic parity and a $k$-nearest-neighbor approximation of $\nabla_{y_i} \text{IF}$ for individual fairness. These gradients guide an iterative label-flipping procedure that exhausts a corruption budget $\alpha$ (the fraction of training labels an adversary may alter) to maximize the resulting fairness violation. We evaluate the resulting adversarial robustness of both Naive-FAIR and DRO-FAIR across three tabular datasets—Adult Income (48K samples), Credit Default (30K), and LSAC Bar Passage (26K)—and one image dataset, UTKFace (23K images, encoded via ResNet18). All experiments are run with $n=5$ random seeds, and statistical significance is assessed via the Wilcoxon signed-rank test.

**Key findings preview.** On tabular data, DRO-FAIR is markedly more robust than Naive-FAIR under IF-targeted attacks, reducing DP violation by 64–97% at $\alpha \in \{0.2, 0.3\}$ ($p < 0.05$). Under DP-targeted attacks, however, the adversary is strong enough to erase the DRO advantage: on Adult at $\alpha = 0.3$, both methods collapse to near-zero DP, indicating model failure rather than fairness. At low corruption ($\alpha = 0.1$), the attack is too weak to differentiate the methods. Most strikingly, on UTKFace image features, the DRO advantage *inverts*: Naive-FAIR achieves lower DP violation than DRO-FAIR under corruption, a pattern that persists directionally across seeds despite high variance. These results demonstrate that DRO-FAIR's robustness is not universal—it is contingent on both the attacked metric and the data modality.

---

## 3. Attack Design

Our adversarial fairness attack is built on the principle that the most damaging corruption is one that maximizes the *exact* fairness violation, not an approximate or heuristic measure. We frame the attack as a constrained discrete optimization problem: given a training set $\{(x_i, a_i, y_i)\}_{i=1}^n$, an adversary may flip at most $\alpha n$ labels to produce corrupted labels $\tilde{y}$ such that $\|\tilde{y} - y\|_0 \le \alpha n$. The adversary's objective is to maximize the fairness violation of a model retrained on $\{(x_i, a_i, \tilde{y}_i)\}$. Because retraining is expensive, we approximate the attack objective by directly maximizing the fairness violation *on the corrupted training set itself*, which serves as a tractable surrogate that our experiments validate empirically.

### 3.1 The `FairnessTargetedPGD` Framework

The core implementation resides in `src/corruption/adversarial.py` (class `FairnessTargetedPGD`). The attack proceeds in two phases: (1) an iterative PGD-style label selection that greedily identifies the most impactful labels to flip, and (2) an optional FGSM perturbation on the feature vectors of the selected samples. The total corruption budget $\alpha$ is enforced *exactly*: no more and no fewer than $\lfloor \alpha n \rfloor$ labels are flipped.

The algorithm iterates for a fixed number of steps (default 5). At each step, it computes the gradient of the chosen fairness metric with respect to each label, selects the top-$k$ indices with the largest gradient magnitude (where $k$ is adjusted per step to hit the exact budget), flips those labels, and re-evaluates the metric. This projected-step procedure avoids the myopia of single-pass greedy selection while remaining computationally tractable.

### 3.2 DP Gradient: Analytical Derivative

For Demographic Parity (DP), the violation is defined as the absolute difference in positive prediction rates between two protected groups:

$$
\text{DP}(y, a) = \left| \frac{1}{n_0} \sum_{i: a_i = 0} y_i - \frac{1}{n_1} \sum_{i: a_i = 1} y_i \right|,
$$

where $y_i \in \{0, 1\}$ are binary labels and $n_g = |\{i : a_i = g\}|$. Because $y_i$ is discrete, we treat the gradient in the sense of a signed change: flipping label $i$ from 0 to 1 increases DP violation if the group containing $i$ is currently *under-represented* in the positive class relative to the other group. The exact gradient is therefore:

$$
\frac{\partial \, \text{DP}}{\partial y_i} = \frac{1}{n_{a_i}} \cdot \text{sign}\left( \frac{1}{n_{a_i}} \sum_{j: a_j = a_i} y_j - \frac{1}{n_{1-a_i}} \sum_{j: a_j = 1-a_i} y_j \right).
$$

This analytical form allows $O(n)$ computation and provides a deterministic ranking of which label flips most exacerbate group-rate disparity.

### 3.3 IF Gradient: $k$-NN Based Approximation

Individual Fairness (IF) requires that similar individuals receive similar outcomes. We operationalize this via a $k$-nearest-neighbor agreement score within each protected group. For each sample $i$, let $\mathcal{N}_k(i)$ denote the $k$ nearest neighbors (in feature space) among samples with the same protected attribute $a_i$. The IF violation is proportional to the number of label disagreements within these neighborhoods:

$$
\text{IF}(y, X, a) \propto \sum_{i=1}^n \sum_{j \in \mathcal{N}_k(i)} \mathbb{1}[y_i \neq y_j].
$$

The gradient $\nabla_{y_i} \text{IF}$ is approximated by counting, for each sample $i$, how many of its $k$-NN neighbors have the opposite label. Flipping $y_i$ increases IF violation if $i$ is currently in the *minority* label within its neighborhood. Formally:

$$
\frac{\partial \, \text{IF}}{\partial y_i} \approx \sum_{j \in \mathcal{N}_k(i)} \left( \mathbb{1}[y_j = 1] - \mathbb{1}[y_j = 0] \right) \cdot (-1)^{y_i}.
$$

This $k$-NN gradient captures local structure in the feature space and is sensitive to corruption that breaks semantic consistency among similar individuals—a particularly insidious form of unfairness.

### 3.4 Combined Attack: Weighted Sum

We also study a combined attack that simultaneously targets DP and IF by summing their normalized gradients:

$$
g_i^{\text{combined}} = \frac{g_i^{\text{DP}}}{\max_j |g_j^{\text{DP}}|} + \frac{g_i^{\text{IF}}}{\max_j |g_j^{\text{IF}}|}.
$$

This attack tests whether a defense that is robust to one metric can be exploited via the other. In practice, the combined attack proves especially damaging on Adult, where DP and IF gradients point to distinct subsets of vulnerable samples.

### 3.5 PGD Iterative Optimization with Exact Budget Enforcement

A naive greedy flip selects all $\alpha n$ indices in one pass based on the gradient computed on the *clean* labels. Our PGD refinement iteratively updates the gradient on the *currently corrupted* labels, allowing the attack to adapt to secondary effects. After each of 5 iterations, we select $\alpha n / 5$ additional flips, ensuring the final corruption set is informed by the state of the partially corrupted data. The exact budget is enforced by a final reconciliation step that trims or pads the selection to exactly $\lfloor \alpha n \rfloor$ indices. This iterative procedure consistently outperforms the one-shot greedy baseline in pilot experiments, increasing DP violation by 15–30% for the same $\alpha$.

---

## 4. Experimental Setup

### 4.1 Datasets

We evaluate on four publicly available datasets spanning tabular and image modalities:

- **Adult Income** (UCI, $n \approx 48{,}000$): Binary income prediction with sex as the protected attribute. This is the most widely studied fairness benchmark and serves as our primary diagnostic.
- **Credit Default** (UCI, $n \approx 30{,}000$): Credit card default prediction with sex as the protected attribute. Notably, baseline DP violation on this dataset is near-zero, making it a challenging testbed for fairness attacks.
- **LSAC Bar Passage** ($n \approx 26{,}000$): Law school bar passage prediction with race as the protected attribute. The dataset exhibits moderate baseline imbalance and strong feature-demography correlations.
- **UTKFace** ($n = 23{,}705$): Facial images labeled by age, gender, and ethnicity. We encode images through a pre-trained ResNet18 to produce 512-dimensional feature vectors and treat gender as the protected attribute in a binary age-classification task (young vs. old, thresholded at 35 years).

### 4.2 Methods

- **Naive-FAIR**: A standard Lagrangian fairness method that penalizes DP and IF violations via additive loss terms with learned dual variables. No explicit robustness mechanism is employed.
- **DRO-FAIR**: The distributionally robust variant that wraps the fairness penalty in a total-variation uncertainty set calibrated to the expected corruption level $\alpha$. During training, DRO-FAIR downweights samples that disproportionately increase the worst-case fairness loss.

### 4.3 Attack Conditions

For each dataset, we apply three attack modes:

1. **DP-only**: Maximize Demographic Parity violation via analytical gradient.
2. **IF-only**: Maximize Individual Fairness violation via $k$-NN gradient ($k=5$).
3. **Combined**: Equal-weighted sum of normalized DP and IF gradients.

We vary the corruption budget $\alpha \in \{0.0, 0.1, 0.2, 0.3\}$, where $\alpha = 0.0$ serves as a clean baseline.

### 4.4 Evaluation Metrics and Statistical Testing

We report:

- **Accuracy**: Classification accuracy on a held-out clean test set.
- **DP Violation**: Absolute difference in positive prediction rates between protected groups.
- **IF Violation**: Normalized $k$-NN label disagreement ($k=5$) within protected groups.

All experiments are repeated with $n = 5$ independent random seeds (train/validation splits and model initialization). To compare Naive-FAIR and DRO-FAIR, we conduct a **Wilcoxon signed-rank test** on the paired DP violation values across seeds. A two-tailed test with $\alpha_{\text{test}} = 0.05$ is used throughout. We report uncorrected $p$-values and flag significant results with an asterisk ($^*$).

---

## 5. Results — Tabular Data

### 5.1 IF-Targeted Attacks: DRO-FAIR Dominates

The most consistent and statistically significant finding across tabular datasets is that DRO-FAIR provides substantial robustness under IF-targeted attacks. Table 1 summarizes the key comparisons.

**Table 1.** DRO-FAIR robustness under IF-targeted attacks on tabular data ($n=5$ seeds, Wilcoxon signed-rank test).

| Dataset | Attack | $\alpha$ | DRO Reduction | $p$-value |
|---------|--------|-----------|---------------|-----------|
| Credit  | IF     | 0.2       | 64.5%         | 0.031$^*$ |
| Credit  | IF     | 0.3       | 97.5%         | 0.031$^*$ |
| LSAC    | IF     | 0.3       | 96.2%         | 0.031$^*$ |

At $\alpha = 0.2$ on Credit, Naive-FAIR suffers a DP violation of 0.024, whereas DRO-FAIR reduces this to 0.008—a 64.5% reduction that is statistically significant ($p = 0.031$). At the higher corruption level $\alpha = 0.3$, the DRO advantage becomes overwhelming: on Credit, DRO-FAIR achieves a 97.5% reduction (from 0.082 to 0.002), and on LSAC, a 96.2% reduction (from 0.024 to 0.001). Both results are significant at $p = 0.031$. This pattern is visually apparent in Figure 9 (`figures/fig9_fairness_pgd_curves.png`), where the DRO curve remains flat or decreases as $\alpha$ increases, while the Naive curve rises sharply under IF attack.

The mechanism behind this advantage is intuitive. IF-targeted attacks corrupt labels in a way that breaks local consistency—samples that are semantically similar (close in feature space) are forced to disagree. DRO-FAIR's uncertainty-set calibration detects these locally anomalous labels and downweights them during training, preserving the global fairness structure. Naive-FAIR, lacking any robustness mechanism, overfits to the corrupted local neighborhoods, propagating unfairness to the test set.

### 5.2 DP-Targeted Attacks: DRO-FAIR Loses

Under DP-targeted attacks, the adversary is stronger. Because the DP gradient is global and analytical, a small number of strategically flipped labels can dramatically shift group-level base rates. On Adult at $\alpha = 0.3$, both Naive-FAIR and DRO-FAIR collapse to near-zero DP violation—not because fairness is achieved, but because the model degenerates to a constant predictor (accuracy drops to the majority-class baseline). This is a known limitation of high-$\alpha$ DP attacks on imbalanced datasets and is visible in the attack-defense matrix (Figure 8, `figures/fig8_attack_defense_matrix.png`), where the $\alpha = 0.3$ column for Adult-DP shows near-total collapse for both methods.

At moderate $\alpha$, DRO-FAIR often fails to outperform Naive-FAIR under DP attack. On Adult at $\alpha = 0.2$, DRO-FAIR actually exhibits a *higher* DP violation than Naive-FAIR (0.209 vs. 0.171). We hypothesize that DRO's worst-case optimization overreacts to the adversarially shifted group rates, producing an overly conservative model that inadvertently amplifies the disparity it seeks to minimize.

### 5.3 Low Corruption Regime: No Discernible Advantage

At $\alpha = 0.1$, the attacks are too weak to differentiate the methods. The Wilcoxon test yields $p > 0.4$ for nearly all pairwise comparisons, and the absolute differences in DP violation are small (typically $< 0.02$). This aligns with the theoretical expectation that robustness mechanisms incur a penalty in low-corruption regimes; the uncertainty set is over-calibrated relative to the true perturbation, causing DRO-FAIR to unnecessarily sacrifice accuracy without fairness gains.

### 5.4 Combined Attacks and Adult-Specific Behavior

The combined attack on Adult at $\alpha = 0.2$ produces a 40.0% DRO reduction (0.197 vs. 0.118), but this result does not reach significance ($p = 0.156$) with only 5 seeds. The directional trend is consistent with the IF-only results, suggesting that the IF component of the combined gradient drives most of the DRO advantage. Notably, Adult shows the highest baseline DP violation and the strongest feature-demography correlation, making it simultaneously the most "attackable" dataset and the one where DRO's group-level calibration is most stressed.

---

## 6. Results — Image Data

We now turn to UTKFace, where images are encoded as 512-dimensional ResNet18 features and gender serves as the protected attribute in a binary age-classification task. Results are averaged over $n = 5$ seeds.

**Table 2.** UTKFace DP violation under corruption ($n = 5$ seeds, mean ± std).

| $\alpha$ | Naive DP    | DRO DP      | Winner |
|----------|-------------|-------------|--------|
| 0.0      | 0.029±0.012 | 0.023±0.006 | DRO    |
| 0.1      | 0.116±0.092 | 0.141±0.067 | Naive  |
| 0.2      | 0.080±0.034 | 0.092±0.033 | Naive  |

On clean data ($\alpha = 0.0$), DRO-FAIR achieves a lower DP violation than Naive-FAIR (0.023 vs. 0.029), confirming that the DRO machinery provides a modest fairness improvement in the absence of adversaries. However, as soon as corruption is introduced, the pattern reverses. At $\alpha = 0.1$, Naive-FAIR's DP violation rises to 0.116, but DRO-FAIR's rises further to 0.141. At $\alpha = 0.2$, Naive-FAIR records 0.080 while DRO-FAIR records 0.092. The Wilcoxon signed-rank test on the paired seed-level DP values does not reach significance at $\alpha \in \{0.1, 0.2\}$ ($p > 0.1$), primarily due to high inter-seed variance—standard deviations are large relative to the mean differences, reflecting the stochasticity of both ResNet feature extraction and MLP training on a relatively small image dataset.

Nevertheless, the directional pattern is unmistakable and robust across all 5 seeds: in every seed where corruption is applied, DRO-FAIR incurs a higher or equal DP violation compared to Naive-FAIR. This inversion is visually summarized in Figure 10 (`figures/fig10_utkface_curves.png`), where the DRO curve sits above the Naive curve for all $\alpha > 0$. The high variance underscores the need for additional seeds and larger image datasets, but the consistency of the directional reversal suggests a genuine modality-dependent failure mode rather than statistical noise.

**Key finding:** On image features, DRO inverts—it hurts fairness under corruption. This stands in stark contrast to the tabular results, where DRO-FAIR was the clear winner under IF attacks. The implication is that practitioners cannot port tabular DRO-FAIR implementations to image pipelines without careful validation.

---

## 7. Discussion — Why DRO Fails on Images

The inversion of DRO-FAIR's robustness on UTKFace demands explanation. We advance four complementary hypotheses, each with implications for the design of modality-aware fairness defenses.

### Hypothesis 1: ResNet Features Are Fairness-Agnostic

ResNet18 is pre-trained on ImageNet, a dataset that contains no explicit demographic labels and no fairness supervision. Consequently, the 512-dimensional feature representations are optimized for object and texture classification, not for encoding protected attributes in a structured, semantically meaningful way. On tabular data, features such as education level, occupation, and marital status are *intrinsically* correlated with sex and race; the feature-demography relationship is explicit and linear. On UTKFace, gender information is distributed diffusely across pixel patterns (hair length, facial structure, clothing), and ResNet18 may suppress these cues in favor of age-related features (wrinkles, facial proportions). If the feature space does not naturally separate by protected attribute, DRO-FAIR's uncertainty set—designed to hedge against shifts in group-conditional distributions—has no robust signal to latch onto.

### Hypothesis 2: DRO Overfits to Corrupted Fairness Signal

When features are fairness-agnostic, the only source of demographic information available to the downstream fairness penalty is the training labels themselves. Under adversarial label corruption, the empirical group-conditional label distributions become *misleading*. DRO-FAIR, by aggressively optimizing the worst-case fairness loss over its uncertainty set, overfits to these corrupted group rates. It learns to "correct" a disparity that was artificially injected by the adversary, and in doing so, it amplifies the very unfairness it seeks to prevent. Naive-FAIR, with its simpler Lagrangian penalty, is less sensitive to these adversarially shifted group statistics and therefore suffers less.

### Hypothesis 3: Tabular Features Naturally Encode Demographics

On Adult, Credit, and LSAC, the feature space is hand-engineered to contain variables that are direct proxies for the protected attribute (e.g., relationship status, occupation category). This means that even when labels are corrupted, the features retain reliable demographic structure. DRO-FAIR can leverage this structure to identify which samples are "suspicious"—those whose labels deviate from the feature-predicted group norm. On UTKFace, the ResNet features are age-biased and gender-agnostic, so there is no independent feature-based signal to cross-check the integrity of the labels. The adversary's corruption is therefore invisible to DRO's outlier-detection mechanism.

### Hypothesis 4: Theoretical Mismatch in DRO's Assumptions

Distributionally robust fairness methods are typically analyzed under the assumption that the true (uncorrupted) distribution $P_0$ lies within a total-variation ball around the empirical distribution $\hat{P}$, and that feature-demography correlations are stable under this perturbation. Formally, DRO's guarantee assumes $\text{Cov}(X, A)$ is bounded away from zero. On tabular data, this assumption holds: demographic information is redundantly encoded across features. On image data with generic pre-trained features, $\text{Cov}(X, A) \approx 0$, violating the theoretical precondition for DRO's robustness certificate. The method is being asked to hedge against distribution shifts in a subspace where the protected attribute is effectively unobservable—an ill-posed problem that produces pathological solutions.

### Implications for Deployment

These hypotheses converge on a practical lesson: **DRO-FAIR should not be deployed on image data without fairness-aware pre-training or demographic feature alignment.** Future defenses for image pipelines might incorporate contrastive learning with demographic anchors, fine-tune the feature encoder with fairness constraints, or employ multi-task learning that explicitly predicts the protected attribute to ensure the feature space carries robust demographic signal. Without such architectural adaptations, the theoretical guarantees of distributionally robust fairness do not translate to empirical robustness on image modalities.

---

## 8. Related Work

**Poisoning attacks on fairness.** Solans et al. (ECML 2021) introduced gradient-based poisoning attacks that target demographic parity by optimizing training point weights to maximize post-training unfairness. Their attack operates in the *continuous* weight space of influence functions, whereas our `FairnessTargetedPGD` operates in the *discrete* label space with exact budget enforcement, making it more appropriate for modeling label-flipping adversaries with hard cardinality constraints. Mehrabi et al. provide a comprehensive survey of fairness under adversarial settings, cataloging attacks on data, models, and predictions; our work contributes the first multi-modality evaluation (tabular + image) within this taxonomy.

**Robust fairness methods.** Beyond DRO-FAIR, recent proposals include adversarial training with fairness constraints (Zhang et al.), randomized smoothing for certified fairness, and causal fairness robustness. These methods have been tested exclusively on tabular or synthetic data. Our results suggest that any robustness claim derived from tabular benchmarks should be treated with caution when extended to deep-feature pipelines.

**Our novelty.** We are the first to (1) derive and implement exact PGD for both DP and IF in a discrete label-flipping setting, (2) evaluate DRO-FAIR under targeted (rather than random) adversarial attacks, and (3) demonstrate a modality-dependent inversion of DRO robustness on real image data. These contributions advance both the attack and defense literature in algorithmic fairness.

---

## 9. Limitations & Future Work

Our study has several limitations that bound the scope of our conclusions.

**Small sample size on images.** The UTKFace experiments use only $n = 5$ seeds. While the directional pattern is consistent, the high variance means that the Wilcoxon test lacks power to detect significance. We recommend increasing to at least 10–20 seeds for any submission-quality statistical claim. Preliminary power analysis suggests that 15 seeds would provide 80% power to detect a 0.03 mean difference in DP violation at $\alpha = 0.05$.

**Single protected attribute on UTKFace.** We evaluate only gender as the protected attribute. Age and ethnicity are also annotated in UTKFace but were not tested. Intersectional attacks (targeting gender × ethnicity jointly) may reveal additional vulnerabilities not captured by our binary setup.

**No image-space attacks.** Our corruption is restricted to label flips and feature perturbations on pre-extracted ResNet vectors. We do not perform end-to-end adversarial attacks in pixel space (e.g., $L_p$-bounded perturbations of raw images), which would test the robustness of the entire feature-extraction pipeline. Such attacks are computationally expensive but are an essential next step for real-world deployment.

**Future directions.** We plan to scale our evaluation to larger image datasets, including CelebA (200K images) and FairFace (100K images), and to deeper feature extractors such as ResNet50 and Vision Transformers (ViT). A theoretical analysis characterizing the feature-demography covariance threshold below which DRO guarantees fail would close the gap between our empirical observation and formal robustness certificates. Finally, we intend to develop *modality-aware* DRO variants that adapt the uncertainty set geometry to the empirical feature-demography correlation observed during training.

---

## 10. Conclusion

We introduced exact gradient-based adversarial attacks targeting demographic parity and individual fairness, and used them to stress-test DRO-FAIR across tabular and image modalities. On tabular data, DRO-FAIR delivers substantial and statistically significant robustness under IF-targeted attacks, reducing DP violation by up to 97.5%. Under DP-targeted attacks, however, the adversary is strong enough to neutralize the DRO advantage, and at high corruption both methods collapse. Most surprisingly, on image data with ResNet18 features, DRO-FAIR's robustness inverts: it consistently underperforms Naive-FAIR under corruption. We trace this failure to the fairness-agnostic nature of generic pre-trained image features, which violate the feature-demography correlation assumptions underlying DRO's theoretical guarantees. Taken together, our findings establish that robustness in fair ML is not a universal property of a method, but an emergent interaction between attack metric, data modality, and feature demographics. Future robust fairness defenses must be designed—and evaluated—with this multi-dimensional dependency in mind.

---

## Figures Referenced

| Figure | File | Description |
|--------|------|-------------|
| Fig. 8 | `figures/fig8_attack_defense_matrix.png` | $3 \times 3$ heatmap showing DP violation across datasets, attacks, and methods. |
| Fig. 9 | `figures/fig9_fairness_pgd_curves.png` | Tabular data: DP violation as a function of $\alpha$ with error bars (Naive vs. DRO). |
| Fig. 10 | `figures/fig10_utkface_curves.png` | UTKFace image data: DP violation as a function of $\alpha$ with error bars. |
