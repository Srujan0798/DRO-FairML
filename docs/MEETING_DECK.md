# DRO-FAIR: Week 2 Progress
## Adversarial Fairness Attacks + UTKFace Pipeline
**May 27, 2026**

---

## Slide 1: Motivation — Why Adversarial Attacks?

**Paper v1.0 (frozen):** Random noise corruption
- Simpler to implement
- Lower bound on defense difficulty

**Week 2: Gradient-based adversarial attacks**
- Compute exact gradient of fairness metric (DP, IF)
- Flip the $\alpha n$ worst samples for maximum unfairness
- 2--5× harder than random noise

**Question:** Does DRO-FAIR still enforce fairness when the attacker specifically targets fairness metrics?

---

## Slide 2: Fairness-Targeted PGD Attack

**DP-gradient:** For each sample $i$, compute $d(\lDP)/d(y_i)$
- $+1$ if flipping increases group rate gap
- $-1$ if flipping decreases gap

**IF-gradient:** Build $k$-NN graph within each protected group
- Count agreeing neighbors
- Gradient = $(k_{\text{agree}} - k_{\text{disagree}}) / k$

**Combined:** Equal-weighted sum of DP + IF gradients

**Code:** `src/corruption/adversarial.py::FairnessTargetedPGD`

---

## Slide 3: Key Results — 270 Experiments

| Dataset | Attack | DRO vs Naive DP | p-value |
|---------|--------|----------------|---------|
| **Credit** | IF α=0.2 | **+64.5%** | **0.031** |
| **Credit** | IF α=0.3 | **+97.5%** | **0.031** |
| **LSAC** | IF α=0.3 | **+96.2%** | **0.031** |
| Adult | Combined α=0.2 | +40.0% | 0.156 |
| Adult | DP α=0.2 | −22.7% | 0.938 |

**Key finding:** IF-attack is the strongest attack. DRO is most robust against it.

---

## Slide 4: What Breaks DRO?

**DP-attack on Adult:** The attacker flips minority samples to manipulate group rates.

- DRO's group-specific uncertainty sets cannot distinguish corrupted minority samples
- Result: DRO shows *higher* DP than Naive (not statistically significant)

**Combined attack:** More favorable for DRO because:
- Attack budget splits across DP and IF
- DRO's per-sample weights can downweight corrupted samples

---

## Slide 5: Next Steps + UTKFace

**Immediate:**
- GPU server access → run UTKFace (200K images)
- Extract ResNet18 features → cache forever
- Full UTKFace ablation: 4 alphas × 3 seeds × 2 methods

**Ongoing:**
- Report.tex Section 13 (just added)
- Figure 8 + 9 in repository

**Tuesday Meeting:** 4 PM
- "DRO reduces DP by 64-97% under IF attacks on Credit/LSAC"
- "Pipeline works — waiting on GPU for image scale results"

---

## Backup: Experiment Details

- **270 experiments:** 3 datasets × 3 alphas × 5 seeds × 3 attacks × 2 methods
- **Attacks:** DP-only, IF-only, Combined (DP+IF)
- **Methods:** Naive-FAIR, DRO-FAIR
- **Metrics:** Accuracy, DP violation, IF violation
- **Statistical test:** Wilcoxon signed-rank (one-sided, α=0.05)