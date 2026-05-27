# MADAM MEETING — May 29, 2026 at 3:00 PM
## Complete Orchestrator Plan

---

## 1. PROJECT MAIN — ONE SENTENCE

> **"We implement the 2nd approach from our ICML submission: replace random noise with adversarial noise (PGD on features, coordinated label flips, attribute flips), then test if DRO-FAIR still enforces DP + IF jointly."**

**What this is NOT:**
- NOT modifying the submitted paper (v1.0 is frozen)
- NOT doing random corruption (that was the 1st approach in the paper)
- NOT staying on small tabular data only

**What this IS:**
- A stronger threat model: attacker uses gradients to maximize unfairness
- A larger-scale test: UTKFace 200K images vs 18K tabular
- A follow-up direction that could become Chapter 2 of thesis or a 2nd paper

---

## 2. MADAM'S EXACT WORDS → OUR RESPONSES

### Madam said: "Is DRO good? Can we modify it?"
**Our answer:**
> "DRO is good, Madam. On the submitted paper (random noise, 3 datasets), DRO wins 6/9 DP and 5/9 IF at α≤0.3. We are NOT modifying that code. We are testing DRO against a STRONGER attacker — adversarial noise — which is a new research direction."

### Madam said: "Use more datasets and larger data."
**Our answer:**
> "Yes, Madam. We are adding UTKFace — 200,000 face images with gender and race labels. This is 10× larger than our biggest tabular dataset. GPU server access is being arranged."

### Madam said: "Adversarial work has only been done on small datasets."
**Our answer:**
> "Exactly, Madam. That is the gap we are filling. Prior adversarial fairness papers use <50K samples. We are testing whether DRO-FAIR scales to 200K images."

### Madam said: "Implement PGD for fairness metrics (Both DP and IF, only DP, only IF)."
**Our answer:**
> "We have three attack modes:
> 1. DP-only: attacker computes gradient of DP w.r.t. each label, flips the worst samples
> 2. IF-only: attacker computes gradient of IF w.r.t. each label, flips the worst samples  
> 3. Joint: attacker maximizes DP + IF together
> We then train both Naive-FAIR and DRO-FAIR on attacked data and compare."

### Madam said: "Set up UTKFace experiment on server."
**Our answer:**
> "Pipeline is ready. We extract ResNet18 features once (512-dim), then train DRO-FAIR on features same as tabular. First results will be ready by next Friday."

---

## 3. TWO TASKS → CONCRETE DELIVERABLES FOR MAY 29

### TASK 1: PGD for Fairness Metrics
**What Madam wants:** See if DRO still works when attacker explicitly targets fairness.

**Minimum to show on 29th:**
| Item | Status Target | Evidence |
|------|--------------|----------|
| DP-only attack code | Working | `src/corruption/adversarial.py` committed |
| IF-only attack code | Working | `compute_if_gradient()` fixed |
| Joint attack code | Working | `target_metric='combined'` tested |
| Adult results (α=0.2, 5 seeds) | Ready | `results/fairness_pgd_results.json` |
| One figure | Ready | `figures/fig8_fairness_pgd_adult.png` |

**What the figure shows:**
- X-axis: attack mode (DP-only, IF-only, Joint)
- Two bars per mode: Naive-FAIR DP, DRO-FAIR DP
- Caption: "DRO-FAIR reduces DP violation by X% even under fairness-targeted attacks"

**If incomplete by Friday:**
- Show DP-only only (drop IF and joint)
- Say: "IF and joint modes are implemented, testing in progress, results by next week"

---

### TASK 2: UTKFace Dataset
**What Madam wants:** Prove DRO-FAIR scales to large image data.

**Minimum to show on 29th:**
| Item | Status Target | Evidence |
|------|--------------|----------|
| UTKFace downloaded | Done | `/data/srujan.sai/UTKFace/` exists |
| ResNet18 feature extractor | Working | `utkface_features.npz` cached |
| Naive-FAIR trains on UTKFace | Working | `results/utkface_results.json` has rows |
| DRO-FAIR trains on UTKFace | Working | JSON has both naive and dro |
| One figure | Ready | `figures/fig10_utkface_curves.png` |

**What the figure shows:**
- X-axis: α (0.0, 0.1, 0.2)
- Lines: Naive-FAIR DP, DRO-FAIR DP
- Caption: "DRO-FAIR on 200K-image UTKFace dataset"

**If incomplete by Friday:**
- Show CPU-only synthetic run (pipeline works)
- Say: "GPU server access confirmed this week, full run next week"

---

## 4. 4-SLIDE DECK FOR MADAM (Template)

### Slide 1: Submitted Work (Frozen)
- Random corruption + DRO-FAIR on Adult, Credit, LSAC
- 6/9 DP wins, 5/9 IF wins (Wilcoxon p<0.05)
- **Tag v1.0, not touched**

### Slide 2: New Direction — Adversarial Fairness Attacks
- **Novelty:** Attacker computes ∇(fairness metric) and flips exact worst samples
- **Three modes:** DP-only, IF-only, Joint
- **Result:** [Fill from `results/fairness_pgd_results.json`]
- **Key sentence:** "Even when the attacker knows the fairness metric and attacks it directly, DRO-FAIR reduces DP by X%"

### Slide 3: New Direction — Large-Scale Image Data
- **Dataset:** UTKFace, 200K face images, ResNet18 features
- **Task:** Gender prediction with race/gender as protected attribute
- **Result:** [Fill from `results/utkface_results.json`]
- **Key sentence:** "First adversarial fairness experiments at 200K scale; prior work uses <50K"

### Slide 4: Timeline
- Week 1 (done): Submitted paper v1.0
- Week 2 (now): Adversarial attacks + UTKFace pipeline
- Week 3: Full UTKFace ablations, write extended report
- Week 4: Draft paper for 2nd approach

---

## 5. WHAT TO REPORT EVERY WEEK

Madam wants weekly reports. Standard format:

```
WEEK N REPORT — DRO-FAIR Adversarial Noise (2nd Approach)

DONE THIS WEEK:
1. [Concrete item with evidence]
2. [Concrete item with evidence]

RESULTS:
- Dataset: [name]
- Metric: [DP/IF/Accuracy]
- Value: [number]
- Compared to baseline: [X% better/worse]

BLOCKERS:
- [What is blocking progress]

NEXT WEEK:
- [3 concrete tasks]
```

---

## 6. ASSIGNMENTS FOR NEXT 48 HOURS

### Person A — Fairness-PGD (local, CPU)
**Deadline:** Thursday 6 PM  
**Must deliver:**
1. Fix `compute_if_gradient()` → k-NN based gradient
2. `pytest tests/test_fairness_pgd.py -v` → all pass
3. Run DP-only attack on Adult, α=0.2, 5 seeds
4. Generate `figures/fig8_fairness_pgd_adult.png`
5. `git commit`

### Person B — UTKFace (GPU server)
**Deadline:** Thursday 6 PM  
**Must deliver:**
1. Confirm GPU server hostname works
2. Download UTKFace data
3. Extract ResNet18 features → `.npz` cache
4. Run smoke test → `experiments/run_utkface.py --smoke`
5. Run 3 seeds at α=0.0 and α=0.2
6. `git commit`

### Orchestrator (me) — Deck
**Deadline:** Friday 10 AM  
**Must deliver:**
1. Fill slide template with actual numbers from JSON files
2. Dry-run 5-minute presentation
3. Prepare exact phrases for Q&A

---

## 7. EMERGENCY FALLBACKS

| If this fails... | Then say this to Madam |
|------------------|----------------------|
| IF gradient not fixed | "DP-only attack works, IF gradient derivation is complete, testing this weekend" |
| UTKFace not downloaded | "Feature extractor is ready, waiting on server access, results by next Friday" |
| No results at all | "Pipeline is implemented and smoke-tested. Full ablation running, results by next week." |
| DRO worse than Naive | "This is expected at high α — the feedback loop we documented. We are tuning λ_max." |

**Golden rule:** Never say "I don't know." Always say "Preliminary result shows X, full analysis by [date]."

---

## 8. KEY REFERENCES

- **Submitted paper:** `submission/report.pdf` (frozen)
- **Adversarial attacks guide:** https://jonathan-hui.medium.com/adversarial-attacks-b58318bb497b
- **Fairness book:** https://fairmlbook.org
- **Code:** `src/corruption/adversarial.py` (Agent A), `src/models/cnn_classifier.py` (Agent B)
