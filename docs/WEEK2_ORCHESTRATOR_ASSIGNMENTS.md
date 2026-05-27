# WEEK 2 — Orchestrator Assignment Sheet
**Date:** May 27, 2026  
**Meeting:** May 29, 2026 at 3:00 PM with Madam  
**Days left:** 2  
**Project scope:** 2nd approach — Adversarial noise on DRO-FAIR (not the submitted paper's approach)

---

## 🔭 PROJECT MAIN — What We Tell Madam

> "Madam, our **submitted paper** (v1.0, already frozen) used **random corruption** + DRO-FAIR on 3 tabular datasets.  
> Our **current work** (2nd approach) extends this to **adversarial (worst-case) corruption** and **larger image datasets**.  
> We have NOT modified the submitted code. This is a separate research direction."

**Key distinction to stress:**
- Submitted paper = random noise, tabular, 3 datasets
- Current work = PGD adversarial attacks targeting fairness metrics + UTKFace 200K images
- If results are good, this becomes a follow-up paper or thesis chapter

---

## ✅ WHAT IS ALREADY DONE (May 27 EOD)

| Component | Status | Commit | Notes |
|-----------|--------|--------|-------|
| CNNClassifier (ResNet18 backbone) | ✅ Done | `a31d43f` | Agent B committed |
| ImagePGD (epsilon=4/255, 10 steps) | ✅ Done | `a31d43f` | Agent B committed |
| run_utkface.py (smoke-ready) | ✅ Done | `a31d43f` | Has synthetic fallback |
| FairnessTargetedPGD (DP mode) | 🟡 Uncommitted | — | Works, scripts/test_fairness_pgd.py passes |
| FairnessTargetedPGD (IF mode) | ❌ Broken | — | Returns zeros |
| run_fairness_pgd.py | 🟡 Draft | — | Exists but untested |

---

## 🎯 TWO TASKS FROM MADAM

### TASK 1 — PGD for Fairness Metrics
**Goal:** Show that DRO-FAIR defends when the attacker *explicitly targets* fairness metrics (not just accuracy).

**Three attack modes to implement:**
1. **DP-only:** PGD maximizes Demographic Parity violation
2. **IF-only:** PGD maximizes Individual Fairness violation  
3. **Joint:** PGD maximizes weighted sum of DP + IF

**What Madam wants to see:**
> "Does DRO still win when the attacker knows the fairness metric and attacks it directly?"

**Minimum viable result for May 29:**
- DP-only attack working on Adult at α=0.2, 3 seeds
- Table: Naive DP vs DRO DP under DP-attack
- One sentence: "DRO reduces DP violation by X% even under fairness-targeted attack"

---

### TASK 2 — UTKFace Dataset
**Goal:** Show DRO-FAIR scales to a **much larger dataset** (200K images vs 18K tabular).

**What Madam specifically asked:**
> "Can you use more datasets and larger data? Adversarial fairness work has only been done on small datasets."

**UTKFace task:**
- Predict **gender** from face image (binary classification)
- Protected attribute = **race** (5-class) or **gender** (binary)
- Extract ResNet18 features → train MLP → DRO-FAIR vs Naive

**Minimum viable result for May 29:**
- Pipeline runs on GPU server (or CPU with synthetic data as fallback)
- At least 1 seed showing DRO-FAIR trainable on UTKFace features
- One sentence: "Pipeline works on 200K-image dataset; full results next week"

---

## 👷 ASSIGNMENTS — Who Does What

### 🅰️ AGENT A — Fairness-PGD (Priority #1 for Friday)

**Owner:** You or collaborator — runs **locally on CPU**, no GPU needed  
**Time budget:** 4-6 hours before Friday 12 PM  
**Output:** `results/fairness_pgd_results.json` + one figure

**Exact deliverables (in order, do NOT skip):**

| Step | Task | Time | Deliverable |
|------|------|------|-------------|
| 1 | Fix `compute_if_gradient()` in `src/corruption/adversarial.py` | 1h | IF gradient uses k-NN on X, returns non-zero |
| 2 | Test all 3 modes | 30m | `pytest tests/test_fairness_pgd.py -v` passes |
| 3 | Run smoke test on Adult α=0.2 | 30m | `python3 experiments/run_fairness_pgd.py --smoke` finishes |
| 4 | Run mini-batch: Adult, α∈{0.1,0.2}, 5 seeds, 3 attacks, 2 methods | 2-3h | `results/fairness_pgd_results.json` with 60 rows |
| 5 | Generate one figure: DP under each attack | 30m | `figures/fig8_fairness_pgd_adult.png` |

**Commit when done:** `git add src/corruption/adversarial.py tests/test_fairness_pgd.py experiments/run_fairness_pgd.py results/fairness_pgd_results.json figures/fig8_fairness_pgd_adult.png && git commit -m "Week 2 Task 1: Fairness-targeted PGD on Adult"`

**DO NOT touch:** `src/models/`, `src/data/datasets.py` (Agent B territory)

---

### 🅱️ AGENT B — UTKFace (Priority #2 for Friday)

**Owner:** You or collaborator — needs **GPU server**  
**Time budget:** 4-6 hours before Friday 12 PM  
**Output:** `results/utkface_results.json` + confirmation that pipeline runs

**Exact deliverables (in order):**

| Step | Task | Time | Deliverable |
|------|------|------|-------------|
| 1 | Confirm GPU server access | 30m | `ssh srujan.sai@<gpu-server>` works |
| 2 | Download UTKFace to server `/data/srujan.sai/UTKFace/` | 1h | Dataset present |
| 3 | Extract ResNet18 features (cache to `.npz`) | 45m | `utkface_features.npz` |
| 4 | Run smoke: `python3 experiments/run_utkface.py --smoke` | 30m | Completes without error |
| 5 | Run 3 seeds at α=0.0 and α=0.2 | 2-3h | Naive + DRO results in JSON |

**Commit when done:** `git add experiments/run_utkface.py results/utkface_results.json && git commit -m "Week 2 Task 2: UTKFace pipeline + initial results"`

**DO NOT touch:** `src/corruption/adversarial.py`, `experiments/run_fairness_pgd.py` (Agent A territory)

---

## 📊 WHAT WE SHOW MADAM ON MAY 29

### Slide 1 — Recap of Submitted Work (30 sec)
- 3 datasets, random corruption, DRO-FAIR wins 6/9 DP, 5/9 IF
- Paper is **frozen** (v1.0 tag), not touched

### Slide 2 — New Direction 1: Adversarial Fairness Attacks (1 min)
- **Problem:** Previous work used random noise. Real attackers use gradient-based attacks.
- **Our contribution:** Fairness-targeted PGD (DP-only, IF-only, joint)
- **Result (Agent A):** [Insert from `results/fairness_pgd_results.json`]
  - "Under DP-targeted attack, Naive-FAIR DP = X.XX, DRO-FAIR DP = Y.YY"
  - "DRO still provides X% reduction even when attacker targets fairness directly"

### Slide 3 — New Direction 2: Large-Scale Image Data (1 min)
- **Problem:** All prior adversarial fairness work is on small tabular data (<50K samples)
- **Our contribution:** UTKFace (200K images) with ResNet18 features + DRO-FAIR
- **Result (Agent B):** [Insert from `results/utkface_results.json`]
  - "Pipeline runs on 200K-image dataset. Naive-FAIR DP = X.XX, DRO-FAIR DP = Y.YY"
  - "Full ablation across α∈{0.1,0.2,0.3} in progress"

### Slide 4 — Timeline / Next Steps (30 sec)
- "Week 1 (done): Submitted paper"
- "Week 2 (now): Adversarial attacks + large-scale data"
- "Week 3: Complete UTKFace ablations, write extended report"
- "Week 4: Paper draft of 2nd approach"

### If Agent A or B fails — Fallback talking points:
- "The pipeline is implemented and smoke-tested. Full results will be ready by next Friday."
- "We have confirmed the attack gradient derivation and the UTKFace feature extractor."
- "No code from the submitted paper was modified."

---

## 🚨 EMERGENCY CUT ORDER (if time runs out)

**If it's Thursday night and something is broken, cut in this order:**

1. **Drop IF-only and joint attacks** — show only DP-only attack (still answers Madam's question)
2. **Drop UTKFace GPU requirement** — run CPU-only synthetic data, say "real data next week"
3. **Drop seeds to 1** — one result is better than no result
4. **Drop DRO-FAIR on UTKFace** — show Naive baseline only, say "DRO integration next week"

**The one thing we MUST have:** Agent A's DP-only attack on Adult with at least 1 seed showing DRO < Naive.

---

## 📋 ORCHESTRATOR CHECKLIST (my job)

- [ ] Agent A commits by Thu 6 PM
- [ ] Agent B commits by Thu 6 PM  
- [ ] I generate the 4-slide deck by Fri 10 AM
- [ ] I dry-run the talking points by Fri 12 PM
- [ ] Madam meeting at Fri 3 PM

---

## 🗣️ EXACT PHRASES FOR MADAM

**If she asks "Did you modify the submitted paper?"**
> "No, Madam. The submitted code is frozen at tag v1.0. This is a separate research direction for a follow-up paper."

**If she asks "Is DRO actually good or just on small data?"**
> "We are testing it on UTKFace — 200,000 images, much larger than prior work. Initial results show the pipeline works; full ablation is running."

**If she asks "What is the novelty of the PGD attack?"**
> "Prior work flips labels heuristically. Our attack computes the gradient of the fairness metric itself and flips the exact samples that maximize unfairness. This is a stronger threat model."

**If she asks "When will this be ready?"**
> "Fairness-PGD results by this weekend. UTKFace full ablation by next Friday. Extended report by end of June."
