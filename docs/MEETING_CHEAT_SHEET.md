# MADAM MEETING CHEAT SHEET
**May 29, 2026, 3:00 PM | Weekly Report #1**

---

## OPENING (30 seconds)

> "Madam, this week we completed BOTH tasks you assigned. Task 1 — gradient-based PGD attacks on tabular data — 270 experiments done. Task 2 — UTKFace on GPU server — 9 experiments done with real ResNet18 features. I will show you both results and the code."

---

## TASK 1: PGD FOR FAIRNESS METRICS (2 minutes)

**What we built:**
- `FairnessTargetedPGD` class in `src/corruption/adversarial.py`
- Computes exact gradient of DP/IF w.r.t. each label
- Three attack modes: DP-only, IF-only, Combined
- PGD iterative optimization (5 steps, coordinated flips)

**Show code (if asked):**
```bash
# Open src/corruption/adversarial.py
# Lines 204-350: FairnessTargetedPGD class
```

**Show figures:**
- `figures/fig8_fairness_pgd_comparison.png` — grouped bars
- `figures/fig9_fairness_pgd_curves.png` — line plots with error bars

**Key results (tabular data: Adult, Credit, LSAC):**

| Attack | Dataset | α | DRO Improvement | p-value |
|--------|---------|---|-----------------|---------|
| IF | Credit | 0.2 | 64.5% reduction | 0.031* |
| IF | Credit | 0.3 | 97.5% reduction | 0.031* |
| IF | LSAC | 0.3 | 96.2% reduction | 0.031* |
| DP | Adult | 0.3 | Both collapse | Known limitation |

> "Under IF-targeted attacks, DRO significantly reduces DP violation by 64-97% (p<0.05). This is our strongest result."

**If Madam asks "Why does Adult DP attack fail?"**
> "Adult has high baseline DP (~0.17). At α=0.3, coordinated DP-targeted flips trigger lambda runaway. This is NOT a bug — it's a real limitation we documented. Credit and LSAC have low baseline DP, so DRO works well. Also, DP-targeted attacks are inherently stronger than IF-targeted because they directly optimize the same metric DRO is trying to protect."

---

## TASK 2: UTKFACE ON GPU SERVER (2 minutes)

**What we did:**
- Extracted ResNet18 features from 23,705 UTKFace images → 512-dim vectors
- Trained Naive-FAIR vs DRO-FAIR on GPU server (flair2.iitgn.ac.in)
- 3 seeds × 3 corruption levels (α = 0.0, 0.1, 0.2)

**Show figure:**
- `figures/fig_utkface_dp_comparison.png`
- `figures/fig_utkface_tradeoff.png`

**Results (UTKFace — NEW FINDING):**

| α | Naive DP | DRO DP | Winner | Notes |
|---|----------|--------|--------|-------|
| 0.0 (clean) | 0.036 | **0.027** | DRO | 25% better on clean data |
| 0.1 (corrupt) | **0.094** | 0.130 | Naive | DRO is 39% WORSE |
| 0.2 (corrupt) | **0.103** | 0.110 | Naive | DRO is 7% WORSE |

> **"This is a surprising finding, Madam. On image data with ResNet features, DRO actually makes fairness WORSE under label corruption. On tabular data (Credit, LSAC), DRO helps. This suggests the effectiveness of DRO depends on the data modality — an important insight for the paper."**

**If Madam asks "Why is DRO worse on images?"**
> "We think it's because ResNet features are already fairness-agnostic. When labels are corrupted, DRO overfits to the corrupted fairness signal. On tabular data, features carry more demographic information, so DRO can still find robust patterns. This is a hypothesis we are testing further."

---

## IF MADAM ASKS SPECIFIC QUESTIONS

**"Did you modify the submission?"**
> "No, Madam. The ICML submission is frozen. All this work is for a follow-up paper on adversarial fairness."

**"Is DRO actually good?"**
> "It depends. DRO is excellent against IF-targeted attacks on tabular data — 64-97% reduction, statistically significant. Against DP-targeted attacks, the adversary is stronger. And on image features, DRO can actually hurt under corruption. This nuance is what makes the research interesting."

**"What is novel here?"**
> "Prior work only tested random label noise. We are the first to compute the exact gradient of the fairness metric and flip the optimal samples. This is a much stronger, more realistic threat model. Our results show DRO's robustness is metric-dependent and modality-dependent."

**"What next?"**
> "Madam, we want to: (1) Run more seeds for UTKFace to confirm the finding, (2) Test on CelebA or FairFace for larger-scale validation, (3) Try deeper networks (ResNet50), (4) Write the extended report. You mentioned more GPU access — that will help us scale."

**"Can you show me the code running?"**
> "Yes. The server experiments completed. I can show you the logs and the feature extraction script."

---

## CLOSING (30 seconds)

> "Madam, to summarize: Task 1 complete — DRO defends well against IF attacks on tabular data. Task 2 complete — UTKFace shows a surprising result where DRO hurts under corruption on image features. Both tasks are done. We are ready for the next phase: larger datasets and deeper analysis. No submission code was modified."

---

## FILES TO HAVE OPEN

1. **Terminal:** `results/utkface_results.json` (9 experiments, real data)
2. **Image viewer:** `figures/fig_utkface_dp_comparison.png`
3. **Image viewer:** `figures/fig8_fairness_pgd_comparison.png`
4. **Text editor:** `src/corruption/adversarial.py` (FairnessTargetedPGD class)
5. **Browser (if needed):** GitHub repo at commit `977422d`

---

## EMERGENCY FALLBACKS

**If Madam asks something unexpected:**
> "That's an excellent point, Madam. Let me note it and include it in next week's report."

**If server/logs question comes up:**
> "All experiments ran on flair2.iitgn.ac.in with 2× NVIDIA L40S GPUs. Logs are saved at `/data/srujan.sai/utkface_real.log`."

**If asked about submission timeline:**
> "The ICML submission is already submitted and frozen. This adversarial fairness work is for a follow-up submission — NeurIPS or ICLR deadline."
