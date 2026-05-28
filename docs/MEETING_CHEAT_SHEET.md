# MADAM MEETING CHEAT SHEET
**May 29, 2026, 3:00 PM**

---

## OPENING (30 seconds)

> "Madam, this week we completed both tasks you assigned. Task 1 — gradient-based PGD attacks targeting fairness metrics — is fully implemented and tested on 270 experiments. Task 2 — UTKFace pipeline — is ready and waiting on GPU server access."

---

## TASK 1: PGD FOR FAIRNESS METRICS (2 minutes)

**What we built:**
- `FairnessTargetedPGD` class in `src/corruption/adversarial.py`
- Computes exact gradient of fairness metric w.r.t. each label
- Three modes: DP-only, IF-only, Joint
- PGD iterative optimization (5 steps)

**Show code:**
```python
# Open src/corruption/adversarial.py, scroll to line 204
class FairnessTargetedPGD:
    def compute_dp_gradient(self, y, a):
        # Computes d(DP)/d(y_i) for each sample
    def compute_if_gradient(self, y, a, X):
        # Uses k-NN to compute d(IF)/d(y_i)
```

**Show figure:**
- Open `figures/fig8_fairness_pgd_comparison.png`
- Point to Credit IF-Attack α=0.2: "Green bar (DRO) much lower than red (Naive)"
- Point to Adult DP-Attack: "Both high — attacker is strong here"

**Key numbers:**
| Result | What to say |
|--------|-------------|
| Credit IF α=0.2: 64.5% reduction, p=0.031 | "DRO significantly reduces DP violation under IF attacks" |
| Credit IF α=0.3: 97.5% reduction, p=0.031 | "At higher corruption, DRO's defense is even stronger" |
| LSAC IF α=0.3: 96.2% reduction, p=0.031 | "Same pattern on LSAC — consistent result" |
| Adult DP α=0.3: both collapse to 0 | "Known issue — Adult feedback loop at high alpha, documented in submission" |

**If Madam asks "Why does Adult fail?"**
> "Adult has high baseline DP (~0.17). At α=0.3, coordinated label flips trigger lambda runaway. This is NOT a bug — it's a real limitation we documented in the submitted paper. Credit and LSAC have low baseline DP, so DRO works well."

---

## TASK 2: UTKFACE (1 minute)

**Show pipeline:**
```bash
# Open terminal
cat scripts/extract_utkface_features.py | head -30
```

**What to say:**
> "ResNet18 feature extractor is ready. 200K images → 512-dim features → DRO-FAIR. Pipeline tested locally with synthetic data. Smoke test passes. Waiting on server credentials to run full extraction."

**Show smoke test result:**
```bash
cat results/utkface_results.json
```
> "Synthetic run: Naive DP=0.077, DRO DP=0.053. Real data as soon as server is accessible."

---

## IF MADAM ASKS SPECIFIC QUESTIONS

**"Did you modify the submission?"**
> "No, Madam. Submission v1.0 is frozen at tag v1.0. This is a separate research direction for a follow-up paper."

**"Is DRO actually good?"**
> "DRO is good against IF-targeted attacks — 64-97% reduction, statistically significant. Against DP-targeted attacks on Adult, the adversary is stronger. This is a NEW finding — prior work only tested random noise, not adversarial."

**"When will UTKFace be ready?"**
> "Pipeline is implemented. Waiting on GPU server access from sysadmin. Feature extraction takes 45 minutes on GPU. Full results by next Friday."

**"What is novel here?"**
> "Prior work flips labels heuristically. We compute the exact gradient of the fairness metric and flip the optimal samples. This is a stronger, more realistic threat model."

**"Can you do more?"**
> "Yes, Madam. Next steps: (1) UTKFace full run on GPU, (2) Test other image datasets, (3) Write extended report, (4) Draft paper for this 2nd approach."

---

## CLOSING (30 seconds)

> "Madam, to summarize: Task 1 is complete with strong results on Credit and LSAC. Task 2 pipeline is ready. We are on track for full UTKFace results by next week. No submission code was modified."

---

## FILES TO HAVE OPEN

1. Terminal showing `results/fairness_pgd_results.json` (270 rows)
2. Image viewer: `figures/fig8_fairness_pgd_comparison.png`
3. Text editor: `docs/ADVERSARIAL_FAIRNESS_REPORT.md`
4. Browser (if needed): GitHub repo showing commits

---

## EMERGENCY FALLBACKS

**If laptop crashes / figures won't open:**
> "Madam, all results are committed to GitHub. I can pull them up on any browser."

**If Madam asks something unexpected:**
> "That's an excellent point, Madam. Let me note it and include it in next week's report."

**If UTKFace server still not accessible:**
> "We confirmed the pipeline works end-to-end with synthetic data. The only blocker is server credentials, which we've requested."
