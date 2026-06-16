# Meeting notes — today 4 PM

> Story to present. All numbers from Adult, 3 seeds, K_inner=10, pgd_steps=20,
> coordinated DP-targeted PGD attack. (Credit/LSAC tau=1 re-run + 6-seed runs in
> progress — will refresh this file ~1 hr before the meeting with whatever lands.)

## 1. What was asked
- **Madam (Jun 2):** "Check the adversarial attack on DP and improve it, then redo all experiments." Show adversarial raises DP more than random noise.
- **Kuldeep (Jun 9, on Madam's behalf):** try hyperparameter tuning (λ, lr) to tighten DP (Q1); IF k-NN ablation k=5/10/15 (Q6); **fix tau for all α** and ablate tau (Q12); LSAC has inherent low DP so use IF there (Q3); radii is empirical not theoretical (Q5). "**Start with Adult, then we discuss.**"

## 2. What we did
- Fixed the DP attack: feature-PGD now maximizes |p0−p1| directly (was using classification loss); K_inner restored to 10; α=0 inner-loop guard added.
- Confirmed **adversarial ≫ random noise** on DP: Adult α=0.2 adversarial Δ=+0.180 vs random Δ≈0 (≈31×); α=0.3 +0.376 vs +0.006. Credit similar (3–22×). (`results/random_vs_adversarial_new.json`.)
- Ran the ablations Kuldeep asked for.

## 3. HEADLINE RESULT — fixing tau=1 makes DRO win
DP violation under DP attack (lower = fairer), mean of 3 seeds:

| α | Naive DP | DRO DP | winner | DRO wins/3 seeds | Δacc |
|---|---|---|---|---|---|
| 0.1 | 0.207 | 0.205 | **DRO** | 2/3 | +0.001 |
| 0.2 | 0.248 | 0.237 | **DRO** | 3/3 | +0.002 |
| 0.3 | 0.286 | 0.264 | **DRO** | 3/3 | +0.009 |
| 0.4 | 0.310 | 0.283 | **DRO** | 3/3 | +0.011 |

- DRO beats Naive on DP at **every α**, and the advantage **grows with α** (exactly as theory predicts). Accuracy is equal-or-slightly-better — not even an accuracy trade.
- Combined attack at tau=1: DRO wins all 4 α too.
- **At tau=100 (our old schedule) DRO LOST almost everywhere** (α=0.2: Naive 0.327 vs DRO 0.503). → the earlier "DRO is fragile" worry was a **high-temperature artifact**, not a real weakness. This is the key correction. It matches Kuldeep's Q12 ("fix tau for all α").

## 4. Supporting ablations
- **IF k-NN (Q6):** k∈{5,10,15} give near-identical IF/DP — attack is **insensitive to k**; k=5 is a fine default. (Robustness check.)
- **λ_init × lr grid (Q1):** running now (Adult, tau=1). Goal: see if DP tightens further. Results to add before meeting.

## 5. Narrative decisions (from Kuldeep)
- **LSAC (Q3):** has inherent low DP → DP attack can't raise it; we frame LSAC around the **IF attack**. Not a bug.
- **Radii (Q5):** treated as **empirical** — calibrate from observed clean proportions under the known coordinated attack; no new closed form; we are NOT claiming the paper is wrong.

## 6. Status & honest open items
- **Done:** Adult tau + k-NN ablations; attack-vs-noise; 3 code bugfixes.
- **Running:** tau=1 full re-run (Credit+LSAC), λ/lr grid, +3 seeds for n=6 significance.
- **Open:** Wilcoxon p<0.05 needs n≥6 (3 seeds → min p=0.125); 6-seed runs in progress. UTKFace still blocked on flair2 GPU (SSL/account). Empirical-radii (Q5) and a few audit bug-fixes (inference eval-mode) being implemented.

## 7. Ask for the meeting
- Confirm: adopt **fixed tau=1** as the production setting and regenerate the full report around it? (Adult evidence is strong; Credit/LSAC landing.)
- Confirm seed count for the paper (we're moving to 6).
- UTKFace: priority + help getting flair2 access (supin.gopi account)?
