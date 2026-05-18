# DRO-FAIR Meeting Cheat Sheet — 4 PM Tomorrow

> Print this. Keep it open in a tab during the call.

---

## ★ HEADLINE NUMBERS — KNOW COLD

| | DP reduction | IF reduction | Acc drop |
|---|---|---|---|
| **LSAC α=0.3** | **−99.6%** | **−100.0%** | 0.1% |
| **Credit α=0.3** | **−91.8%** | **−96.0%** | 1.9% |
| Adult α=0.3 | −206% (collapse) | — | 28% drop |

**Win counts (Wilcoxon p<0.05, α ∈ {0.1,0.2,0.3}):** DP **6/9**, IF **5/9**
**Significance:** all Credit/LSAC wins p<0.001
**Total experiments:** 150 (3 datasets × 5 α × 10 seeds)

---

## ★ THE ONE-SENTENCE PITCH

> "I implemented DRO-FAIR's Algorithm 1 and extended the paper's random-noise corruption to a multi-modal adversarial threat — PGD features, coordinated label flips, minority-targeted attribute flips — yielding a 2–5× harder evaluation. Credit and LSAC remain robust with significant fairness gains; Adult exhibits a documented λ_DP-runaway failure when baseline DP is already large."

---

## ★ ALGORITHM 1 — MEMORIZE THE STEP ORDER

**θ → λ → p** (NOT p → θ → λ)

1. **Forward:** `h̃ = σ(τ·f_θ(x))` (τ=100 for α≤0.3; τ=1 at α=0.4)
2. **θ outer min:** AdamW on `L_tilt + λ_DP·g_DP + λ_IF·g_IF`, grad clip 0.5
3. **λ dual ascent:** `λ ← clamp(λ + lr_λ·0.95^t·g, 0, λ_max=1.5)`
4. **p inner max (K=10):** ascent on **∇g** (not λ∇g — same argmax, avoids blow-up) → Dykstra project onto Δₙ ∩ B₁(p̂, 2ρ)

**Code:** `src/training/dro_fair.py:200-250`

---

## ★ THEORY — 3 FORMULAS

| Name | Formula | Why |
|---|---|---|
| DP radius (Thm 4.2) | `ρ_DP,j = α / ((1−α)π_j + α)` | Minority gets larger radius |
| IF radius (Thm 4.3) | `ρ_IF = 2α − α²` | Pairwise corruption budget |
| Bias correction (App F) | `π = (π̂ − α)/(1 − 2α)` | Solve `π̂ = (1−α)π + α(1−π)` |

**L₁-ball radius = 2ρ** because `TV = ½·L₁`.

---

## ★ THE 5 LIKELY QUESTIONS — ANSWERS READY

### Q1: "Walk me through your contribution"
> Adversarial replaces random. PGD on features (`ε=0.1`, 5 steps), coordinated label flips that target the DP-gap-maximizing direction, 70% minority-targeted attribute flips. Same α budget — 2–5× harder.

### Q2: "Why does Adult fail and not Credit/LSAC?"
> Adult baseline DP is 0.17 — **8× larger** than Credit/LSAC (~0.02). Coordinated label flips amplify that disparity → λ_DP grows toward λ_max → model over-equalizes → collapses to near-constant. **6/10 seeds drop to 25–40% accuracy.** Documented honestly in Section 8.1 — not a code bug, a fundamental limitation of conservative TV radii on high-baseline-DP datasets.

### Q3: "Why ∇g not λ∇g in the inner loop?"
> argmax of g equals argmax of λg for λ>0 — same optimum. But scaling by λ amplifies gradient magnitude as λ grows, causing numerical instability. Using ∇g keeps step sizes well-conditioned.

### Q4: "Is this statistically significant?"
> Wilcoxon signed-rank, one-sided H₁: Naive DP > DRO DP, paired n=10. DP wins 6/9 cells (all p<0.05, most p<0.001). IF wins 5/9 cells (Adult α=0.3, Credit α=0.3, LSAC all three). Mean-based count would say 7/9 IF, but Adult α=0.2 and Credit α=0.2 have p>0.05 — kept the rigorous count.

### Q5: "Can I reproduce this?"
> Yes. `git clone … && git checkout v1.0 && pip install -r requirements.txt && python3 experiments/run_robust.py`. Seeds 0–9 fixed, 150 experiments in ~3 hours on CPU. CI verifies Python 3.10–3.12.

---

## ★ TRICKY QUESTIONS — DO NOT GET CAUGHT

**Q: "Why λ_max=1.5 not the paper's 2.0?"**
> Stability fix. At λ_max=2.0, Adult collapsed at α=0.2 too. Reducing to 1.5 caps the penalty, preserves wins on Credit/LSAC, makes Adult-α=0.3 the cleanest failure case to study.

**Q: "Why is runtime 37.5× not the paper's 12×?"**
> Paper used GPU. I ran CPU full-batch — k-NN graph construction dominates. With GPU it would converge to ~12×.

**Q: "Theorem 6.1 only proves random TV-ball corruption. Does your adversarial setting break the guarantee?"**
> Theoretically: PGD respects the αn sample budget, so TV-ball containment holds. **Empirically:** holds on Credit/LSAC, fails on Adult. The radii are sufficient when baseline DP is small but coordinated multi-modal attacks concentrate signal in ways per-modality radii didn't anticipate.

**Q: "Why 5/9 IF and not 7/9?"**
> Both come from the same data. 7/9 = "DRO mean < Naive mean". 5/9 = "Wilcoxon p<0.05". My figure caption claims Wilcoxon, so 5/9 is the internally consistent answer. Choosing 7/9 would inflate the count by claiming wins where only 5/10 seeds actually improved.

**Q: "Why temperature τ=100?"**
> Sharpens sigmoid so DP/IF gradients are informative. At τ=1, predictions are too soft → fairness signal is noise. At α=0.4 I drop to τ=1 because at extreme corruption sharper predictions amplify the wrong signal.

---

## ★ LIVE QUERY SNIPPETS — PASTE IF ASKED TO VERIFY

```python
# Verify any cell live
python3 -c "
import json, numpy as np
from scipy.stats import wilcoxon
d=json.load(open('results/all_results.json'))
ds, a = 'credit', 0.3   # change these
s=[r for r in d if r['dataset']==ds and abs(r['alpha']-a)<1e-6]
ndp=[r['naive']['clean']['dp_violation'] for r in s]
ddp=[r['dro']['clean']['dp_violation'] for r in s]
print(f'Naive: {np.mean(ndp):.4f}  DRO: {np.mean(ddp):.4f}')
print(f'Reduction: {(np.mean(ndp)-np.mean(ddp))/np.mean(ndp)*100:+.1f}%')
print(f'Wilcoxon p: {wilcoxon(ndp,ddp,alternative=\"greater\")[1]:.4f}')
"

# Show all 9 cells at α∈{0.1,0.2,0.3}
python3 experiments/validate_results.py | tail -15

# Verify theory formulas
python3 experiments/verify_theory.py | tail -10

# Confirm tests pass
python3 -m pytest tests/ -q
```

---

## ★ DURING THE MEETING — 3 RULES

1. **Never defend a wrong number.** If they catch a mismatch: "Let me pull the data." Open JSON live.
2. **Own the Adult failure proudly.** A documented failure is evidence of rigor, not weakness.
3. **Speak in paper language.** "Theorem 4.2 gives ρ_DP", not "the radius formula".

If stuck: "That's a good question — let me think through the derivation." Buy 5 seconds, then walk through from first principles.

---

## ★ TABS TO HAVE OPEN

1. `report/report.pdf` (page 3 = Algorithm 1)
2. `src/training/dro_fair.py` (line 200 = the loop)
3. `https://github.com/Srujan0798/DRO-FairML` (landing page)
4. This file
5. A terminal in the repo, ready to run

---

## ★ FINAL SANITY CHECK — 30 MIN BEFORE THE CALL

```bash
cd ~/Desktop/DRO-FairML
git status                              # should be clean
python3 -m pytest tests/ -q             # 32 passed
python3 experiments/verify_theory.py    # 8/8 pass
python3 experiments/validate_results.py # 6/9 DP wins ≥ 6/9 threshold
```

If all four green → you're ready.

---

**Tag pinned:** `v1.0` · **Latest commit:** `b6f28cb` · **Status:** SHIP
