# BRUTAL AUDIT — Complete Project State
**Date:** 2026-06-09 · **Verified by:** orchestrator, grounded in current code + JSON

> Every claim below is sourced to a file path, line number, or JSON row.
> No claims without proof. Honest about what we don't know.

---

# 0. TL;DR (read this first)

| Question | Answer |
|---|---|
| Are the bugs from the audit fixed? | **Most yes, one no.** Oracle leak gone, lambda hack gone, pgd_steps=20. K_inner still 5 not 10. |
| Did the attack get stronger? | **Partial.** Works on 6/9 cells; on LSAC DP-mode it *decreases* DP (broken). |
| Does DRO win? | **No — and that itself may be the finding.** Most cells: DRO ≤ Naive. |
| Is the α=0 result valid? | **No — there's a bug.** At α=0 DRO should equal Naive; LSAC shows DRO 6× worse. |
| Was random-vs-adversarial done? | **No.** Script exists, not run. |
| Was UTKFace re-run? | **No.** Still using old data. |
| Should we claim "paper is wrong"? | **No.** We don't have evidence for that. Could be our bugs. |

---

# 1. TIMELINE — Every Request From Madam/Sir

## 1.1 Original assignment (before May 18)
**Source:** professor (auto-memory)
> Replace DRO-FAIR paper's random-noise corruption with adversarial corruption (PGD on features, coordinated label flips, minority-targeted attribute flips). Evaluate on Adult, Credit, LSAC. K_inner=10 and epochs=60 are MANDATORY.

**Status:** Shipped as v1.0, tagged on GitHub. Frozen.

## 1.2 First weekly meeting (May 18)
**Source:** user transcript of madam's message
> **Task 1:** Implement PGD for fairness metrics — both DP+IF, only DP, only IF — and see DRO performance on Adult etc.
>
> **Task 2:** Set up an experiment for the UTKFace dataset on the server and repeat the similar experiment. Use more datasets, larger ones.

## 1.3 Sir's follow-up question (after first sub-meeting)
**Source:** user transcript
> "At lower corruption levels (α=0.1): DRO does not significantly outperform Naive — the attack is too weak to differentiate. **Does the attack affect the radius? If the attack is too weak, then DRO would perform well? Specially at α=0.1.**"

## 1.4 Second weekly meeting (~June 2)
**Source:** user transcript of madam's message
> "**Check the adversarial attack on DP and improve it. Then, redo all the experiments.**"
>
> Madam also said (during the meeting):
> - "In the Naive itself I should see DP to be very, very high under attack. How is it not?"
> - "Just perform DP attack and show me that by attacking DP, you are increasing DP significantly more than the random noise."
> - "If DP and IF is not increasing even after attack, then 'DP is hard to attack' is itself a major finding."
> - "Even with DRO nothing — I think your attack itself is not very successful."
> - "Negative means decrease in DRO. It is DP minus 572.5%. But I cannot see that much difference. Because you are dividing it by 0.02..." (presentation problem)

---

# 2. WHAT WE DID — Phase By Phase

## 2.1 Phase 1: Task 1 — Fairness PGD (May 19–31)
**Outcome:** Implemented `FairnessTargetedPGD` class with three modes (`dp`, `if`, `combined`).
**Files created/modified:**
- `src/corruption/adversarial.py` — added `FairnessTargetedPGD` class
- `experiments/run_fairness_pgd.py` — experiment driver
- `tests/test_fairness_pgd.py` — 8 unit tests

**Results at that time:**
- 270 experiments on Adult/Credit/LSAC
- Claimed: Credit α=0.3 IF → +97.5% DP reduction, LSAC α=0.3 IF → +96.2%
- All committed to `results/fairness_pgd_results.json`

## 2.2 Phase 2: Task 2 — UTKFace baseline (May 27–31)
**Outcome:** UTKFace pipeline with ResNet18 features.
**Files created:**
- `src/data/datasets.py` — `load_utkface()` function
- `src/models/cnn_classifier.py` — CNN model
- `src/corruption/image_pgd.py` — image PGD
- `experiments/run_utkface.py` — UTKFace driver

**Results at that time:** 15 baseline runs at α∈{0.0, 0.1, 0.2}.

## 2.3 Phase 3: First audit (June 8) found 18 bugs
**Source:** brutal audit (this conversation)

| # | Bug | Where | Severity |
|---|---|---|---|
| 1 | Oracle leak: corruption_rates passed to DRO | `experiments/run_fairness_pgd.py:67-75, 108` | CRITICAL |
| 2 | pgd_steps=5 in run script (designed=20) | `experiments/run_fairness_pgd.py:150` | CRITICAL |
| 3 | K_inner=5 (memory says 10 mandatory) | `experiments/run_fairness_pgd.py:149` | HIGH |
| 4 | lambda_max=0.5 hack ONLY on Adult | `experiments/run_fairness_pgd.py:30-33` | CRITICAL |
| 5 | lambda_warmstart=0.01 (not in v1.0 spec) | `experiments/run_fairness_pgd.py:107` | MEDIUM |
| 6 | No α=0.0 baseline | results file | HIGH |
| 7 | No random-vs-adversarial regenerated | `results/random_vs_adversarial.json` dated May 15 | CRITICAL (madam's #1 ask) |
| 8 | Only 3 seeds — Wilcoxon p<0.05 impossible | min p = 0.125 | MEDIUM |
| 9 | UTKFace not re-run with fixed attack | `results/utkface_all_results.json` buckets None | HIGH |
| 10 | STATUS.md claims 270 vs actual 216 | docs | MEDIUM |
| 11 | PGD feature attack uses BCE loss, not DP | `src/corruption/adversarial.py:570` | MEDIUM |
| 12-18 | Various attack code issues | various | MEDIUM |

## 2.4 Phase 4: Fixes applied (June 9)
**Commits:** `292eab9`, `b8e0e25`, `6dcf13a`, `dad7e5a`

### Confirmed fixes (verified by grep):
- ✅ Bug 1 (oracle leak): `grep "corruption_rates" src/training/dro_fair.py` → 0 results
- ✅ Bug 4 (lambda hack): `grep "get_lambda_max" experiments/run_fairness_pgd.py` → 0 results
- ✅ Bug 5 (warmstart): `grep "lambda_warmstart" src/training/dro_fair.py` → 0 results
- ✅ Bug 2 (pgd_steps): line 131 shows `smoke_pgd_steps = 20`
- ✅ Bug 6 (α=0 baseline): 54 runs at α=0 in `results/fairness_pgd_results.json`

### NOT fixed:
- ❌ Bug 3 (K_inner): line 130 shows `smoke_k_inner = 5  # practical speed on CPU`
  - **Deviation reason:** CPU speed (commit `b8e0e25`: "pragmatic: K_inner=5 for CPU feasibility")
  - **Impact:** DRO inner optimization gets half the iterations spec'd
- ❌ Bug 7 (random vs adversarial): `results/random_vs_adversarial.json` still dated May 15
  - Script exists (`experiments/run_random_vs_adversarial.py` 5.2KB) but no fresh JSON output
- ❌ Bug 8 (seeds): still using 3 seeds, p<0.05 still impossible via Wilcoxon
- ❌ Bug 9 (UTKFace): no re-run with fixed attack
- ⚠️ Bug 11 (PGD feature targets BCE): not fixed but not regressing

---

# 3. NEW FINDINGS FROM CURRENT DATA (270 runs, post-fix)

## 3.1 Attack effectiveness — IS THE ATTACK WORKING?

The α=0 baseline (no corruption) vs α=0.3 (heavy corruption) tells us if the attack moves Naive's DP. Source: `results/fairness_pgd_results.json` aggregated.

| Dataset | Attack | Naive_DP α=0 | Naive_DP α=0.3 | Δ | Attack works? |
|---|---|---|---|---|---|
| Adult | DP | 0.157 | **0.531** | +0.374 | ✅ YES (3.4×) |
| Adult | IF | 0.157 | 0.038 | **-0.119** | ❌ NO — DECREASES DP |
| Adult | combined | 0.157 | 0.429 | +0.272 | ✅ YES (2.7×) |
| Credit | DP | 0.013 | 0.038 | +0.025 | ✅ YES (2.9×) |
| Credit | IF | 0.013 | 0.011 | -0.002 | ⚠️ Roughly no change |
| Credit | combined | 0.013 | 0.027 | +0.014 | ✅ YES (2.1×) |
| LSAC | DP | 0.007 | **0.0004** | -0.007 | ❌ NO — DP almost ZEROED OUT |
| LSAC | IF | 0.007 | 0.101 | +0.093 | ✅ YES (14×) |
| LSAC | combined | 0.007 | 0.360 | +0.353 | ✅ YES (50×) |

**Findings:**
- **6/9 attacks actually increase DP** as intended
- **3/9 attacks fail or backfire:**
  - Adult IF mode reduces DP (because IF and DP can be inversely related)
  - LSAC DP mode reduces DP from 0.007 to 0.0004 — this is suspicious
- **This is partially what madam predicted:** "if DP is hard to attack, that's itself a finding"

## 3.2 DRO performance under (real) attack

| Dataset | α | Attack | Naive_DP | DRO_DP | Δ | Verdict |
|---|---|---|---|---|---|---|
| Adult | 0.2 | DP | 0.328 | **0.503** | DRO −53% | DRO LOSES (p=0.028*) |
| Adult | 0.3 | DP | 0.531 | 0.562 | DRO −6% | LOSES (p=0.040*) |
| Adult | 0.4 | DP | 0.310 | 0.283 | DRO +9% | WINS (p=0.0006**) |
| Credit | 0.2 | DP | 0.032 | 0.033 | tied | n.s. |
| Credit | 0.4 | combined | 0.010 | 0.010 | tied | n.s. |
| LSAC | 0.4 | DP | 0.109 | 0.158 | DRO −46% | LOSES (p=0.002**) |
| LSAC | 0.4 | combined | 0.170 | 0.161 | DRO +5% | WINS (p=0.011*) |

**Findings:**
- **Old "+97.5% Credit IF win" and "+96.2% LSAC IF win" are GONE.** Those were oracle-leak artifacts.
- DRO **loses on Adult under DP attack** at α=0.2, 0.3 (consistent finding)
- DRO **only wins at α=0.4 on Adult DP** and **LSAC combined**
- Most cells: tied or DRO worse

## 3.3 α=0 ANOMALY — UNRESOLVED BUG

When α=0 (no corruption), the DRO radius ρ_DP,j = α/((1−α)π_j + α) = 0. With zero radius, the inner-max projection forces p back to uniform weights. So DRO should be mathematically identical to Naive at α=0.

**Reality** (source: `results/fairness_pgd_results.json` rows where α=0.0):

| Dataset | Naive_DP α=0 | DRO_DP α=0 | Δ | Should be 0? |
|---|---|---|---|---|
| Adult | 0.157 | 0.169 | **+0.012** | YES — bug |
| Credit | 0.013 | 0.013 | ~0 | ✅ matches |
| LSAC | **0.007** | **0.045** | **+0.038** | YES — 6× worse, BIG BUG |

**Diagnosis (best guess):** The DRO inner-max loop runs K_inner=5 extra optimizer steps per epoch. Even if the projection puts p back to uniform, those steps consume torch RNG state, which advances the random sequence (dropout masks, etc.) differently than Naive. Over 60 epochs this compounds.

**Severity:** This is a real bug. If DRO differs from Naive without corruption, the entire framework's claims are suspect.

## 3.4 Radii Mismatch Hypothesis — UNVERIFIED

The meeting prep doc claims:
> "DRO's radii formula assumes uniform corruption, but attack uses coordinated targeting (70% minority). On Adult: formula estimates Female=7.5%, true=32.5%."

**Critical thinking:** Theorem 4.2 of the paper proves the radius is a *worst-case* bound over ALL α-budget adversaries — including 100% targeting minority. So mathematically, the formula IS calibrated for coordinated attacks.

**Possibilities:**
- (a) The formula is correctly worst-case, and DRO fails for other reasons (our bugs, not paper's)
- (b) The bias correction `π_clean = (π̂ − α)/(1 − 2α)` is being applied wrong in our code
- (c) The paper's bound is loose at high α, allowing slop

**We do NOT have proof the paper is wrong.** Walking into the meeting and claiming "paper is wrong" without proof = bad science.

---

# 4. WHAT'S STILL BROKEN (concrete list)

| # | Issue | Where | Why It Matters |
|---|---|---|---|
| S1 | K_inner=5 not 10 | `experiments/run_fairness_pgd.py:130` | Spec violation; DRO under-trained |
| S2 | α=0 anomaly: DRO ≠ Naive on Adult and LSAC | data rows | Suggests DRO has reproducibility bug |
| S3 | LSAC DP attack DECREASES DP | data | Attack mis-targeted on LSAC |
| S4 | Adult IF attack DECREASES DP | data | IF and DP inversely related — fix targeting |
| S5 | Random-vs-adversarial NOT regenerated | `results/random_vs_adversarial.json` May 15 | **Madam's #1 explicit question is unanswered** |
| S6 | UTKFace NOT re-run with fixed attack | `results/utkface_all_results.json` | Task 2 still uses buggy data |
| S7 | Only 3 seeds | data | Wilcoxon p<0.05 impossible (n=3 → min p=0.125) |
| S8 | "Radii mismatch" claim unverified | meeting prep doc | May be wrong — paper's bound IS worst case |
| S9 | 19 commits unpushed | git status | Risk of data loss |
| S10 | Sprawl of docs again | repo root | MEETING_CHEAT_SHEET, MEETING_PREP_JUNE_9, STATUS, BUGFIX_SUMMARY, ... |

---

# 5. WHAT WE CAN HONESTLY SAY TO MADAM

## ✅ Defensible claims
1. "I removed the oracle leak (corruption_rates) and the Adult-only lambda_max hack."
2. "I added the α=0 baseline you asked for."
3. "The DP-mode attack increases Naive DP by 3.4× on Adult (0.16→0.53), 2.9× on Credit. Attack works on those."
4. "Without the oracle leak, DRO does NOT outperform Naive on most cells. My previous +97.5% wins were oracle artifacts."
5. "DRO is significantly WORSE than Naive on Adult under DP attack (consistent finding from previous weeks)."

## ⚠️ Honest caveats to state proactively
1. "I haven't yet run the random-noise comparison you explicitly asked for."
2. "UTKFace still uses buggy data — I haven't re-run with the fixed attack."
3. "I'm using K_inner=5 not 10 to fit in CPU budget. That's a spec deviation."
4. "With only 3 seeds, my p-values are exploratory not confirmatory."
5. "I see DRO differing from Naive even at α=0 (no corruption). I don't fully understand why — there may be a reproducibility bug in my DRO code."
6. "I have a hypothesis about radii mismatch under coordinated attacks, but I'm not confident enough to claim the paper is wrong."

## ❌ Things to NOT claim
1. "DRO wins on Credit/LSAC" (oracle artifact)
2. "The paper's theory is wrong" (no evidence)
3. "Significant at p<0.05" (mathematically impossible with n=3)
4. "All experiments redone" (UTKFace not done; random-vs-adv not done)

---

# 6. THE PROPER PATH FORWARD

## Tier 1 — MUST do (otherwise re-presenting bad data)

| # | Task | Where | Cost |
|---|---|---|---|
| T1.1 | Run `experiments/run_random_vs_adversarial.py` end-to-end | local CPU | 3h |
| T1.2 | Fix α=0 anomaly: diagnose why DRO ≠ Naive | `src/training/dro_fair.py` | 2h |
| T1.3 | Re-run UTKFace with fixed code on GPU | server | 8h (depends on GPU) |
| T1.4 | Push 19 unpushed commits | git | 1 min |

## Tier 2 — Should do (improves rigor)

| # | Task | Cost |
|---|---|---|
| T2.1 | Increase to 5 seeds minimum (for any cells used in slides) | 5h |
| T2.2 | Verify "radii mismatch" hypothesis with theory check (re-derive Theorem 4.2 worst case) | 2h |
| T2.3 | Investigate why LSAC DP attack DECREASES DP | 2h |
| T2.4 | Document K_inner=5 deviation in writeup | 30min |

## Tier 3 — Nice to have

| # | Task | Cost |
|---|---|---|
| T3.1 | Restore K_inner=10 and re-run (sanity check K=5 vs K=10 are qualitatively equal) | 10h |
| T3.2 | Add seeds 3,4 → 5 seeds total for all cells | 10h |
| T3.3 | Consolidate sprawling docs again | 1h |

---

# 7. FINAL VERDICT

**We are NOT in a "diamond" state.** We are in:
- **Good** for "I fixed my bugs, here's honest data"
- **Adequate** for "DRO shows mixed results, here are the cells where it wins/loses"
- **Bad** for "I have a publishable finding"

The honest path is: present the honest data, ask madam for guidance on the unresolved questions (α=0 anomaly, radii hypothesis, K_inner trade-off), and propose Tier 1 work for next week.

**Do NOT claim things that can be checked and found false.**
