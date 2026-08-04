# Next Week Plan — Diagnose & Fix the DP Attack
**Today:** Tue June 2, post-meeting · **Next meeting:** Tue June 9, ~3 PM (7 days)

---

## 📌 What Madam Said (the single thing that matters)

> "Just perform DP attack and show me that by attacking DP, you are increasing
> DP **significantly more than random noise**. If you can show that, then
> reproduce all these results."

> "Even with DRO, you are not really seeing any improvement because **your
> attack itself is not very successful.**"

**The week reduces to one yes/no question:** Is our DP attack stronger than random noise on Naive-FAIR?
- **Yes** → improve where possible, then rerun all 285 experiments
- **No** → improve the attack until yes, OR document "DP is hard to attack" as a finding

Also: presentation issue. The "−572%" numbers come from dividing by tiny baselines (0.02). Switch to **absolute DP values** in figures.

---

## 📅 7-Day Schedule

| Day | Date | Owner | Deliverable |
|---|---|---|---|
| 1 | Wed Jun 3 | Agent A | `diagnose_attack_strength.py` + diagnostic table |
| 2-3 | Thu-Fri Jun 4-5 | Agent A | `ImprovedDPAttack` class + unit tests |
| 4 | Sat Jun 6 | Agent A | Re-run diagnostic with new attack |
| 5-6 | Sun-Mon Jun 7-8 | Agent B | Full 285 experiments (random + adversarial side by side) |
| 7 | Mon Jun 8 | Agent B + orch | New figures (absolute scale) + report |
| 8 | Tue Jun 9 | You | Meeting |

---

# 🅰️ AGENT A BRIEF — Day 1 (Wed): Diagnostic

**Copy-paste verbatim to Agent A:**

```
You are AGENT A for /Users/srujansai/Desktop/DRO-FairML.
Today is Wed June 3. Next madam meeting Tue June 9 at 3 PM.

WHY THIS TASK:
Madam said our adversarial attack on DP may not be strong enough. Before we
improve anything, we need NUMBERS to confirm or refute. This is Phase 1 only.
DO NOT modify the attack code today. Just measure.

DELIVERABLE: experiments/diagnose_attack_strength.py

What it does:
- For each (dataset ∈ {adult, credit, lsac}, alpha ∈ {0.1, 0.2, 0.3}, seed ∈ {0..4}):
    - Train Naive-FAIR on CLEAN data → measure DP_clean
    - Train Naive-FAIR on RANDOM-corrupted data → measure DP_random
    - Train Naive-FAIR on CURRENT-AdversarialCorruptor → measure DP_adv_v1
    - Train Naive-FAIR on FairnessTargetedPGD(target='dp') → measure DP_adv_v2
    - Record all 4 numbers
- Save results to results/attack_diagnostic.json with schema:
  [{dataset, alpha, seed, DP_clean, DP_random, DP_adv_v1, DP_adv_v2,
    abs_diff_random, abs_diff_adv_v1, abs_diff_adv_v2,
    ratio_v1_over_random, ratio_v2_over_random}]
- After saving, print a summary table:

  Dataset | α   | DP_clean | DP_rand | DP_adv_v1 | DP_adv_v2 | v1/rand | v2/rand
  --------+-----+----------+---------+-----------+-----------+---------+--------
  adult   | 0.1 | 0.034    | 0.045   | 0.048     | 0.051     | 1.2x    | 1.5x
  ...

INTERPRETATION RULES (state these in your output):
- ratio > 3x: attack is genuinely strong, the issue is elsewhere
- ratio 1.5-3x: attack is okay but improvable
- ratio < 1.5x: attack is too weak (matches madam's diagnosis)

CONSTRAINTS:
- Use existing code only — DO NOT modify src/corruption/adversarial.py today
- Use existing Naive-FAIR trainer
- 5 seeds is enough for diagnostic (statistical sig not needed)
- Total runtime should be ~1h on CPU

OUTPUT: commit experiments/diagnose_attack_strength.py and
results/attack_diagnostic.json with message:
"Phase 1: diagnostic — measure attack strength vs random noise"

REPORT BACK to orchestrator with the summary table and the verdict per dataset.
```

---

# 🅰️ AGENT A BRIEF — Days 2-3 (Thu-Fri): Improve the Attack

**Paste this ONLY AFTER Phase 1 numbers are in and orchestrator approves:**

```
You are AGENT A. Phase 1 diagnostic showed our DP attack is weak (ratio < 3x
on most cells). Madam wants us to improve it. Today is Thu/Fri.

DELIVERABLE: src/corruption/adversarial.py:ImprovedDPAttack class

DESIGN — three changes from existing AdversarialCorruptor:

(1) COORDINATE all three modalities toward DP-maximizing direction
    For each sample i, decide:
      - Should i become "majority-predicted-positive" or "minority-predicted-negative"?
      - That decision determines: feature perturbation direction, label flip target,
        attribute flip target — all consistent.
    Pseudocode:
      For each candidate sample i:
        current_group = a[i]
        # Decision: which direction increases DP gap?
        # Group 0's positive rate UP, Group 1's positive rate DOWN
        # (Assuming group 0 is currently "advantaged")
        if current_group == advantaged_group:
            target_label = 1  # push group 0 positive rate up
            attr_flip_to = advantaged_group  # keep
            feat_direction = +∇x P(y=1|x)
        else:
            target_label = 0  # push group 1 positive rate down
            attr_flip_to = advantaged_group  # flip attr
            feat_direction = -∇x P(y=1|x)

(2) USE A SURROGATE MODEL for gradient estimation
    Current code uses heuristic ∂(DP_gap)/∂y. Replace with:
      - Train a quick surrogate (LR or small MLP) on clean data
      - For each candidate sample, evaluate ΔDP if you flipped it
      - Pick the top αn samples with largest ΔDP

(3) STRONGER FEATURE PGD
    - Increase ε from 0.1 to 0.3
    - Increase pgd_steps from 5 to 20
    - Use the surrogate model's gradient (not just classification loss)

PARAMETERS in __init__:
  alpha=0.2, epsilon=0.3, pgd_steps=20, surrogate='lr',
  coordinated=True, random_state=None

TESTS in tests/test_improved_dp_attack.py:
  test_attack_increases_dp_above_random: assert ratio > 3.0 on synthetic data
  test_coordination: assert feature/label/attribute directions are consistent
  test_alpha_budget: assert sum(corrupt_mask) == int(αn)
  test_deterministic_with_seed: same seed = same output

CONSTRAINTS:
- Keep AdversarialCorruptor untouched (it's our "current" baseline)
- ImprovedDPAttack is NEW class, not a modification
- All 4 tests must pass before commit

OUTPUT: commit src/corruption/adversarial.py and tests/test_improved_dp_attack.py
with message: "Phase 2: ImprovedDPAttack with coordinated multi-modal + surrogate gradient"

REPORT BACK with test output and 1 paragraph describing the algorithm.
```

---

# 🅰️ AGENT A BRIEF — Day 4 (Sat): Validate Improved Attack

**Paste after Phase 2 is committed:**

```
You are AGENT A. ImprovedDPAttack is implemented. Today: validate it.

DELIVERABLE: experiments/diagnose_improved_attack.py (extends Phase 1 script)

What it does:
- Same as diagnose_attack_strength.py BUT adds a 5th column: DP_adv_v3
  (using ImprovedDPAttack)
- Same 3 datasets, 3 alphas, 5 seeds
- Save to results/attack_diagnostic_v2.json
- Print updated summary table with all 5 conditions
- Compute ratio_v3_over_random per cell

SUCCESS CRITERIA:
- ratio_v3_over_random > 5x on at least 6 of 9 cells (3 datasets × 3 α)
- If yes → Phase 3 PASS, escalate to orchestrator to launch Phase 4
- If no → diagnose why. Try:
    (a) increase ε to 0.5
    (b) use 50 PGD steps
    (c) test on synthetic data with known fairness structure
  After 2 more attempts, escalate to orchestrator with findings.

OUTPUT: commit with message "Phase 3 validation: ImprovedDPAttack vs current"

REPORT BACK with the 5-column table and the verdict.
```

---

# 🅱️ AGENT B BRIEF — Days 5-6: Full Re-run with Random Baseline

**Paste after Agent A's Phase 3 validates:**

```
You are AGENT B for /Users/srujansai/Desktop/DRO-FairML.
Agent A validated ImprovedDPAttack. Today: rerun all experiments with the new
attack + add random-corruption baseline side by side.

DELIVERABLE: experiments/run_full_v2.py

What it does:
- For each (dataset ∈ {adult, credit, lsac, utkface}, alpha ∈ {0, 0.1, 0.2, 0.3},
            seed ∈ {0..4}, corruption ∈ {random, improved_adv}, method ∈ {naive, dro}):
    - Train method on corruption-corrupted data
    - Evaluate on clean test set
    - Record: acc, dp, if
- Total: 4 × 4 × 5 × 2 × 2 = 320 experiments
  Tabular runs ~30s each on CPU = ~2.5h
  UTKFace runs on GPU = ~4h (if server up)
- Save to results/full_v2_results.json

RUN ORDER:
- Tabular first (CPU, overnight)
- UTKFace second (GPU, if available — otherwise mark as pending)

SCHEMA:
  {dataset, alpha, seed, corruption, method, acc, dp, if, runtime}

OUTPUT: commit with message "Phase 4: Full re-run with ImprovedDPAttack + random baseline"

REPORT BACK with total experiment count and any failures.
```

---

# 🅱️ AGENT B BRIEF — Day 7: Clean Figures

**Paste after Phase 4 completes:**

```
You are AGENT B. Full results are in. Build the new figures madam asked for.

DELIVERABLES: experiments/generate_figures_v2.py producing:

(1) figures/fig_attack_validity.pdf
    Bar chart per dataset (3 subplots): adult, credit, lsac
    For each subplot, x-axis = alpha values, grouped bars:
      - Bar 1: DP under clean training (baseline)
      - Bar 2: DP under random corruption
      - Bar 3: DP under our improved adversarial attack
    Y-AXIS: ABSOLUTE DP values (not percentages!)
    Y-axis range: 0 to max(observed DP) + 0.05
    Title: "Attack Validity: Adversarial DP increase vs Random Noise"

(2) figures/fig_dro_defense.pdf
    3 subplots (datasets). For each:
      x = alpha
      Lines: Naive-FAIR DP under improved attack (red),
             DRO-FAIR DP under improved attack (green)
      Y-AXIS: ABSOLUTE DP values
      Title: "DRO Defense Under Improved Adversarial Attack"

(3) figures/fig_random_vs_adv.pdf (the key validation figure)
    2x3 grid:
      Top row: Naive-FAIR DP under random / adversarial (side by side per dataset)
      Bot row: DRO-FAIR DP under random / adversarial (side by side per dataset)
    Show that adversarial >> random on Naive top row.
    Show that DRO closes the gap on bottom row.

STYLE:
- Computer Modern fonts
- Error bars (caps, no shading)
- 300 dpi, both PDF and PNG
- Colors: Naive #c44e2b, DRO #1a7a3a, Random #888888

OUTPUT: commit with message "Phase 5: New figures with absolute DP scale"

REPORT BACK with file sizes and 1 paragraph describing each figure.
```

---

# 🟢 ORCHESTRATOR (me) — End of Week

I will:
1. **Daily standup review** — verify each phase's numbers before next phase starts
2. **Day 7 evening:** write `NEXT_TUESDAY_REPORT.md` with:
   - Diagnostic verdict
   - Improved attack design (1 paragraph)
   - Validation numbers (table)
   - Full results comparison
   - 3 honest findings
3. **Day 7 evening:** archive old fig8/fig9 to `figures/_old/` (madam said the −572% format is confusing)
4. **Tue morning:** dry-run the meeting with you

---

# ✅ Success Criteria for Tuesday June 9 Meeting

You walk in and say:

> "Madam, the diagnostic confirmed your suspicion — the old attack only
> increased Naive DP by ~1.4x over random noise. I implemented ImprovedDPAttack
> with coordinated multi-modal perturbation and a surrogate gradient.
> Validation: new attack is **5.8x stronger** than random noise on average.
> Reran all 285 experiments plus random baselines. With the stronger attack,
> DRO defense is now clearly visible — here's the new figure with absolute DP
> values, not percentages."

OR if attack stays weak:

> "Madam, after three improvement attempts, DP remains hard to attack on these
> three tabular datasets. We documented this as the finding — fairness on
> Adult/Credit/LSAC may be intrinsically robust to small-α adversaries. Here's
> the evidence."

**Both outcomes are valid. The point is to KNOW which one is true.**

---

# 🛑 Quality Rules (you asked for these)

1. **One number per claim.** No more "X% of Y normalized by Z".
2. **Absolute DP everywhere.** Percentages only when baseline is meaningful.
3. **Diagnose before coding.** Phase 1 numbers gate Phase 2.
4. **One figure per finding.** Not 7 variations.
5. **Daily 3-line update.** Not 10-page reports.
6. **Don't touch v1.0.** That's the original submission. Frozen.

---

# 🚨 Drop-Scope Triggers

- **GPU down all week** → ship tabular-only (no UTKFace re-run). Madam will accept.
- **Improved attack still weak** → ship Outcome B (honest finding). Don't fake numbers.
- **Time pressure on Day 6** → ship random + improved adversarial only, skip the old AdversarialCorruptor comparison.

---

# 📋 Today (Tue evening, post-meeting)

**You:** rest. You got scolded. Don't open the laptop tonight.

**Tomorrow morning:** spawn Agent A with the Phase 1 brief above. That's it.
