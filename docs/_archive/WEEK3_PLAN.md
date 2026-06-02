# Week 3 Plan — Tuesday June 2 Meeting
**Today:** Sunday May 31, 2026 · **Meeting:** Tuesday June 2, 3 PM (~48 hours)
**Owner:** Srujan Sai · **Orchestrator:** Claude

---

## 🎯 The Story for Tuesday

> "Madam, last week you asked: (1) PGD on fairness metrics, (2) UTKFace experiment.
> I shipped both on tabular data, then **extended** the UTKFace experiment to use the
> new fairness-PGD attacks too. I found that **DRO's behavior inverts on image
> features** — it helps on tabular, hurts on UTKFace. Here are my three hypotheses
> for why, and the next investigations."

---

## 📊 Where We Are Now

| Layer | Status |
|---|---|
| Original v1.0 submission (DRO-FAIR on tabular) | ✅ Frozen, no changes |
| Task 1: FairnessTargetedPGD (DP/IF/combined) on tabular | ✅ 270 expts, fig8/fig9, report |
| Task 2: UTKFace baseline (with Week 1 corruption) | ✅ 15 expts at α∈{0,0.1,0.2}, fig10 |
| **GAP 1:** FairnessTargetedPGD on UTKFace | ❌ NOT YET — Agent A this week |
| **GAP 2:** UTKFace at α=0.3, 0.4 | ❌ NOT YET — Agent B this week |
| **GAP 3:** Hypothesis for "DRO inverts on images" | ❌ NOT YET — Agent B this week |

---

# 🅰️ AGENT A BRIEF — UTKFace × FairnessTargetedPGD

**Copy-paste to Agent A:**

```
You are AGENT A for /Users/srujansai/Desktop/DRO-FairML.
Today is Sunday May 31. Madam meeting Tuesday June 2 at 3 PM.

GOAL: Run the new FairnessTargetedPGD attacks on UTKFace and produce results.

CONTEXT:
- FairnessTargetedPGD exists in src/corruption/adversarial.py.
  It supports target_metric='dp', 'if', 'combined'. Works on tabular data (270 expts).
- UTKFace pipeline in experiments/run_utkface.py uses ImagePGD or AdversarialCorruptor.
- 15 UTKFace baseline experiments exist in results/utkface_results.json.

YOUR DELIVERABLES — STRICTLY IN ORDER:

== STEP 1 (1h): Wire FairnessTargetedPGD into run_utkface.py ==
- Add CLI flag --fairness_attack {dp,if,combined,none} (default: none = current behavior)
- When set, replace the corruptor with FairnessTargetedPGD(
    alpha=alpha, target_metric=attack, coordinated=True, random_state=seed)
- For IF mode, pass X (ResNet18 cached features) to compute_if_gradient
- Save results to results/utkface_fpgd_results.json (separate file from baseline)
- Each output row schema:
  {dataset:'utkface', alpha:float, seed:int, attack:str, method:str,
   acc_clean:float, dp_clean:float, if_clean:float, total_time:float}

== STEP 2 (30 min): Smoke test ==
Run: python3 experiments/run_utkface.py --smoke --fairness_attack dp
- Must complete in <10 min
- Must produce valid JSON with at least 1 row
- If smoke crashes, fix and rerun BEFORE proceeding

== STEP 3 (overnight, on GPU): Full run ==
On GPU server (flair2.iitgn.ac.in if accessible):
  nohup python3 experiments/run_utkface.py \
    --alphas 0.1 0.2 0.3 \
    --n_seeds 5 \
    --methods naive dro \
    --fairness_attacks dp if combined \
    > logs/utkface_fpgd.log 2>&1 &
Expected total: 3 attacks × 3 α × 5 seeds × 2 methods = 90 runs.
At ~3-5 min/run on GPU = ~5-7 hours. Should finish overnight.

If GPU NOT accessible: use cached ResNet features on CPU, drop to 2 seeds
and only α=0.1, 0.2 → 12 runs, ~3 hours.

== STEP 4 (Mon AM, 2h): Analysis + figure ==
Create experiments/analyze_utkface_fpgd.py:
- Load results/utkface_fpgd_results.json
- For each (attack, α): compute mean ± SE of (acc, dp, if) across seeds
- Wilcoxon paired test: naive_dp vs dro_dp (alt='greater')
- Save to results/utkface_fpgd_wilcoxon.csv with columns:
  attack, alpha, n, naive_dp, dro_dp, dp_reduction_pct, dp_pvalue,
  naive_if, dro_if, if_reduction_pct, if_pvalue

Create figures/fig11_utkface_fpgd.pdf:
- 3 subplots (one per attack mode)
- x-axis: α ∈ {0.1, 0.2, 0.3}
- y-axis: DP violation
- Lines: Naive (red #c44e2b), DRO (green #1a7a3a) with error bars (caps, no shading)
- Title for each subplot: "Attack: DP" / "Attack: IF" / "Attack: Combined"
- Suptitle: "UTKFace under FairnessTargetedPGD (n=5 seeds per cell)"
- Computer Modern fonts, figsize=(15, 5), 300 dpi
- Save both .pdf and .png

== STEP 5 (Mon PM, 30 min): Commit + push ==
git add experiments/run_utkface.py experiments/analyze_utkface_fpgd.py \
        results/utkface_fpgd_results.json results/utkface_fpgd_wilcoxon.csv \
        figures/fig11_utkface_fpgd.*
git commit -m "Week 3: UTKFace × FairnessTargetedPGD (90 expts) + analysis + fig11"
git push origin main

== STEP 6 (Mon PM, 30 min): One-paragraph results summary ==
Append to docs/UTKFACE_RESULTS.md a section titled
"## Week 3 Extension: FairnessTargetedPGD on UTKFace"
- One paragraph stating the headline result with numbers
- Bullet list: 3 key findings (e.g. "DRO defends/fails under DP attack at α=X with p=Y")

RULES:
- DO NOT touch: src/training/dro_fair.py, src/training/naive_fair.py,
  src/corruption/adversarial.py (only IMPORT and USE — never modify).
- DO NOT delete or overwrite the existing utkface_results.json (it's the baseline).
- COORDINATE: Agent B is also using GPU; agree on run windows so you don't conflict.

REPORT BACK at end of each step boundary.
```

---

# 🅱️ AGENT B BRIEF — Extend baseline + Hypothesis writeup

**Copy-paste to Agent B:**

```
You are AGENT B for /Users/srujansai/Desktop/DRO-FairML.
Today is Sunday May 31. Madam meeting Tuesday June 2 at 3 PM.

GOAL: (1) Extend UTKFace baseline to α=0.3, 0.4. (2) Write hypothesis for
why DRO inverts on images.

CONTEXT:
- results/utkface_results.json has 15 baseline experiments (α=0,0.1,0.2; 5 seeds).
- Observation: DRO yields HIGHER DP than Naive at α=0.1 (−35%) and α=0.2 (−10%).
- This INVERTS the tabular result where DRO yields lower DP on Credit/LSAC.

== STEP 1 (15 min): Wait for Agent A ==
Agent A is modifying experiments/run_utkface.py to add --fairness_attack flag.
DO NOT start Step 2 until Agent A reports they've committed Step 1.
While waiting, draft the hypothesis doc structure (Step 4 below).

== STEP 2 (2h on GPU): Extend baseline ==
After Agent A's commit lands:
  nohup python3 experiments/run_utkface.py \
    --alphas 0.3 0.4 \
    --n_seeds 5 \
    --methods naive dro \
    > logs/utkface_extend.log 2>&1 &
Expected: 2 α × 5 seeds × 2 methods = 20 runs, ~1.5h on GPU.
IMPORTANT: Append to results/utkface_results.json (don't overwrite).
If run_utkface.py overwrites by default, save to results/utkface_extend.json
and merge with a small Python script:
  python3 -c "
  import json
  base = json.load(open('results/utkface_results.json'))
  new  = json.load(open('results/utkface_extend.json'))
  combined = base + new
  json.dump(combined, open('results/utkface_results.json','w'), indent=2)
  "

== STEP 3 (30 min): Regenerate fig10 with full α range ==
Update experiments/generate_fig10.py or analyze_utkface.py to include α=0.3, 0.4
in the curves. Regenerate figures/fig10_utkface_curves.pdf.

== STEP 4 (Mon AM, 2h): Write hypothesis doc ==
Create docs/UTKFACE_INVERSION_HYPOTHESIS.md with this structure:

  # Why Does DRO Invert on UTKFace?

  ## Observation
  - On tabular (Credit/LSAC): DRO reduces DP by 64–97% under corruption (p<0.05).
  - On UTKFace (ResNet18 features): DRO INCREASES DP relative to Naive
    (−35% at α=0.1, −10% at α=0.2). Baseline (α=0): DRO is +23% better.

  ## Three candidate hypotheses

  ### H1: ResNet18 features are already pre-fair
  - The CNN was pretrained on ImageNet; its features may be implicitly balanced
    across the gender attribute used for UTKFace.
  - DRO's worst-case reweighting then over-corrects in a low-DP regime.
  - Evidence FOR: baseline DP on UTKFace (α=0) is 0.029, comparable to Credit (0.02)
    — but DRO still helps on Credit. So pre-fairness alone is unlikely the full story.

  ### H2: The corruption signal doesn't propagate well through ResNet18
  - We attack the cached 512-dim ResNet features directly. The features were
    extracted from clean images, so attacking them doesn't simulate real-world
    image-space corruption.
  - DRO is therefore defending against a corruption pattern that doesn't match
    what would happen in practice.
  - Evidence: image-space PGD (pixel attacks → re-extract features) might give
    different results.

  ### H3: Inner maximization amplifies noise on continuous embeddings
  - On tabular, features are standardized but discrete-ish; the inner max over p
    finds clear high-disparity samples.
  - On 512-dim continuous embeddings, the inner max may find spurious directions
    that don't correspond to real fairness violations, leading λ_DP to over-grow.

  ## Proposed next experiments (Week 4+)
  1. Trace λ_DP over training epochs on UTKFace — does it runaway like Adult?
  2. Run with λ_max=0.5 (instead of 1.5) on UTKFace — does DRO stop inverting?
  3. Run image-space PGD (attack pixels, re-extract features) vs feature-space attack
  4. Try a different backbone (ResNet50, or random features) to test H1

  ## Honest caveats
  - n=5 seeds per cell is small; results may shift with more seeds.
  - GPU runs are deterministic but ResNet feature extraction has tiny float drift.
  - We have no theory for why DRO should/shouldn't help on images — this is empirical.

== STEP 5 (Mon PM, 30 min): Commit + push ==
git add results/utkface_results.json figures/fig10_utkface_curves.* \
        docs/UTKFACE_INVERSION_HYPOTHESIS.md logs/utkface_extend.log
git commit -m "Week 3: extend UTKFace to α=0.3/0.4 + inversion hypothesis doc"
git push origin main

RULES:
- DO NOT touch experiments/run_utkface.py (Agent A is editing it).
- DO NOT touch src/* (only USE existing code).
- COORDINATE GPU TIME with Agent A — run sequentially, not simultaneously.

REPORT BACK with file sizes and one-paragraph hypothesis summary.
```

---

# 🟢 ORCHESTRATOR (Claude) — My Mon Tasks

While agents work, I will:

1. **Mon PM (2h):** Consolidate the 27 sprawling docs in `docs/` into a single
   clean `TUESDAY_JUN2_AGENDA.md` that contains:
   - Recap of Tasks 1 & 2
   - Headline numbers from tabular + UTKFace
   - Inversion finding
   - Hypothesis summary
   - Proposed next-week tasks

2. **Mon PM (30 min):** Archive old docs into `docs/_archive/` to declutter.

3. **Tue AM (2h):** Help build the 5-min slide deck — concrete bullets per slide.

4. **Tue 2:30 PM:** Final dry-run of the meeting.

---

# 🎤 Tuesday June 2 — Meeting Deck Structure

**Slide 1 (30s) — Status recap**
- Tasks 1 and 2 both shipped on tabular + UTKFace
- Plus: extended UTKFace with FairnessTargetedPGD (going beyond what was asked)

**Slide 2 (60s) — Tabular results (Adult/Credit/LSAC × 3 attacks)**
- fig8 attack-defense matrix
- One headline: DRO recovers 96-97% DP on Credit/LSAC under IF attack at α=0.3

**Slide 3 (90s) — UTKFace results**
- fig10 baseline + fig11 fairness-PGD curves
- Headline: DRO behavior INVERTS on image features

**Slide 4 (90s) — Why does DRO invert? (3 hypotheses)**
- H1: pre-fair features
- H2: feature-space attacks don't transfer
- H3: continuous-embedding noise amplification

**Slide 5 (30s) — Next week**
- Trace λ_DP on UTKFace
- Try λ_max=0.5
- Image-space PGD

---

# ⏱️ Critical Path Timeline

```
Sun 6pm: Spawn Agent A → wire run_utkface.py + smoke test
Sun 7pm: Agent A launches full run on GPU (overnight)
Sun 8pm: Spawn Agent B → wait for Agent A's commit, then write hypothesis draft
Mon 8am: Agent A's experiment finishes; Agent B starts GPU extension run
Mon 10am: Agent B's extension finishes; both agents start writeups
Mon 1pm: Both agents commit results + writeups
Mon 2pm: Orchestrator consolidates docs
Mon 5pm: All artifacts ready
Tue 9am: Build slides
Tue 2:30pm: Dry run
Tue 3pm: Meeting
```

---

# 🚨 Drop-Scope Triggers

- **GPU dies** → fall back to existing 15 UTKFace + 270 tabular. Report as "Week 4 work pending GPU".
- **Agent A smoke fails** → ship only baseline UTKFace extension (Agent B's Step 2 still works without Agent A).
- **Mon evening unfinished** → cut combined-attack from UTKFace, ship DP + IF only.
- **Both agents stuck** → Tuesday meeting becomes "what's done + plan for next week" — madam values honesty over overreach.

---

# 📞 Reporting Cadence

Each agent reports at:
- End of Step 1
- End of Step 3 (after smoke / full run launch)
- End of Step 5 (final commit)

You forward each report to me, I'll review.

---

**Status:** Plan locked. Spawn agents in this order: Agent A first, then Agent B 15 min later.
