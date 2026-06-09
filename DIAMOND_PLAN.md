# DIAMOND-QUALITY PLAN — Diagnose, Fix, Re-run Everything
**Today:** Mon June 9, 2026 (post-meeting day) · **Deadline:** Mon June 16, 3 PM
**Net working time:** 7 days · **Standard:** every claim sourced to JSON, no oracle leaks, n≥10 seeds, no ad-hoc tuning

---

## 🎯 The Goal

Walk into next Monday's meeting with:

1. **One number madam wants:** "Naive DP under our adversarial attack is K× larger than under random noise" — a single, defensible comparison.
2. **Honest DRO verdict:** With oracle leaks removed and lambda_max consistent, does DRO actually help? Yes/No, with n≥10 seed Wilcoxon.
3. **UTKFace re-run with the fixed attack.** Real numbers, not stale baselines.
4. **A single source-of-truth status doc.** Every claim links to a JSON row.
5. **Zero discrepancies** between code/data/figures/docs.

If we can't show DRO helps, **that's the finding.** Madam said it herself: "If DP is hard to attack, that's itself a major finding." We do not fake numbers to look good.

---

## 🚨 The 18 Issues To Fix (from yesterday's audit)

| # | Category | Issue | Action |
|---|---|---|---|
| A1 | Attack | pgd_steps=5 in run script (designed=20) | Fix run script |
| A2 | Attack | Feature PGD uses BCE loss (not DP) when target='dp' | Re-design PGD loss for each target_metric |
| A3 | Attack | Surrogate is LR (model is MLP) | Use trained MLP as oracle for PGD AFTER first naive training pass, OR use bigger surrogate |
| A4 | Attack | epsilon ball uniform (not per-feature std) | Add std scaling option |
| A5 | Attack | Surrogate trained on y_orig but PGD targets y_c | Train surrogate on `(X, y_c)` (post-flip) OR document |
| B1 | Design | K_inner=5 (mandatory=10) | Hard-code k_inner=10 in run script |
| B2 | Design | lambda_max=0.5 hack for Adult | REMOVE — use 1.5 everywhere |
| B3 | Design | lambda_warmstart=0.01 (not in v1.0) | Revert to 0.0 |
| B4 | Design | corruption_rates leak — oracle info to DRO | REMOVE — DRO uses only α |
| C1 | Stats | n=3 → Wilcoxon p<0.05 impossible | Run n=10 seeds |
| C2 | Stats | "Significance" claims with n=3 | Re-do all p-values after C1 |
| D1 | Data | No α=0.0 baseline | Add α=0.0 runs |
| D2 | Data | No random vs adversarial comparison | Add corruption='random' arm |
| D3 | Data | UTKFace not re-run | Re-run all UTKFace with fixed attack |
| E1 | Docs | STATUS.md 270 vs reality 216 | Auto-generated from JSON |
| E2 | Docs | BUGFIX_SUMMARY overclaims | Rewrite honestly |
| E3 | Docs | STATUS.md says "RUNNING" stale | Auto-update |
| F1 | Git | Old 97.5% Credit win overwritten | Document the version history |

---

## 📅 7-Day Schedule

| Day | Date | Owner | Phase | Output |
|---|---|---|---|---|
| 1 | Mon Jun 9 (today) | Agent A | Code fixes (A1, B1-B4, A2) | Clean `adversarial.py`, `run_fairness_pgd.py`, `dro_fair.py` |
| 2 | Tue Jun 10 | Agent A | Smoke validation gate | Numbers proving each fix |
| 3 | Wed Jun 11 | Agent B | Tabular full run launch | All 1800 cells running |
| 4 | Thu Jun 12 | Agent B | UTKFace full run on GPU | UTKFace results JSON populated |
| 5 | Fri Jun 13 | Agent A + B | Analysis + figures + Wilcoxon | CSVs + 3 final figures |
| 6 | Sat Jun 14 | Orchestrator | QA cross-check | Every claim → JSON row mapping |
| 7 | Sun Jun 15 | Orchestrator | Final report + slides | `WEEK4_REPORT.md`, 5-slide deck |
| 8 | Mon Jun 16 | You | Meeting at 3 PM | — |

---

# 🅰️ AGENT A BRIEF — DAY 1 (Mon June 9): Code Fixes

**Copy-paste verbatim. DO NOT add features. DO NOT run experiments. ONLY fix the listed bugs.**

```
You are AGENT A for /Users/srujansai/Desktop/DRO-FairML.
Today is Mon June 9. Meeting Mon June 16 at 3 PM. We have one week.

TODAY'S JOB: surgically fix bugs in 3 files. No experiments. No new features.
Every change below is targeted at a specific bug from yesterday's audit.

═══════════════════════════════════════════════════════════
FIX 1: experiments/run_fairness_pgd.py — Bugs A1, B1, B2, B3, B4
═══════════════════════════════════════════════════════════

(a) pgd_steps and k_inner (Bugs A1, B1)
    Find the section that sets smoke_k_inner and smoke_pgd_steps in main().
    Replace ALL the "smoke" prefixed variables with non-smoke names.
    In the NON-smoke branch (the else), set:
        k_inner = 10
        pgd_steps = 20
        epochs = 60
    These are paper/spec mandatory values.

(b) Remove lambda_max hack for Adult (Bug B2)
    Delete the entire function get_lambda_max.
    In the DRO branch, use lambda_max=1.5 always.
    Add comment: "lambda_max=1.5 per v1.0 paper. Adult collapse documented as finding."

(c) Remove lambda_warmstart override (Bug B3)
    In the DroFairTrainer(...) call, REMOVE the `lambda_warmstart=0.01` argument.
    Let the trainer use its v1.0 default of 0.0.

(d) Remove corruption_rates oracle leak (Bug B4)
    Delete lines 67-75 (the loop computing per-group corruption_rates).
    In the DroFairTrainer(...) call, REMOVE the `corruption_rates=corruption_rates`
    argument.
    DRO must use only α — same info as v1.0.

(e) Add corruption mode arg (for Bug D2)
    Add CLI flag: --corruption {adversarial, random, clean}  (default: adversarial)
    When 'clean': skip the attack_obj.corrupt() call. X_train_att = X_train.
    When 'random': use RandomCorruptor from src/corruption/adversarial.py.
    When 'adversarial': use FairnessTargetedPGD (current behavior).
    Record this in the output JSON as 'corruption' field.

(f) Add α=0.0 to default sweep (Bug D1)
    Change default alphas from [0.1,0.2,0.3,0.4] to [0.0, 0.1, 0.2, 0.3, 0.4].
    When alpha==0.0, skip the corruption call entirely (X_train_att=X_train).

═══════════════════════════════════════════════════════════
FIX 2: src/training/dro_fair.py — Bugs B3, B4
═══════════════════════════════════════════════════════════

(a) Default lambda_warmstart=0.0 (Bug B3)
    In __init__, ensure default is 0.0.

(b) Default corruption_rates=None (Bug B4) — keep parameter but document
    DO NOT delete the parameter (would break unit tests).
    But default to None, and in _compute_radii use the v1.0 formula
    `rho_dp_j = alpha / ((1-alpha)*pi_clean[j] + alpha)` when corruption_rates is None.
    Add a docstring warning that passing corruption_rates is "ORACLE INFO and
    should only be used for ablation studies."

═══════════════════════════════════════════════════════════
FIX 3: src/corruption/adversarial.py — Bug A2 (PGD targets correct loss)
═══════════════════════════════════════════════════════════

In _attack_features_pgd, currently the PGD always uses BCE loss.
This makes "DP-targeted PGD" actually a classification-targeted PGD.

CHANGE: make the loss target the same metric as self.target_metric.

Pseudocode for the loss inside the PGD loop:
    if self.target_metric == 'dp':
        # We want to MAXIMIZE the DP gap of the surrogate's predictions.
        # The surrogate is a classifier; use its sigmoid output as soft prediction.
        probs = torch.sigmoid(model(X_batch))  # shape (k,)
        a_batch = torch.tensor(a[corrupt_idx], dtype=torch.float32, device=device)
        # weighted group rates from CURRENT batch
        n0 = (a_batch == 0).float().sum().clamp(min=1)
        n1 = (a_batch == 1).float().sum().clamp(min=1)
        p0 = (probs * (a_batch == 0).float()).sum() / n0
        p1 = (probs * (a_batch == 1).float()).sum() / n1
        loss = -torch.abs(p0 - p1)   # NEGATIVE because PGD does gradient ASCENT
        # (or just maximize abs(p0-p1) — equivalent via sign flip)
    elif self.target_metric == 'if':
        # Similar but maximizing pairwise prediction disagreement among k-NN
        # (use precomputed_neighbors if available)
        # For simplicity, fall back to BCE for IF (or implement IF surrogate gradient)
        loss = F.binary_cross_entropy_with_logits(model(X_batch), y_batch)
    else:  # combined or default
        loss = F.binary_cross_entropy_with_logits(model(X_batch), y_batch)

    grad = torch.autograd.grad(loss, X_batch)[0]

Add corrupt_idx to the function signature so 'a' can be accessed.
Update the caller in corrupt() to pass a, corrupt_idx.

═══════════════════════════════════════════════════════════
TESTS TO ADD
═══════════════════════════════════════════════════════════

tests/test_attack_fixes.py — pytest-style:

def test_pgd_steps_actually_20_in_full_run():
    # Verify default in FairnessTargetedPGD is 20
    a = FairnessTargetedPGD(alpha=0.1)
    assert a.pgd_steps == 20

def test_run_fairness_pgd_pgd_steps_20():
    # Read the source of run_fairness_pgd.py and assert it sets pgd_steps=20
    import inspect
    from experiments import run_fairness_pgd
    src = inspect.getsource(run_fairness_pgd)
    assert 'pgd_steps = 20' in src or 'pgd_steps=20' in src

def test_dro_no_corruption_rates_by_default():
    # Verify default DroFairTrainer doesn't take corruption_rates
    trainer = DroFairTrainer(MLPClassifier(10), alpha=0.2)
    assert trainer.corruption_rates is None

def test_dp_targeted_pgd_increases_dp():
    # Synthetic data, run DP-targeted attack, check naive DP increases
    np.random.seed(0)
    X = np.random.randn(1000, 5)
    y = (X[:,0] > 0).astype(int)
    a = (X[:,1] > 0).astype(int)
    # Baseline DP on a fit LR
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(X, y)
    dp_clean = abs(lr.predict(X[a==0]).mean() - lr.predict(X[a==1]).mean())
    # Run attack
    att = FairnessTargetedPGD(alpha=0.2, target_metric='dp', random_state=0)
    Xc, yc, ac, _ = att.corrupt(X, y, a)
    lr2 = LogisticRegression().fit(Xc, yc)
    dp_atk = abs(lr2.predict(X[a==0]).mean() - lr2.predict(X[a==1]).mean())
    assert dp_atk > dp_clean * 1.5, f"Attack failed to increase DP: {dp_clean} -> {dp_atk}"

ALL existing tests must pass + new tests must pass.

═══════════════════════════════════════════════════════════
END-OF-DAY DELIVERABLE
═══════════════════════════════════════════════════════════

Commit message: "Phase 1: surgical fixes — pgd_steps=20, K_inner=10, remove
oracle leak, DP-targeted PGD, lambda_max=1.5"

Push to main. NO EXPERIMENTS today. Just code fixes and unit tests.

REPORT BACK with:
1. pytest output (all green)
2. git diff stat
3. One paragraph describing each fix and why it was needed.
```

---

# 🅰️ AGENT A BRIEF — DAY 2 (Tue June 10): Smoke Validation Gate

**Paste after Day 1 commit lands and you've reviewed it.**

```
You are AGENT A. Day 1 fixes are committed. Today: prove each fix actually works
before launching the full re-run. This is the validation gate.

DELIVERABLE: experiments/smoke_validation.py + results/smoke_validation.json

For each (dataset, alpha, seed in [0,1,2]):
  - Adult only, alpha in [0.0, 0.1, 0.2], 3 seeds
  - Measure 6 numbers per cell:
      DP_clean      → train naive on clean data
      DP_random     → train naive on random-corrupted data (new RandomCorruptor)
      DP_adversarial_dp → train naive on FairnessTargetedPGD(target='dp')
      DP_adversarial_if → train naive on FairnessTargetedPGD(target='if')
      DP_adversarial_combined → train naive on FairnessTargetedPGD(target='combined')
      Also: DRO DP under each (to make sure DRO trains without crashing)

  Save all to results/smoke_validation.json.

After running, print summary:

  Adult α=0.0: DP_clean=X.XXX (n=3 ± SE)
  Adult α=0.1:
    random   DP=X.XXX
    adv-dp   DP=X.XXX  (ratio over random: K.KKx)
    adv-if   DP=X.XXX  (ratio: K.KKx)
    adv-comb DP=X.XXX  (ratio: K.KKx)
  Adult α=0.2: ...

VALIDATION CRITERIA (must all hold):
  - DP_clean ≈ DP_random (random shouldn't hurt much)
  - DP_adversarial_dp > DP_random by at least 2× on Adult α=0.2
  - DP_adversarial_dp > DP_clean by at least 3× on Adult α=0.2
  - DRO trains without crashes
  - Re-running with same seed produces IDENTICAL DP (reproducibility check)

If validation passes → escalate to orchestrator, ready for full run.
If any criterion fails → STOP, report which one, do not proceed.

REPRODUCIBILITY CHECK:
  Run experiments/smoke_validation.py twice with same seeds.
  Diff results/smoke_validation.json runs.
  All numbers must be identical to 6 decimal places.

Commit message: "Phase 2: smoke validation — Adult adv-PGD K× stronger than random"

REPORT BACK with the summary table and the validation verdict.
```

---

# 🅱️ AGENT B BRIEF — DAY 3 (Wed June 11): Full Tabular Run

**Paste after Day 2 validation passes.**

```
You are AGENT B for /Users/srujansai/Desktop/DRO-FairML.
Day 2 smoke validated. Today: launch the full tabular re-run.

DELIVERABLE: results/fairness_pgd_v2_results.json with all 1800 cells.

EXPERIMENT GRID:
  3 datasets × 5 alphas × 10 seeds × 3 corruption types × 2 methods = 900
  Plus: 3 datasets × 5 alphas × 10 seeds × 'clean' (α=0 baseline only) × 2 methods = 30
  Plus: 3 datasets × 4 alphas (0.1-0.4) × 10 seeds × 1 random × 2 methods = 240
  ...

OK simpler: build a SINGLE grid:
  datasets = [adult, credit, lsac]
  alphas = [0.0, 0.1, 0.2, 0.3, 0.4]    # 0.0 = no corruption regardless of mode
  corruptions = [clean, random, adv_dp, adv_if, adv_combined]
  methods = [naive, dro]
  seeds = [0..9]

  Cells: 3 × 5 × 5 × 2 × 10 = 1500

  But: when alpha=0.0 and corruption is anything other than 'clean', skip
       (same as 'clean' anyway).
       Effective: 3 × (1 + 4×5) × 2 × 10 = 3 × 21 × 2 × 10 = 1260

EXECUTION:
  Launch in background with incremental save:
    nohup python3 experiments/run_full_v2.py \
      --datasets adult credit lsac \
      --alphas 0.0 0.1 0.2 0.3 0.4 \
      --corruptions clean random adv_dp adv_if adv_combined \
      --methods naive dro \
      --n_seeds 10 \
      > logs/full_v2.log 2>&1 &

  Save to results/fairness_pgd_v2_results.json incrementally.
  Each cell schema: {dataset, alpha, seed, corruption, method, acc, dp, if, runtime}

  Use exactly the fixed code from Phase 1.
  K_inner=10, pgd_steps=20, lambda_max=1.5 always (no Adult hack).
  No corruption_rates passed.

  Expected runtime: ~30s/cell on CPU = ~10 hours total.
  Use logging.basicConfig(level=INFO) for visibility.

QUALITY GATES:
  - Every cell must complete or be marked FAILED
  - No silent crashes
  - acc_clean values are in [0, 1]
  - dp/if values are non-negative
  - Reproducibility: re-run cell (Adult, α=0.2, seed=0, adv_dp, naive) must give
    identical numbers to ±1e-6

REPORT BACK:
  - Total cells completed / target
  - Any failures
  - Wall-clock time
  - Confirm reproducibility with 1 cell
```

---

# 🅱️ AGENT B BRIEF — DAY 4 (Thu June 12): UTKFace Re-run on GPU

```
You are AGENT B. Tabular run is going. Today: UTKFace re-run with FIXED attack.

PREREQUISITE: GPU server access (flair2.iitgn.ac.in). If still blocked, escalate.

DELIVERABLE: results/utkface_v2_results.json

GRID:
  1 dataset × 4 alphas (0.0, 0.1, 0.2, 0.3) × 5 seeds × 5 corruptions × 2 methods
  = 200 cells

  At ~3 min/cell on GPU = ~10 hours.

USE FIXED ATTACK:
  Edit experiments/run_utkface.py to use the SAME corruption logic as run_fairness_pgd.py
  after Phase 1 fixes. Specifically:
    - K_inner=10, pgd_steps=20, lambda_max=1.5
    - No corruption_rates leak
    - Add --corruption {clean, random, adv_dp, adv_if, adv_combined} flag

  IMPORTANT: For images, the surrogate model question is open.
  Use the SAME LogisticRegression surrogate trained on cached ResNet18 features.
  This is what tabular uses; consistency matters.

LAUNCH:
  ssh into GPU server, run inside tmux:
    nohup python3 experiments/run_utkface.py \
      --alphas 0.0 0.1 0.2 0.3 \
      --corruptions clean random adv_dp adv_if adv_combined \
      --methods naive dro \
      --n_seeds 5 \
      > logs/utkface_v2.log 2>&1 &

QUALITY GATES (same as Day 3 plus):
  - GPU memory usage < 90%
  - No CUDA OOM errors
  - Image features match between local cache and server cache (md5 check)

REPORT BACK with completion stats and any failures.
```

---

# 🟢 ORCHESTRATOR BRIEF — DAY 5-6: Analysis + QA

**My job, Fri-Sat:**

```
== DAY 5 (Fri Jun 13): Analysis + Figures + Wilcoxon ==

Build experiments/analyze_v2.py producing:

  results/v2_summary.csv:
    columns: dataset, corruption, alpha, method, n_seeds, acc_mean, acc_se,
             dp_mean, dp_se, if_mean, if_se

  results/v2_wilcoxon.csv:
    For each (dataset, corruption, alpha): Wilcoxon test naive_dp vs dro_dp
    columns: dataset, corruption, alpha, n, dp_naive_mean, dp_dro_mean,
             abs_diff, dp_pvalue, dp_significant (p<0.05)

  results/v2_attack_validity.csv:
    For each (dataset, alpha): compare DP_naive under each corruption
    columns: dataset, alpha, n,
             DP_naive_clean, DP_naive_random, DP_naive_adv_dp,
             DP_naive_adv_if, DP_naive_adv_combined,
             ratio_adv_dp_vs_random, ratio_adv_if_vs_random

  figures/fig_attack_validity.pdf:
    Absolute DP values (not %).
    3 subplots (one per dataset).
    Each subplot: x=alpha, grouped bars per corruption mode on Naive only.
    This is THE figure madam asked for.

  figures/fig_dro_defense.pdf:
    3 subplots (one per dataset).
    Each subplot: x=alpha, lines for (Naive, DRO) under adv_dp attack.
    Error bars (caps, no shading).

  figures/fig_random_vs_adv_heatmap.pdf:
    3×5 heatmap (datasets × alphas).
    Cell value: ratio (adv_dp / random) of Naive DP increases.
    Color: green if ratio > 2, red if ≤ 1.

== DAY 6 (Sat Jun 14): QA Cross-Check ==

Build experiments/qa_cross_check.py — verifies every claim:

  For every number that will appear in WEEK4_REPORT.md or slides:
    - Find the JSON row it sources from
    - Recompute the number from raw data
    - Assert it matches to ±1e-4

  Outputs results/qa_cross_check_report.txt listing each claim and pass/fail.

  Also runs:
    - All tests must pass: pytest tests/ -v
    - Reproducibility check: 5 random cells re-run from raw data → identical
    - No NaN / Inf anywhere
    - No orphan claims (number in report with no JSON source)

QA report must show 100% pass before proceeding to Day 7.
```

---

# 🟢 ORCHESTRATOR BRIEF — DAY 7: Final Report + Slides

**My job, Sun Jun 15:**

```
DELIVERABLE: WEEK4_REPORT.md + slides/week4.html

STRUCTURE (matches what madam will ask):

  # Week 4 — Diamond Re-run

  ## What madam asked
  > "Check the adversarial attack on DP and improve it. Then, redo all the experiments."

  ## The 18 issues we found and fixed (1 paragraph)

  ## ATTACK VALIDITY (the one number madam wants)
  Table: DP_naive under each corruption type, per (dataset, alpha).
  Headline: "On Adult α=0.2, adv_dp attack increases Naive DP by K.K× over random noise."

  ## DRO defense under STRONG attack
  Table: DP_naive vs DP_dro under adv_dp at each (dataset, alpha).
  Wilcoxon p-values from n=10 seeds.
  Honest verdict per dataset.

  ## UTKFace
  Same structure on UTKFace.

  ## Honest limitations
  - PGD feature attack: now targets the correct metric, but surrogate is still LR
  - Adult collapse documented (consistent with v1.0)
  - 10 seeds: significance achievable but tight (min p ≈ 0.001)
  - GPU server access: still working as of June 12

  ## Reproducibility
  - Tag v1.1
  - All hyperparameters in run_full_v2.py
  - Seeds 0..9 fixed
  - SHA256 of results/v2_summary.csv: <hash>

5-slide deck:
  S1: Status + 18 fixes
  S2: Attack validity figure (THE money slide)
  S3: DRO defense table with p-values
  S4: UTKFace
  S5: Limitations + next week
```

---

## ✅ Diamond Quality Criteria — Each Phase Must Pass

Each agent reports back with a CHECKLIST. Phase doesn't advance until all green:

### Phase 1 (Code fixes)
- [ ] All 9 specific code changes done
- [ ] Each change has a corresponding test
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] git diff shows ONLY the targeted changes (no scope creep)
- [ ] Committed and pushed

### Phase 2 (Smoke validation)
- [ ] Adult α=0.2 adv_dp is ≥ 2× DP_random
- [ ] Adult α=0.2 adv_dp is ≥ 3× DP_clean
- [ ] DRO completes without crash on all alphas
- [ ] Reproducibility: re-run = identical to 6 decimals
- [ ] Committed

### Phase 3 (Tabular full run)
- [ ] 1260 cells expected
- [ ] All cells complete OR explicitly marked FAILED with reason
- [ ] No NaN/Inf anywhere
- [ ] Wall-clock matches estimate ± 30%
- [ ] Committed

### Phase 4 (UTKFace)
- [ ] 200 cells expected
- [ ] GPU usage logged
- [ ] All complete OR marked FAILED

### Phase 5 (Analysis)
- [ ] All CSVs produced
- [ ] 3 figures rendered (300dpi, both pdf+png)
- [ ] Wilcoxon n=10, real p-values reported

### Phase 6 (QA)
- [ ] Every report number sourced to JSON row
- [ ] Recomputation matches ±1e-4
- [ ] Reproducibility on 5 random cells
- [ ] Zero orphan claims

### Phase 7 (Final report)
- [ ] Numbers cross-checked against Phase 6 report
- [ ] Limitations explicit and honest
- [ ] Slides match doc exactly

---

## 🚨 Drop-Scope Triggers (cut early, don't pile up)

- **Day 2 smoke fails any validity criterion** → STOP. Diagnose. Don't launch Phase 3.
- **Day 3 estimate >18h** → Drop to 5 seeds, document.
- **Day 4 GPU dies** → UTKFace becomes "pipeline pending GPU"; do not fake results.
- **Day 6 QA finds >5 discrepancies** → Halt. Fix before report.

---

## 📞 Daily Standup (3 lines per agent, 5 PM)

```
[YYYY-MM-DD] AGENT X
DID: <one sentence>
NEXT: <one sentence>
BLOCKER: <one sentence or "none">
```

Post to a single file: `daily_standup.md`. Append-only.

---

## 🎯 The Bottom Line

**A diamond result is not "DRO wins everywhere."**
A diamond result is:

1. The attack is genuinely strong (provable)
2. The DRO comparison is fair (no oracle, no per-dataset hacks)
3. The numbers reproduce exactly
4. Every claim in the report has a JSON source
5. Honest limitations are listed

If DRO doesn't help under the fair comparison, **that is the finding**.
Madam will respect honesty + rigor more than tuned-up numbers.

---

## 🚀 RIGHT NOW (Mon Jun 9 evening)

1. Read this entire doc once.
2. Spawn Agent A with the Day 1 brief above.
3. By tomorrow morning, Day 1 commit lands.
4. Tomorrow noon, spawn Agent A with Day 2 brief.
5. Wednesday morning, smoke validates → spawn Agent B with Day 3.

Path is locked. Now execute.
