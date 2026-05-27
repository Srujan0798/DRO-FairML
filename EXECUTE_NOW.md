# EXECUTE_NOW — Week 2 Catch-up Plan
**Today:** May 27, 2026 · **Next meeting:** Tue June 2, 2026 (6 days)
**Status:** Day 1 complete. Agent B committed. Agent A smoke test pending. Full runs pending.

---

## ✅ What's DONE (May 27)

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| CNNClassifier | Agent B | **COMMITTED** `a31d43f` | ResNet18 backbone, 3 FC layers |
| ImagePGD | Agent B | **COMMITTED** `a31d43f` | epsilon=4/255, steps=10 |
| run_utkface.py | Agent B | **COMMITTED** `a31d43f` | Synthetic fallback when no UTKFace data |
| FairnessTargetedPGD (DP only) | Agent A | UNCOMMITTED | `scripts/test_fairness_pgd.py` validates dp mode |
| load_utkface() | Agent A | UNCOMMITTED | Placeholder in `src/data/datasets.py` |
| GPU server | Agent B | BLOCKED | `gpu-server` hostname not resolvable |

---

## 🧨 What MUST be true by Tuesday 4 PM

1. **Task 1 (PGD-Fairness on Adult):** Done with DP-only, IF-only, joint modes; results table; mini-writeup.
2. **Task 2 (UTKFace):** At minimum, Naive-FAIR baseline running on GPU server with 3 seeds. DRO-FAIR if time permits.
3. **GitHub:** Tagged `v1.1`, README updated with Week 2 progress section.
4. **Slides:** 5-minute deck ready (Sunday night latest).

---

## 🔀 Agent Assignments

### 🅰️ Agent A — Fairness-PGD (tabular, runs locally)

**Total estimated work: 8-10 hours over Tue-Thu.**

Copy-paste this into Agent A:

```
You are AGENT A for the DRO-FAIR project at /Users/srujansai/Desktop/DRO-FairML.
Today is May 27. You have until end of Thursday May 29 to finish everything below.

CURRENT STATE:
- src/corruption/adversarial.py contains a FairnessTargetedPGD class — UNCOMMITTED.
  - target_metric='dp' works (validated by scripts/test_fairness_pgd.py)
  - target_metric='if' is BROKEN — compute_if_gradient returns zeros
  - target_metric='combined' is also broken because of the IF placeholder
- No experiment driver exists.
- No proper tests in tests/ folder (only scripts/test_fairness_pgd.py exists).

YOUR DELIVERABLES (in this exact order):

== STEP 1: Fix IF gradient (1h) ==
In src/corruption/adversarial.py, replace compute_if_gradient with:
- Signature: compute_if_gradient(self, y, a, X=None, k=5)
- Algorithm: For each protected group, build sklearn k-NN on X.
  For each sample i, count agreeing vs disagreeing neighbors of same group.
  grad[i] = (agree - disagree) / k_eff
- Update compute_fairness_gradient to pass X through.
- Update _attack_labels_fairness to accept and pass X.
- Update corrupt() to pass X through to _attack_labels_fairness.

== STEP 2: Write proper tests (1h) ==
Create tests/test_fairness_pgd.py with pytest-style tests:
- test_dp_attack_increases_dp: synthetic 1000-sample dataset, run dp-attack at α=0.2,
  assert dp_after > dp_before * 1.2
- test_if_attack_increases_if: same with if-attack
- test_combined_attack: assert both metrics increase
- test_alpha_budget: assert sum(corrupt_mask) == int(α * n)
- test_minority_targeted: with coordinated=True and α=0.2, assert ≥60% of corruptions hit minority group
Run: pytest tests/test_fairness_pgd.py -v → all 5 must pass.

== STEP 3: Write experiment driver (2h) ==
Create experiments/run_fairness_pgd.py.
- argparse flags: --datasets (default: adult credit lsac), --attacks (default: dp if combined),
  --methods (default: naive dro), --alphas (default: 0.1 0.2 0.3), --n_seeds (default: 10),
  --smoke (sets n_seeds=1, alphas=[0.2], datasets=[adult])
- For each (dataset, alpha, seed, attack, method):
    train method on clean training data
    apply attack to TRAINING data (this is the corruption the model is trained against;
    we're testing whether DRO's defense holds when attacker targets fairness)
    retrain method on attacked data
    evaluate on CLEAN test set
    record: dataset, alpha, seed, attack, method, acc, dp, if
- Save to results/fairness_pgd_results.json as a list of dicts.
- Print running progress: "[12/360] adult α=0.2 seed=3 attack=if method=dro: dp=0.041..."

== STEP 4: Smoke run (5 min) ==
Run: python3 experiments/run_fairness_pgd.py --smoke
- Must complete in <5 min
- Must produce a valid JSON file with 6 rows: 3 attacks × 2 methods × 1 dataset × 1 seed
- If smoke passes, COMMIT all changes with message:
  "Add FairnessTargetedPGD with DP/IF/combined attacks + experiment driver"

== STEP 5: Full run (overnight Tue→Wed) ==
Launch in background:
  nohup python3 experiments/run_fairness_pgd.py > logs/full_fpgd.log 2>&1 &
Expected: 3 datasets × 3 attacks × 2 methods × 3 alphas × 10 seeds = 540 trainings.
At ~30s/training for Adult/Credit/LSAC = ~4.5 hours. Should finish by Wed morning.

== STEP 6: Aggregation + figures (2h) ==
Create experiments/analyze_fairness_pgd.py that:
- Loads results/fairness_pgd_results.json
- For each (dataset, attack), computes mean ± SE of dp and if across seeds
- For each (dataset, attack), runs Wilcoxon paired test: Naive DP > DRO DP?
- Outputs:
  - results/fairness_pgd_summary.csv
  - figures/fig8_attack_defense_matrix.pdf — heatmap: rows=attacks, cols=datasets,
    cells=DP reduction% of DRO over Naive
  - figures/fig9_fairness_pgd_curves.pdf — like fig1 but with attack mode as the line

== STEP 7: Writeup (1h) ==
Create docs/FAIRNESS_PGD_RESULTS.md (~2 pages):
- Setup (what each attack does)
- Table: mean DP under each attack, both methods
- Key findings (3-5 bullets)
- Honest limitations (e.g., Adult collapse pattern persists under IF attack)

== STEP 8: Commit + push ==
Final commit message: "Week 2 Task 1: Fairness-targeted PGD experiments complete"

CRITICAL: do NOT touch src/data/, src/models/, or experiments/run_utkface.py
— those are Agent B's territory. Coordinate with orchestrator only.
```

---

### 🅱️ Agent B — UTKFace (image, GPU server)

**Status May 27 EOD:** Steps 0-5 DONE. Smoke test PASSED. Committed as `a31d43f`.

Remaining work (Thu-Fri):

```
You are AGENT B for the DRO-FAIR project at /Users/srujansai/Desktop/DRO-FairML.
You have already committed: CNNClassifier, ImagePGD, run_utkface.py (all pass smoke test).
GPU server is BLOCKED — hostname not resolvable. Use CPU-only with synthetic fallback.

REMAINING DELIVERABLES:

== STEP 6: GPU server access (30 min) ==
- Try: ssh srujan.sai@<gpu-server>
- If working: proceed to Step 7.
- If blocked: skip GPU run, proceed with CPU-only synthetic experiment.

== STEP 7: Full UTKFace run (overnight Wed→Thu) ==
If GPU available:
  nohup python3 experiments/run_utkface.py --alphas 0.0 0.1 0.2 0.3 --n_seeds 3 > logs/full_utkface.log 2>&1 &
Else (CPU/synthetic):
  python3 experiments/run_utkface.py --alphas 0.0 0.2 --n_seeds 1 --smoke
  (small run to confirm pipeline works; full run when GPU secured)

== STEP 8: Aggregation + figures (2h) ==
Create experiments/analyze_utkface.py:
- Wilcoxon DP comparison Naive vs DRO at each α
- figures/fig10_utkface_curves.pdf — DP/IF curves vs α
- results/utkface_summary.csv

== STEP 9: Writeup (1h) ==
docs/UTKFACE_RESULTS.md (~2 pages):
- Setup (dataset, model, features)
- Numbers
- Whether DRO defends on a much larger / image dataset
- Limitations

== STEP 10: Commit + push ==
Final commit: "Week 2 Task 2: UTKFace experiments complete"

CRITICAL: do NOT touch src/corruption/adversarial.py, experiments/run_fairness_pgd.py,
or anything in Agent A's territory.
```

---

### 🟢 Orchestrator (Claude) — What I Track

I will NOT touch code. I will:
1. Review commits as they come in.
2. Reconcile results if Agent A and B step on each other.
3. Help write the meeting deck on Sunday/Monday.
4. Flag if either agent goes off-spec.

---

## 📆 6-Day Timeline

| Day | Date | Agent A (PGD) | Agent B (UTKFace) | Status |
|-----|------|---------------|------------------|--------|
| Wed | May 27 | Steps 1-4 done (smoke pending) | Steps 0-5 done, COMMITTED | ✅ Day 1 |
| Thu | May 28 | Steps 5-6 (full run + analysis) | Steps 7-8 (analysis) | 🔄 |
| Fri | May 29 | Step 7 (writeup) → done | Step 9-10 (full run + commit) | 🔄 |
| Sat | May 30 | (idle/help B) | Step 10 → done | 🔄 |
| Sun | May 31 | — | — | 📄 Update report.tex |
| Mon | Jun 1 | — | — | 🟢 5-slide deck, dry-run |
| Tue | Jun 2 | — | — | **4 PM Meeting** |

---

## 🚨 If Time Runs Out — Cut In This Order

1. **Drop UTKFace seeds** from 3 → 1 (still produces a result, just no error bars)
2. **Drop UTKFace DRO-FAIR run** — show only Naive-FAIR baseline, frame DRO as "next week"
3. **Drop joint-attack from PGD experiments** — keep just DP-only and IF-only
4. **Drop full Adult+Credit+LSAC PGD coverage** — show only Adult (the failure dataset)

The professor specifically asked about Adult — so as long as Adult-PGD results exist, you have something to present.

---

## 📞 Standup Format — End of Each Day

Reply to me in 3 lines per agent:

```
A: [what done today] / [doing tomorrow] / [blockers]
B: [what done today] / [doing tomorrow] / [blockers]
```

I'll flag anything off-track immediately.

---

## 📋 Current Git Status

```
COMMITTED (Agent B — May 27):
  a31d43f Add UTKFace pipeline: CNN classifier, ImagePGD attack, and experiment runner
    - src/models/cnn_classifier.py
    - src/corruption/image_pgd.py
    - experiments/run_utkface.py

UNCOMMITTED (Agent A):
  - src/corruption/adversarial.py (FairnessTargetedPGD with broken IF gradient)
  - src/data/datasets.py (load_utkface placeholder)

UNTRACKED:
  - scripts/test_fairness_pgd.py
  - scripts/extract_utkface_features.py
  - docs/FAIRNESS_PGD_DESIGN.md
  - docs/UTKFACE_PIPELINE.md
```

---

## ▶ NEXT ACTION

1. **Agent A:** Run `python3 scripts/test_fairness_pgd.py` to verify DP mode still works, then fix IF gradient and run smoke test
2. **Agent B:** Confirm `gpu-server` hostname with sysadmin; if unresolved by Thu, proceed CPU-only
3. **Orchestrator:** Monitor both agents, flag conflicts