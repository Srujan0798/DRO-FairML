# MEETING PLAN — Fri May 29, 3 PM (~45 hours away)

**Today:** Thu May 27, evening
**Status:** Partial work exists, neither task fully done

---

## 🎯 What madam asked

1. **Implement PGD for fairness metrics** (DP-only, IF-only, both) — see DRO performance on Adult
2. **Set up UTKFace experiment on GPU server** — repeat similar experiment

---

## 📦 What we have right now

| Item | Owner | Status |
|---|---|---|
| `FairnessTargetedPGD` class (DP works) | Agent A | UNCOMMITTED, IF mode broken (returns zeros) |
| `docs/FAIRNESS_PGD_DESIGN.md` | Agent A | Committed |
| `CNNClassifier`, `ImagePGD`, `run_utkface.py` | Agent B | Committed `a31d43f` (CPU/synthetic fallback) |
| `load_utkface()` | Agent A (mistake) | UNCOMMITTED placeholder |
| GPU server access | — | BLOCKED (hostname unresolvable) |
| Tests in `tests/` | — | Missing |
| Experiment driver `run_fairness_pgd.py` | — | Missing |
| Results, figures, writeup | — | None |

---

## ⏱ 45-hour schedule

| When | Hours | Who | What |
|---|---|---|---|
| **Tonight May 27 (6 PM → 12 AM)** | 6 | Agent A | Fix IF gradient, write tests, COMMIT |
| **Thu May 28 morning (8 AM → 1 PM)** | 5 | Agent A | Write `run_fairness_pgd.py`, smoke test |
| **Thu May 28 morning (8 AM → 1 PM)** | 5 | Agent B | Retry GPU, otherwise CPU synthetic smoke test |
| **Thu May 28 afternoon (1 → 6 PM)** | 5 | Agent A | Launch full Adult+Credit+LSAC × 3 attacks × 2 methods × 5 seeds |
| **Thu May 28 afternoon (1 → 6 PM)** | 5 | Agent B | Launch UTKFace (GPU full or CPU smoke depending on access) |
| **Thu evening → Fri morning** | overnight | (background) | Experiments run |
| **Fri May 29 (8 AM → 12 PM)** | 4 | Orchestrator | Aggregate JSON, generate figures, write 1-page results docs |
| **Fri May 29 (12 PM → 2:30 PM)** | 2.5 | You + Orchestrator | Build slides, dry run |
| **Fri May 29 3 PM** | 1 | You | MEETING |

---

## 🅰️ AGENT A BRIEF (paste verbatim — START NOW)

```
You are AGENT A for /Users/srujansai/Desktop/DRO-FairML. Today is Thu May 27.
You have ONE EVENING + tomorrow morning to ship Task 1 (Fairness-PGD).
Meeting is Fri May 29 at 3 PM. Do not over-engineer. Ship working code.

CONTEXT:
- src/corruption/adversarial.py has FairnessTargetedPGD class (UNCOMMITTED).
- DP mode works (verified by scripts/test_fairness_pgd.py).
- IF mode is BROKEN: compute_if_gradient returns zeros.
- combined mode is also broken (depends on IF).
- No experiment driver exists.
- No tests in tests/ folder.
- Agent B owns src/data/, src/models/, src/corruption/image_pgd.py,
  experiments/run_utkface.py — DO NOT TOUCH THESE.

YOUR DELIVERABLES — STRICTLY IN ORDER:

== STEP 1 (1.5h): Fix IF gradient ==
In src/corruption/adversarial.py:
- Modify compute_if_gradient(self, y, a, X=None, k=5) signature
- Implementation:
  from sklearn.neighbors import NearestNeighbors
  grad = np.zeros(len(y))
  for group in [0, 1]:
      mask = (a == group)
      if mask.sum() < k+1: continue
      X_g, y_g = X[mask], y[mask]
      knn = NearestNeighbors(n_neighbors=min(k+1, len(X_g)))
      knn.fit(X_g)
      _, idxs = knn.kneighbors(X_g)
      group_indices = np.where(mask)[0]
      for local_i, global_i in enumerate(group_indices):
          neighbor_locals = idxs[local_i][1:]   # skip self
          neighbor_globals = group_indices[neighbor_locals]
          agree = np.sum(y[neighbor_globals] == y[global_i])
          disagree = len(neighbor_locals) - agree
          grad[global_i] = (agree - disagree) / max(len(neighbor_locals), 1)
  return grad
- Update compute_fairness_gradient(...) to pass X through to compute_if_gradient
- Update _attack_labels_fairness(...) signature to accept X
- Update corrupt(...) to pass X when calling _attack_labels_fairness

== STEP 2 (1h): Write tests ==
Create tests/test_fairness_pgd.py with pytest functions:
  test_dp_attack_increases_dp(): synth 500 samples, run dp attack at α=0.2,
    assert dp_after >= 1.2 * dp_before
  test_if_attack_increases_if(): same with if attack
  test_combined_attack(): assert at least one metric increases meaningfully
  test_alpha_budget(): assert corrupt_mask.sum() == int(α * n)
  test_minority_targeted(): with coordinated=True, assert majority of corruptions
    hit minority group (>= 60%)
Run: pytest tests/test_fairness_pgd.py -v
ALL 5 MUST PASS before proceeding.

== STEP 3 (2h): Experiment driver ==
Create experiments/run_fairness_pgd.py.
Inputs:
  --datasets {adult,credit,lsac}+   default: adult credit lsac
  --attacks  {dp,if,combined}+      default: dp if combined
  --methods  {naive,dro}+           default: naive dro
  --alphas   float+                 default: 0.1 0.2 0.3
  --n_seeds  int                    default: 5    (NOT 10 — time pressure)
  --smoke                           flag: 1 seed, 1 dataset, 1 alpha
Loop:
  for dataset, alpha, seed, attack, method:
      X, y, a = load_dataset(dataset)
      X_tr, X_te, y_tr, y_te, a_tr, a_te = stratified_split(seed)
      attacker = FairnessTargetedPGD(alpha=alpha, target_metric=attack,
                                     coordinated=True, random_state=seed)
      X_tr_c, y_tr_c, a_tr_c, _ = attacker.corrupt(X_tr, y_tr, a_tr)
      trainer = NaiveFair(...) if method=='naive' else DroFair(...)
      trainer.fit(X_tr_c, y_tr_c, a_tr_c)
      metrics = evaluate(trainer, X_te, y_te, a_te)  # clean test
      results.append({dataset, alpha, seed, attack, method,
                      acc:..., dp:..., if:...})
Save to results/fairness_pgd_results.json
Print progress: "[i/N] adult α=0.2 seed=3 attack=if method=dro: dp=0.041, if=0.012"

== STEP 4 (10 min): Smoke ==
python3 experiments/run_fairness_pgd.py --smoke
Must complete in <8 min and produce JSON with 6 rows (3 attacks × 2 methods).
If smoke passes, COMMIT now:
  git add src/corruption/adversarial.py tests/test_fairness_pgd.py \
          experiments/run_fairness_pgd.py
  git commit -m "Add FairnessTargetedPGD (DP/IF/combined) + experiment driver + tests"
  git push origin main

== STEP 5 (overnight): Full run ==
Launch in background:
  mkdir -p logs
  nohup python3 experiments/run_fairness_pgd.py > logs/fpgd_full.log 2>&1 &
  echo $! > logs/fpgd.pid
Expected: 3 datasets × 3 attacks × 2 methods × 3 alphas × 5 seeds = 270 trainings
At ~30s/training = ~2.5 hours total. Should finish overnight.
Verify it started: tail -f logs/fpgd_full.log (look for first "[1/270]" line).

== STEP 6 (Fri AM, 1.5h): Aggregation ==
Create experiments/analyze_fairness_pgd.py:
- Loads results/fairness_pgd_results.json
- For each (dataset, attack), computes mean ± SE of acc/dp/if across seeds
- For each (dataset, attack), runs scipy wilcoxon(naive_dp, dro_dp, alt='greater')
- Output: results/fairness_pgd_summary.csv with columns
  dataset, attack, alpha, naive_dp, dro_dp, naive_if, dro_if, dp_reduction%, if_reduction%, p_value, sig
- Print summary table to stdout

== STEP 7 (Fri AM, 1.5h): Figures ==
Create figures/fig8_attack_defense_matrix.pdf:
  Heatmap, rows = attacks (dp, if, combined), cols = datasets (adult, credit, lsac)
  Cell color = DP reduction % of DRO over Naive (green = good)
  Cell text = "%, p<0.05" or "n.s."
Create figures/fig9_fairness_pgd_curves.pdf:
  3 subplots (1 per dataset), x-axis = alpha, y-axis = DP violation
  Lines: naive_dp (red), dro_dp (green) for each attack type
  Style = same as existing fig1 (Computer Modern fonts, error bars)
Both: PDF + PNG, 300dpi, savefig with bbox_inches='tight'

== STEP 8 (Fri AM, 30 min): Writeup ==
Create docs/FAIRNESS_PGD_RESULTS.md (1-2 pages):
- ## Setup (3 attack modes, what they do, equations)
- ## Results (table from CSV)
- ## Key findings (3 bullets, e.g. "DP-PGD: DRO defends on Credit/LSAC, Adult collapses")
- ## Limitations (Adult feedback loop persists; no IF-only attack tested before)

== STEP 9 (Fri 11 AM): Final commit + push ==
git add results/ figures/fig8* figures/fig9* docs/FAIRNESS_PGD_RESULTS.md \
        experiments/analyze_fairness_pgd.py
git commit -m "Week 2 Task 1: Fairness-PGD experiments + figures + writeup"
git push origin main

REPORT BACK to orchestrator at each step boundary with: what you ran, key output.

DO NOT touch: src/data/, src/models/, src/corruption/image_pgd.py, run_utkface.py.
```

---

## 🅱️ AGENT B BRIEF (paste verbatim — START IN PARALLEL)

```
You are AGENT B for /Users/srujansai/Desktop/DRO-FairML. Today is Thu May 27.
Meeting Fri May 29 at 3 PM. You already shipped CNNClassifier + ImagePGD +
run_utkface.py with synthetic fallback (commit a31d43f). GPU server is BLOCKED.

YOUR DELIVERABLES — STRICTLY IN ORDER:

== STEP 1 (30 min): Retry GPU access ==
Check email for sysadmin reply. Try ssh again with any hostname they provided.
If working: skip to Step 2 (full run).
If still blocked: send polite follow-up email, proceed with CPU/synthetic plan.

== STEP 2 (1h): UTKFace smoke ==
Whether GPU or CPU:
  python3 experiments/run_utkface.py --smoke
Smoke should: load synthetic 200-image proxy, train 3 epochs, output 1 JSON row.
Must complete in <5 min. If crash → fix and rerun. Report exit status.

== STEP 3 (overnight, depends on GPU): Full run ==
IF GPU AVAILABLE:
  nohup python3 experiments/run_utkface.py \
    --alphas 0.0 0.1 0.2 0.3 \
    --n_seeds 3 \
    --methods naive dro \
    > logs/utkface_full.log 2>&1 &
  Expected: ~8 hours total.

IF NO GPU (CPU/synthetic):
  python3 experiments/run_utkface.py \
    --alphas 0.0 0.1 0.2 0.3 \
    --n_seeds 1 \
    --methods naive dro \
    --synthetic
  Expected: ~2 hours on CPU. Document in writeup that this is preliminary
  (no real images, no DRO claims), waiting on GPU access.

== STEP 4 (Fri AM, 1.5h): Aggregation + figure ==
Create experiments/analyze_utkface.py:
- Wilcoxon DP test Naive vs DRO at each alpha
- Save results/utkface_summary.csv
Create figures/fig10_utkface_curves.pdf:
- 2 subplots: DP curve, IF curve, x=alpha, Naive vs DRO
- Same style as fig1

== STEP 5 (Fri AM, 30 min): Writeup ==
Create docs/UTKFACE_RESULTS.md (1-2 pages):
- ## Setup (dataset, model architecture, image PGD)
- ## Status (GPU access status — be honest)
- ## Results (numbers if any)
- ## Limitations (synthetic vs real, sample sizes, what's pending GPU)

== STEP 6 (Fri 11 AM): Commit + push ==
git add experiments/run_utkface.py experiments/analyze_utkface.py \
        figures/fig10* results/utkface* docs/UTKFACE_RESULTS.md
git commit -m "Week 2 Task 2: UTKFace pipeline + results (GPU pending)"
git push origin main

REPORT BACK with GPU access status at Step 1.

DO NOT touch: src/corruption/adversarial.py (Agent A's), experiments/run_fairness_pgd.py.
```

---

## 🟢 ORCHESTRATOR (Claude) — my responsibilities

I will NOT write code. I will:
1. **Tonight (Thu evening):** Build skeleton of the 5-minute deck (slides without numbers)
2. **Thu PM:** Review Agent A's commit when it lands. Sanity-check IF gradient.
3. **Fri AM:** Help format results into the deck. Spot-check numbers vs claims.
4. **Fri PM before 3 PM:** Final dry-run of your 5-min pitch.

If agents go off-spec → flag immediately, don't let them drift.

---

## 🎤 What the meeting will look like (5 min)

**Slide 1 (30s) — Recap of last week + this week's task list**
> "Madam, last week you asked me to (1) implement PGD on fairness metrics,
> (2) set up UTKFace. Here's the status."

**Slide 2 (90s) — Fairness-PGD attack design**
- Equations for 3 attack modes (DP / IF / combined)
- One-line: "I attack the fairness gradient, not the classification loss"

**Slide 3 (90s) — Attack-vs-defense matrix on Adult/Credit/LSAC**
- Show fig8 (heatmap) + key numbers
- Honest: where DRO holds (Credit/LSAC) and where it doesn't (Adult, again)

**Slide 4 (60s) — UTKFace status**
- Pipeline built (CNN + image PGD + DRO adapted)
- GPU access status (honest: blocked / unblocked / running)
- Preliminary numbers from synthetic if GPU still down

**Slide 5 (30s) — Next week**
- 2 concrete next steps based on this week's findings

---

## ⚠️ Risk register

| Risk | Mitigation |
|---|---|
| Agent A IF gradient still wrong | Orchestrator reviews commit, runs tests manually |
| Full PGD run crashes | Agent A drops to 3 seeds, only Adult+Credit |
| GPU never granted | UTKFace becomes "pipeline-ready, awaiting GPU" — honest in slides |
| Adult still collapses under PGD-DP | Frame as expected (consistent with Week 1 finding), not failure |
| Out of time | Drop combined-PGD attack; ship DP-only + IF-only only |

---

## 🚨 IF YOU HIT A WALL TONIGHT

If Agent A can't fix IF gradient in 2 hours, **immediately drop scope**:
- Ship DP-PGD only (already works)
- Document "IF-PGD in progress, will demo Friday"
- Don't sit there debugging at 2 AM

The professor will accept honest "in progress" more than overpromise + miss.

---

## ✅ Right now (Thu evening, May 27)

1. **Spawn Agent A with the brief above** (step 1 must finish by 9 PM)
2. **Spawn Agent B with the brief above** (step 1 by 7 PM — just GPU check)
3. **Reply to me with:**
   - Did Agent A start? Initial output?
   - GPU status from Agent B?
4. I'll start building the deck skeleton in parallel.

GO.
