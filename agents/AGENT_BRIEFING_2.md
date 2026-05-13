# Agent Briefing #2: Parallel Cleanup Tasks

> Run these WHILE the hyperparameter sweep is running. They take ~5 minutes total.

---

## Task A: Run Professor Review Simulator

This tells you what your professor will see when they review your work.

```bash
cd /Users/srujansai/Desktop/DRO-FairML
python3 experiments/professor_review_simulator.py
```

**Expected output right now:** Mostly FAIL (because experiments aren't done yet). That's OK — this is a PREVIEW.

**Why run it now:** You'll see exactly which checks are already passing (code correctness) vs which need experiments (results). Share the output with your architect.

---

## Task B: Fix Stale Config File

```bash
cd /Users/srujansai/Desktop/DRO-FairML
python3 experiments/fix_stale_files.py
```

This updates `configs/default.yaml` with correct hyperparameters and validates all imports.

**Expected:** All ✅ checks.

---

## Task C: Clean Up README (High Priority)

The current `README.md` has **FAKE RESULTS** that don't exist. Your professor will see this and know you didn't run the experiments.

**What to do:**

1. Open `README.md`
2. Find the "Experimental Results" section (lines 17-60)
3. Replace the fake table with this honest placeholder:

```markdown
## Experimental Results

> ⚠️ **Results are being generated.** Full Table 1 will be available after completing
> 150 experiments (3 datasets × 5 alphas × 10 seeds).
>
> Current status: experiments in progress.

### Preliminary Results (Adult α=0.2, 5 seeds)

| Method | Accuracy | DP Violation | IF Violation |
|--------|----------|-------------|--------------|
| Naive-FAIR | 0.8113±0.0039 | 0.1537±0.0120 | 0.0195±0.0010 |
| DRO-FAIR | 0.8121±0.0014 | 0.1603±0.0229 | 0.0199±0.0023 |

**Note:** DRO-FAIR performance is under active tuning. See `FINAL_REPORT.md` for full analysis.
```

4. Also fix:
   - Line 6: Remove "DRO-FAIR must reduce DP violations by up to 83%" claim until proven
   - Line 254: Change "23 unit tests" to "32 unit tests"
   - Anywhere it says "adversarial noise" as the main method — the CURRENT code uses RANDOM corruption for experiments (matching the paper). Only the `AdversarialCorruptor` class exists for comparison/ablation.

---

## Task D: Create Results Validation Script (For Later)

After experiments finish, run this to validate them:

```bash
cd /Users/srujansai/Desktop/DRO-FairML
python3 experiments/professor_review_simulator.py
```

This will give you the PASS/FAIL verdict your professor will use.

---

## What to Report Back

After running Tasks A, B, C:

1. **Professor simulator output:** How many checks pass? Which ones fail?
2. **Fix stale files:** Did all imports work?
3. **README:** Did you update it? Show the diff (what changed).

---

## Do NOT Do These Yet

- ❌ Run the full 150 experiments (wait for hyperparameter sweep results first)
- ❌ Modify training algorithms (wait for sweep to tell you if hyperparameters fix it)
- ❌ Generate plots/tables (no data yet)
