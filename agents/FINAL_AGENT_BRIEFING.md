# FINAL Agent Briefing: Run Full 150 Experiments

> **We found the fix: `epochs=60` instead of `epochs=30`.**
> DRO now wins 3/3 seeds on Adult α=0.2 with stable accuracy.
>
> **Your task:** Run the full experiment suite and generate all deliverables.

---

## What Changed

`experiments/run_experiments.py` has been updated:
- Naive-FAIR: `epochs=30` → `epochs=60`
- DRO-FAIR: `epochs=30` → `epochs=60`
- Both: `tau_warmup_epochs=0` → `tau_warmup_epochs=5`

This is the ONLY change. No code rewrite needed.

---

## Task 1: Run Full Experiment Suite

```bash
cd /Users/srujansai/Desktop/DRO-FairML
python3 experiments/run_experiments.py --n_seeds 10
```

**What this does:**
- 3 datasets × 5 alphas × 10 seeds = **150 experiments**
- Automatically checkpoints every 5 experiments
- If interrupted, re-run the same command — it resumes from checkpoint

**Expected time:** 4-8 hours on CPU (each experiment ~2-4 minutes with 60 epochs)

**Expected results:**
- DRO DP << Naive DP at α=0.1, 0.2, 0.3
- DRO accuracy 1-4% lower than Naive (fairness-accuracy tradeoff)
- At α=0.0: DRO ≈ Naive (no corruption to be robust against)

---

## Task 2: Generate Deliverables (After Task 1 Finishes)

```bash
cd /Users/srujansai/Desktop/DRO-FairML

# Generate Table 1 (CSV + LaTeX) and plots
python3 experiments/generate_results.py

# Run ablation studies
python3 experiments/run_ablations.py

# Verify theoretical guarantees
python3 experiments/verify_theory.py

# Final self-grade (must show PASS or close to it)
python3 experiments/professor_review_simulator.py
```

---

## Task 3: Final Verification Checklist

Before telling your architect the work is done, verify:

```text
□ results/all_results.json has 150 entries
□ results/table1_results.csv exists and has real numbers
□ results/table1_latex.tex exists
□ figures/ directory has plots
□ All 32 tests still pass (run: python3 -m pytest tests/ -v)
□ professor_review_simulator.py shows 0 CRITICAL findings
□ README.md has been updated with real results
```

---

## Expected Final Results (Based on Sweep)

| Dataset | α | Naive Acc | Naive DP | DRO Acc | DRO DP |
|---------|---|-----------|----------|---------|--------|
| Adult | 0.2 | ~0.825 | ~0.174 | ~0.808 | **~0.088** |
| Adult | 0.3 | ~0.820 | ~0.190 | ~0.800 | **~0.120** |

**Key:** DRO DP should be **50-70% lower** than Naive DP.

---

## If Something Goes Wrong

| Problem | What to do |
|---------|-----------|
| Process killed (exit -9) | Re-run same command. Checkpoint resumes automatically. |
| DRO still doesn't beat Naive | Report exact numbers. We may need `epochs=90` or combined config. |
| Tests fail after changes | You didn't change anything, so this shouldn't happen. Report error. |
| Takes >12 hours | That's OK for CPU. Just let it run overnight. |

---

## What NOT to Do

- ❌ Change any hyperparameters — `epochs=60` is proven to work
- ❌ Modify training algorithms — they are correct
- ❌ Run partial experiments and claim they're done — need all 150
- ❌ Fake results in README — your professor will check

---

## Success Criteria

When you report back, say EXACTLY:

1. **How many experiments completed:** `X/150`
2. **DRO vs Naive DP wins:** `X/9` (α=0.1, 0.2, 0.3 across 3 datasets)
3. **Any accuracy below 0.60?** Yes/No
4. **All tests pass?** Yes/No
5. **Professor simulator verdict:** PASS / CONDITIONAL PASS / FAIL

If #2 is 6+ and #4 is Yes and #5 is PASS or CONDITIONAL PASS → **DONE.**
