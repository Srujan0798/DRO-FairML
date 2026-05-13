# Agent Briefing: DRO-FAIR Hyperparameter Sweep

> **Your task:** Run one script. Report results. No other changes needed.

## What You're Doing

The DRO-FAIR algorithm is not beating the Naive baseline. We suspect the hyperparameters are wrong. Your job is to run a sweep and find settings where DRO wins.

## Step 1: Run the Sweep Script

```bash
cd /Users/srujansai/Desktop/DRO-FairML
python3 experiments/hyperparam_sweep.py
```

This will run **18 experiments** (6 configs × 3 seeds) on Adult dataset, α=0.2. Takes **1-2 hours** on CPU.

## Step 2: Read the Output

The script prints a table like this:

```
Config               Naive DP       DRO DP     DRO Wins          Status
--------------------------------------------------------------------------------
baseline             0.1537±0.012  0.1603±0.023   1/3       ~ Mixed
lr_lambda_0.02       0.1537±0.012  0.0456±0.008   3/3        🎉 ALL WINS
...
```

## Step 3: Report Back

Tell your architect (the user) EXACTLY:

1. **Which config won the most seeds?**
2. **Did ANY config win all 3 seeds?** (This is the gold standard.)
3. **What was the best DRO DP value achieved?**
4. **How long did it take?**

## What the Configs Mean

| Config | What changed | Why we're testing it |
|--------|-------------|---------------------|
| `baseline` | Nothing — current code | Reference point |
| `lr_lambda_0.02` | Dual ascent 4× faster | Maybe λ grows too slowly |
| `K_inner_50` | 50 p-update steps (vs 10) | Maybe p doesn't find worst case |
| `lr_p_0.02` | p-weights move 4× faster | Maybe p converges too slowly |
| `epochs_60` | Train for 60 epochs (vs 30) | Maybe needs more time |
| `dp_only` | Only DP constraint, no IF | Maybe IF interferes with DP |

## If Something Breaks

- **Tests fail?** Run `python3 -m pytest tests/ -v` and report which test fails.
- **Script crashes?** Copy-paste the full error message.
- **Takes >3 hours?** That's normal for the baseline config (DRO is slow). Don't kill it.

## Success Criteria

- 🎉 **Any config wins 3/3 seeds → VICTORY.** Use that config for full experiments.
- ⚠️ **Best config wins 2/3 → TUNABLE.** Try combining best settings (e.g., lr_lambda=0.02 + K_inner=50).
- ❌ **Best config wins 0-1/3 → MAJOR REWRITE NEEDED.** Read `MASTER_PROTOCOL.md` Task 1 (minibatch implementation).

## Files You'll Touch

**ONLY read these. Do NOT modify anything unless told to.**
- `experiments/hyperparam_sweep.py` — the script you run
- `results/hyperparam_sweep.json` — output file (auto-created)

## Questions?

Don't ask. Just run it and report results. If the script runs without crashing, you're doing it right.
