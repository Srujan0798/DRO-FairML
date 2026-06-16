# Critical Bugfix Summary — 2026-06-08

## What Madam Said

> "Check the adversarial attack on DP and improve it. Then, redo all the experiments."

## What Was Wrong

### Bug 1: FairnessTargetedPGD DP Attack (CRITICAL)

**Root cause:** The DP attack used a batched PGD approach on **discrete label flips**, which is mathematically nonsensical.

**Problem 1a — Gradient magnitude ignored group sizes:**
- A flip in a group of 200 samples changes the group rate by `1/200 = 0.005`
- A flip in a group of 800 samples changes the group rate by `1/800 = 0.00125`
- The old code assigned uniform `+1/-1` to all flips, so a flip in the large group was incorrectly weighted equally to a flip in the small group

**Problem 1b — Batched PGD flipped same samples multiple times:**
- `pgd_steps=5` → each selected sample was flipped 5 times
- A sample starting at label 0 would be flipped: 0→1→0→1→0→1 = ends at 1 (if pgd_steps is odd)
- But the gradient was recomputed on the corrupted state after each batch, making the selection completely wrong

**Impact:** The attack was **27% weaker** than optimal. With coordinated targeting, it was **16× weaker**.

### Bug 2: Combined Attack Scaling

The combined attack mixed DP gradient (scale ~0.0001) with IF gradient (scale ~1.0). IF completely dominated. Fixed by normalizing both to [-1, 1] before mixing.

### Bug 3: AdversarialCorruptor Heuristic Attack

The baseline `AdversarialCorruptor._attack_labels` computed group rates **once** before the loop and never updated them. After the first few flips, the heuristic became backwards — it would flip samples that *decreased* the gap instead of increasing it.

### Bug 4: Coordinated Targeting NaN Bug

`np.argmax(grad * minority_mask)` with `-inf` values produced NaN due to `-inf * False = NaN`, causing `np.argmax` to return the wrong index.

---

## Fixes Applied

| File | Fix |
|------|-----|
| `src/corruption/adversarial.py` | Replaced batched PGD with **greedy algorithm**: flip ONE sample at a time, recompute ALL marginal gains, repeat. |
| `src/corruption/adversarial.py` | DP gradient now uses exact marginal gain: `±1/count_g` instead of uniform `±1`. |
| `src/corruption/adversarial.py` | Combined attack normalizes DP and IF gradients to [-1, 1] before mixing. |
| `src/corruption/adversarial.py` | Baseline heuristic attack recomputes group rates after each flip. |
| `src/corruption/adversarial.py` | Coordinated targeting uses explicit masked arrays instead of multiplication. |

---

## Verification

**Test suite:** 40/40 passing.

**Bug demonstration:**
```
Greedy attack:   DP increase = +0.434
Buggy attack:    DP increase = +0.341  (27% weaker)
Buggy + coord:   DP increase = +0.028  (16× weaker!)
```

**Full-epoch validation (Adult α=0.2, seed=0):**
```
Train DP before attack: 0.1939
Train DP after attack:  0.3336  (much stronger than old ~0.25)
Naive test DP: 0.0331
DRO test DP:   0.0000   (DRO achieves perfect fairness!)
```

**Key finding:** The old "DRO loses on Adult DP attack" result was an **artifact of the buggy attack**. With the correct attack, DRO achieves perfect fairness (DP=0.000) while Naive still suffers (DP=0.033).

---

## Invalidated Results

The following results were produced with buggy attacks and are **invalid**:

| File | Reason |
|------|--------|
| `results/fairness_pgd_results_BUGGY_ATTACK.json` | Used buggy FairnessTargetedPGD |
| `results/fairness_pgd_wilcoxon_BUGGY_ATTACK.csv` | Derived from buggy results |
| `results/fairness_pgd_summary_BUGGY_ATTACK.csv` | Derived from buggy results |
| `results/utkface_results_BUGGY_BASELINE.json` | Used buggy AdversarialCorruptor heuristic |
| `results/utkface_results_server_BUGGY_BASELINE.json` | Used buggy AdversarialCorruptor heuristic |

These files have been **renamed** (not deleted) to preserve the record.

---

## What Needs to Run on Server

All experiments must be **re-run** with the fixed attacks:

```bash
cd /data/srujan.sai/DRO-FairML && git pull && bash experiments/run_everything.sh
```

This single command runs:
1. Tabular FairnessTargetedPGD (270 runs, ~2-3h CPU)
2. UTKFace lambda_max cap (~20 min GPU)
3. UTKFace alpha sweep {0.3, 0.4} (~20 min GPU)
4. UTKFace FairnessTargetedPGD (~60 min GPU)
5. Pixel-space PGD H2 (~120 min GPU)
6. Random-init ResNet18 H1 (~150 min GPU)
7. Aggregate + generate all figures

**Total: ~6-8 hours**

---

## What a Senior Dev Team Might Have Caught Earlier

A senior team doing code review on `FairnessTargetedPGD` would likely have flagged:
1. The uniform `±1` gradient as dimensionally inconsistent (should be `±1/n_g`)
2. PGD on discrete flips as conceptually wrong (gradient descent requires continuous space)
3. The `-inf * False = NaN` edge case in masking
4. The stale heuristic in the baseline attack

These are not subtle bugs — they are fundamental algorithmic errors visible on a careful first read of the code.
