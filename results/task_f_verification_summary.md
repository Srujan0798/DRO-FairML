# TASK F — Verification Summary (independent recompute from committed data)

**Date:** 2026-08-10 · **Verifier:** fresh recompute from `results/canonical_tau1.json`
(locked 540) vs `results/canonical_tau1_cosine.json` (re-run 540).
**Method:** paired by (dataset, alpha, seed, method, attack); Wilcoxon via
`experiments.compute_canonical_wilcoxon.py::compute_wilcoxon` (one-sided,
`alternative="greater"`, `zero_method="wilcox"`, n=6).

---

## 1. What the memo predicted vs what actually happened

| | Memo prediction (docs/MEMO_FOR_ADVISOR.md §6b) | Actual (measured) |
|---|---|---|
| accuracy | reproduces exactly | **249/540 byte-identical**; drift elsewhere |
| DP | shifts ~1e-7 | 200/540 within 1e-6; max drift 0.085 (combined α≥0.2) |
| IF | goes from noise (~1e-11) to real | ✅ confirmed — old noise 360/540, new real 540/540 |
| Combined attack | not predicted | **drifts materially at α≥0.2** (acc max 7.5%, DP max 8.5%) |

**The memo's prediction was partially wrong.** It was correct for DP- and
IF-attack rows and for the IF column fix, but wrong for the COMBINED-attack
rows, which drift at α≥0.2.

## 2. Measured deltas (540 paired rows)

| Attack | n | acc max | acc mean | DP max | DP mean |
|---|---|---|---|---|---|
| dp | 180 | 0.0028 | 0.0003 | 0.0031 | 0.0003 |
| if | 180 | 0.0053 | 0.0004 | 0.0082 | 0.0004 |
| combined | 180 | **0.0746** | **0.0112** | **0.0850** | **0.0160** |

- accuracy identical (diff < 1e-10): **249/540**
- DP shift < 1e-6: **200/540**
- Old IF was noise (< 1e-9): **360/540** → New IF is real (> 1e-3): **540/540**

## 3. Significance impact (n=6, seed-paired Wilcoxon on DP)

This is the part the last commit's "paper claims robust" verdict missed.
Recomputing significance on the *re-run* data gives **two flips**:

| Cell | Old (locked) | New (re-run) | Flip |
|---|---|---|---|
| **adult / dp / α=0.1** | 5/6, p=0.0312, **sig** | 4/6, p=0.1094, **not sig** | ❌ **FLIP** |
| **lsac / combined / α=0.1** | 6/6, p=0.0156, **sig** | 4/6, p=0.0781, **not sig** | ❌ **FLIP** |

All other cells keep the same significance status. Notable non-flips that
stayed significant: adult/dp α=0.0,0.2,0.3,0.4 (6/6); adult/combined all α
(α=0.1 drops to 5/6 but still p=0.0312); credit/dp all α (6/6); credit/combined
all α (α=0.2 drops to 5/6, p=0.0312).

### Why adult/dp/α=0.1 flips despite tiny drift
The drift there is small (< 0.002), but seed 1's Naive DP drops just below DRO's
(0.20640 vs 0.20653), turning that seed from a win into a loss. The cell was
already the documented honest 5/6 borderline cell; the re-run tips it to 4/6.

## 4. Why combined drifts (root cause)

Combined attack = 0.5·DP + 0.5·IF gradient. Under the old Euclidean IF metric
the IF component was degenerate (~1e-11), so combined ≈ DP-only. After the
cosine fix, IF contributes a real gradient, changing which samples get
corrupted at high α — so the combined-attack rows at α≥0.2 are genuinely
different runs.

## 5. Verdict

- **DP- and IF-attack headline claims are robust** to the re-run: every
  significant cell stays significant EXCEPT adult/dp/α=0.1.
- **adult/dp/α=0.1 is now at risk:** it was already reported as the honest
  5/6 cell; under the corrected-metric code it is 4/6 and no longer
  significant at n=6. This is a **headline-adjacent claim** that the paper
  states explicitly ("Adult/DP α=0.1 is 5/6, p=0.031") and must be
  re-qualified or the re-run data adopted.
- **lsac/combined/α=0.1 also flips** (6/6 → 4/6), affecting the "LSAC/Combined
  is a genuine win" statement.
- The IF column fix works as intended (noise → real measurement).

**Action required before submission:** decide whether to (a) keep the locked
`canonical_tau1.json` as the paper's source (as AGENT_PROMPT_FINAL instructed,
that is Prof. Manisha's call) and re-qualify the two α=0.1 cells, or (b) adopt
the re-run file. The paper currently claims the re-run "reproduces accuracy
exactly and shifts DP by O(10⁻⁷)" — **that sentence is factually wrong** and
must be corrected in `paper/sections/discussion.tex` regardless of (a) or (b).
