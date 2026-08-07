# TASK E — Independent review of DRO-FAIR-AL: verdict

**Reviewer:** fresh, no project history. **Date:** 2026-08-07. **Commit:** `53e4716`.
Full evidence: `docs/AL_REVIEW.md` (recomputed from raw JSON + live runs; the
`results/*_summary.md` files were treated as claims, not ground truth).

## Verdict: the core claim is CONFIRMED; the framing is overstated.

**Confirmed (recomputed from raw data):**
- µ=0 is a byte-identical no-op end-to-end (same config run at the pre-AL commit
  `4adb128` and at HEAD µ=0: acc/dp/if match to full precision); µ=20 reproduces
  stored rows exactly.
- Augmented-Lagrangian gradient is correct (`d[(µ/2)g²]/dθ = µg·dg/dθ` via
  autograd; g_dp/g_if non-negative by construction; µ=0 gradient bit-identical).
- Headline Wilcoxon: seed-paired by SEED, p = 0.015625 (= 0.0156), DP reduction
  70.8% (α=0.2) / 81.7% (α=0.0), 6/6 seeds. Floors Adult/Credit match; lambda
  starvation (max λ 0.0119 vs 1.5) verified from raw history. No leakage: the only
  µ-dependent code path is the penalty term. α=0.4 / Credit / radius-compound
  boundaries hold under the pre-registered rules. `pytest tests/ -q`: **101 passed**.

**Defects (moderate):**
1. "Accuracy held or improved / Pareto improvement" is false at α=0.0 — accuracy
   drops 0.8147→0.7966, all 6 seeds. 2. At α=0.2 the recommended µ=20 collapses
   seed 3 to the exact constant predictor (acc = test majority rate to 16 digits);
   the aggregate hides it. 3. Step-6 mechanism: near-collapse of predicted-positive
   rates is NOT seed-0-specific — direct runs (seeds 0–3) give group-0/group-1
   train pos rates at µ=20 of (0.0/4.4%), (0.02/14.9%), (0.0/5.4%),
   (0.0/0.0%); seed 3 is a full constant-negative predictor on test (acc =
   majority rate to 16 digits). "Pushes toward the majority class" describes µ=20
   better than "denoises the attack".

**Minor:** "3/6 seeds at or below the floor" is really 3/6 at the floor+0.005
*threshold* (1/6 at the floor); LSAC floor 0.9016 is the test (not training)
majority (0.9019); the +0.005 safe/degen margin is arbitrary and several verdicts
hinge on it. All table numbers reproduce exactly — no fabrication found.

Branch left unmerged for review.
