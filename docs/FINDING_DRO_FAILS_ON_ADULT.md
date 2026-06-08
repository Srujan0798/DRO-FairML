# Critical Finding: DRO Fails on Adult (Even with Fixed Attack)

## Observation from Partial Results (19/270 experiments)

For **Adult α=0.1**, DRO consistently produces **HIGHER** DP violation than Naive:

| Attack | Naive DP | DRO DP | DRO Worse By |
|--------|----------|--------|-------------|
| DP | 0.1827 | 0.1995 | +9.2% |
| IF | 0.1499 | 0.1795 | +19.8% |
| Combined | 0.1673 | 0.1789 | +6.9% |

## Why This Happens

### Root Cause: DRO Radii Assume Uniform Corruption

The DRO trainer computes TV radii using:
```python
pi_clean[j] = (pi_obs[j] - alpha) / (1 - 2 * alpha)
```

This formula **assumes uniform random corruption** across the dataset. But the actual `FairnessTargetedPGD` attack uses **coordinated targeting** (70% of corruption budget goes to the minority group).

### Consequence

1. The observed group proportions `pi_obs` under coordinated attack are **very different** from what uniform corruption would produce
2. The "bias-corrected" `pi_clean` is therefore **wrong**
3. The resulting radii `rho_dp` are **miscalibrated**
4. DRO's inner maximization doesn't explore the right uncertainty set
5. `lambda_DP` stays small (~0.046-0.050) because the DRO formulation doesn't "see" the true adversarial distribution

### Lambda Diagnostic Confirms This

| Dataset | λ_DP final | DRO DP |
|---------|-----------|--------|
| Adult | 0.046-0.050 | 0.136-0.159 |
| Credit | 0.015-0.023 | **0.000** |
| LSAC | 0.013-0.014 | **0.000** |

Credit and LSAC achieve perfect fairness because their **base rates are more imbalanced**, making the coordinated attack less effective at fooling DRO. Adult has more balanced groups, so the coordinated attack exploits the radii mismatch more severely.

## What Madam Needs to Know

1. **The attack fix is correct and working** — DP violations are now ~3-5× larger than before (0.047 → 0.18-0.20)
2. **DRO's radii formula is the issue** — it assumes uniform corruption but the attack is coordinated
3. **This is a research design problem, not a code bug** — the paper's theory doesn't account for coordinated adversaries
4. **Potential fixes:**
   - Compute radii using the actual coordinated attack distribution
   - Use a larger fixed radius that covers coordinated attacks
   - Abandon radii and use a fixed budget (like the old code did)

## Recommendation

Present this finding to madam as:
> "We fixed the attack (it was indeed too weak — now 3-5× stronger). With the correct attack, we discovered that DRO's radii computation assumes uniform corruption, but our attack is coordinated. This causes DRO to underperform on Adult. The fix requires either changing the radii formula or switching to a fixed-budget uncertainty set."
