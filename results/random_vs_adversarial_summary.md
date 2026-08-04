# Agent A4 — Random vs Adversarial summary (τ=1, DP attack)

Analysis-only. No new training. Source: `results/random_vs_adversarial.json` 
(43/144 rows). Historical claim in paper: adversarial corruption raises DP **12-40×** more than random corruption.

## Coverage

- RvA rows present: **43/144** (29.9%)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## ΔDP = naive_dp − dro_dp per (dataset, α, corruptor)

H1 (Wilcoxon): naive_dp > dro_dp. * marks p<0.05.

| dataset | α | corruptor | n | DP_naive | DP_dro | ΔDP | wins_dro | p | sig |
|---|---|---|---|---|---|---|---|---|---|
| adult | 0.1 | adversarial | 6 | 0.2026 | 0.1999 | +0.0027 | 5/6 | 0.0312 | * |
| adult | 0.1 | random | 6 | 0.1174 | 0.1050 | +0.0124 | 6/6 | 0.0156 | * |
| adult | 0.2 | adversarial | 3 | 0.2480 | 0.2371 | +0.0109 | 3/3 | 0.1250 |  |
| adult | 0.2 | random | 2 | 0.0920 | 0.0823 | +0.0097 | 2/2 | 0.2500 |  |

## Multiplier table — ΔDP(adv) / ΔDP(random)

Claimed range: **12-40×**. Cells outside this range are flagged explicitly.

| dataset | α | n_adv | n_random | ΔDP_adv | ΔDP_random | multiplier | in 12-40×? |
|---|---|---|---|---|---|---|---|
| adult | 0.1 | 6 | 6 | +0.0027 | +0.0124 | 0.21 | **NO — below 12×** |
| adult | 0.2 | 3 | 2 | +0.0109 | +0.0097 | 1.12 | **NO — below 12×** |

## Verdict — is the 12-40× claim substantiated?

Corrected: of 2 finite cells, 0 fall in 12-40×, 2 are below 12×, 0 are above 40× (min=0.2×, median=0.7×, max=1.1×). The '12-40×' claim should be revised to the observed range.
