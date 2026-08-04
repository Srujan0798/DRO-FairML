# Agent A4 — Random vs Adversarial summary (τ=1, DP attack)

Analysis-only. No new training. Source: `results/random_vs_adversarial.json` 
(24/144 rows). Historical claim in paper: adversarial corruption raises DP **12-40×** more than random corruption.

## Coverage

- RvA rows present: **24/144** (16.7%)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## ΔDP = naive_dp − dro_dp per (dataset, α, corruptor)

H1 (Wilcoxon): naive_dp > dro_dp. * marks p<0.05.

| dataset | α | corruptor | n | DP_naive | DP_dro | ΔDP | wins_dro | p | sig |
|---|---|---|---|---|---|---|---|---|---|
| adult | 0.1 | adversarial | 5 | 0.2047 | 0.2017 | +0.0030 | 4/5 | 0.0625 |  |
| adult | 0.1 | random | 5 | 0.1175 | 0.1049 | +0.0125 | 5/5 | 0.0312 | * |

## Multiplier table — ΔDP(adv) / ΔDP(random)

Claimed range: **12-40×**. Cells outside this range are flagged explicitly.

| dataset | α | n_adv | n_random | ΔDP_adv | ΔDP_random | multiplier | in 12-40×? |
|---|---|---|---|---|---|---|---|
| adult | 0.1 | 5 | 5 | +0.0030 | +0.0125 | 0.24 | **NO — below 12×** |

## Verdict — is the 12-40× claim substantiated?

Corrected: of 1 finite cells, 0 fall in 12-40×, 1 are below 12×, 0 are above 40× (min=0.2×, median=0.2×, max=0.2×). The '12-40×' claim should be revised to the observed range.
