# Agent A4 — Random vs Adversarial summary (τ=1, DP attack)

Analysis-only. No new training. Source: `results/random_vs_adversarial.json` 
(144/144 rows). Historical claim in paper: adversarial corruption raises DP **12-40×** more than random corruption.

## Coverage

- RvA rows present: **144/144** (100.0%)

## ΔDP = naive_dp − dro_dp per (dataset, α, corruptor)

H1 (Wilcoxon): naive_dp > dro_dp. * marks p<0.05.

| dataset | α | corruptor | n | DP_naive | DP_dro | ΔDP | wins_dro | p | sig |
|---|---|---|---|---|---|---|---|---|---|
| adult | 0.1 | adversarial | 6 | 0.2026 | 0.1999 | +0.0027 | 5/6 | 0.0312 | * |
| adult | 0.1 | random | 6 | 0.1174 | 0.1050 | +0.0124 | 6/6 | 0.0156 | * |
| adult | 0.2 | adversarial | 6 | 0.2452 | 0.2334 | +0.0118 | 6/6 | 0.0156 | * |
| adult | 0.2 | random | 6 | 0.0883 | 0.0777 | +0.0106 | 6/6 | 0.0156 | * |
| credit | 0.1 | adversarial | 6 | 0.0150 | 0.0134 | +0.0016 | 6/6 | 0.0156 | * |
| credit | 0.1 | random | 6 | 0.0108 | 0.0098 | +0.0010 | 6/6 | 0.0156 | * |
| credit | 0.2 | adversarial | 6 | 0.0197 | 0.0178 | +0.0019 | 6/6 | 0.0156 | * |
| credit | 0.2 | random | 6 | 0.0093 | 0.0077 | +0.0017 | 6/6 | 0.0156 | * |
| lsac | 0.1 | adversarial | 6 | 0.2193 | 0.2539 | -0.0347 | 0/6 | 1.0000 |  |
| lsac | 0.1 | random | 6 | 0.1328 | 0.1234 | +0.0093 | 6/6 | 0.0156 | * |
| lsac | 0.2 | adversarial | 6 | 0.1824 | 0.2230 | -0.0406 | 0/6 | 1.0000 |  |
| lsac | 0.2 | random | 6 | 0.1025 | 0.0899 | +0.0126 | 6/6 | 0.0156 | * |

## Multiplier table — ΔDP(adv) / ΔDP(random)

Claimed range: **12-40×**. Cells outside this range are flagged explicitly.

| dataset | α | n_adv | n_random | ΔDP_adv | ΔDP_random | multiplier | in 12-40×? |
|---|---|---|---|---|---|---|---|
| adult | 0.1 | 6 | 6 | +0.0027 | +0.0124 | 0.21 | **NO — below 12×** |
| adult | 0.2 | 6 | 6 | +0.0118 | +0.0106 | 1.11 | **NO — below 12×** |
| credit | 0.1 | 6 | 6 | +0.0016 | +0.0010 | 1.57 | **NO — below 12×** |
| credit | 0.2 | 6 | 6 | +0.0019 | +0.0017 | 1.13 | **NO — below 12×** |
| lsac | 0.1 | 6 | 6 | -0.0347 | +0.0093 | -3.72 | **NO — below 12×** |
| lsac | 0.2 | 6 | 6 | -0.0406 | +0.0126 | -3.21 | **NO — below 12×** |

## Verdict — is the 12-40× claim substantiated?

Corrected: of 6 finite cells, 0 fall in 12-40×, 6 are below 12×, 0 are above 40× (min=-3.7×, median=0.7×, max=1.6×). The '12-40×' claim should be revised to the observed range.
