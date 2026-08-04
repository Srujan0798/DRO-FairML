# Agent A5 — Empirical radii summary (Adult, DP attack)

Analysis-only. No new training. Source: `results/empirical_radii.json` 
(69/180 rows). Arms: (uniform,uncoordinated)=canonical, 
(uniform,coordinated), (empirical,coordinated).

## Coverage

- Empirical-radii rows present: **69/180** (38.3%)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## Per-cell means for DRO (α, radii_mode, coordinated)

| α | radii_mode | coordinated | n | DP | IF | acc |
|---|---|---|---|---|---|---|
| 0.0 | uniform | False | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.0 | uniform | True | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.0 | empirical | True | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.1 | uniform | False | 4 | 0.2005 | 0.0470 | 0.8195 |
| 0.1 | uniform | True | 4 | 0.1807 | 0.0532 | 0.8224 |
| 0.1 | empirical | True | 4 | 0.1807 | 0.0532 | 0.8224 |
| 0.2 | uniform | False | 0 | — | — | — |
| 0.2 | uniform | True | 0 | — | — | — |
| 0.2 | empirical | True | 0 | — | — | — |
| 0.3 | uniform | False | 0 | — | — | — |
| 0.3 | uniform | True | 0 | — | — | — |
| 0.3 | empirical | True | 0 | — | — | — |
| 0.4 | uniform | False | 0 | — | — | — |
| 0.4 | uniform | True | 0 | — | — | — |
| 0.4 | empirical | True | 0 | — | — | — |

## Clean comparison: empirical+coordinated vs uniform+coordinated (DRO)

Same corrupted data (coordinated=True), different radius calibration. H1 (Wilcoxon on DP): dp_uniform > dp_empirical (empirical radii lower DP = better). * marks p<0.05.

| α | n | DP_uni | DP_emp | ΔDP(uni-emp) | wins_emp | p_DP | IF_uni | IF_emp | ΔIF | acc_uni | acc_emp | Δacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 6 | 0.1426 | 0.1426 | +0.0000 | 0/6 | 1.0000  | 0.0933 | 0.0933 | +0.0000 | 0.8147 | 0.8147 | +0.0000 |
| 0.1 | 4 | 0.1807 | 0.1807 | -0.0000 | 2/4 | 0.5625  | 0.0532 | 0.0532 | -0.0000 | 0.8224 | 0.8224 | +0.0000 |

## Verdict — does attack-aware radius calibration improve DRO?

No: empirical+coordinated does NOT lower DP in any cell (0/2); attack-aware radius calibration does not help under this coordinated attack.
