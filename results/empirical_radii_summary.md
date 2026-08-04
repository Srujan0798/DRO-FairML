# Agent A5 — Empirical radii summary (Adult, DP attack)

Analysis-only. No new training. Source: `results/empirical_radii.json` 
(49/180 rows). Arms: (uniform,uncoordinated)=canonical, 
(uniform,coordinated), (empirical,coordinated).

## Coverage

- Empirical-radii rows present: **49/180** (27.2%)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## Per-cell means for DRO (α, radii_mode, coordinated)

| α | radii_mode | coordinated | n | DP | IF | acc |
|---|---|---|---|---|---|---|
| 0.0 | uniform | False | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.0 | uniform | True | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.0 | empirical | True | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.1 | uniform | False | 2 | 0.2106 | 0.0477 | 0.8201 |
| 0.1 | uniform | True | 1 | 0.1880 | 0.0521 | 0.8273 |
| 0.1 | empirical | True | 1 | 0.1880 | 0.0521 | 0.8273 |
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
| 0.1 | 1 | 0.1880 | 0.1880 | -0.0000 | 0/1 | 1.0000  | 0.0521 | 0.0521 | +0.0000 | 0.8273 | 0.8273 | +0.0000 |

## Verdict — does attack-aware radius calibration improve DRO?

No: empirical+coordinated does NOT lower DP in any cell (0/1); attack-aware radius calibration does not help under this coordinated attack.
