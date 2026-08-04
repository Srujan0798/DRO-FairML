# Agent A5 — Empirical radii summary (Adult, DP attack)

Analysis-only. No new training. Source: `results/empirical_radii.json` 
(180/180 rows). Arms: (uniform,uncoordinated)=canonical, 
(uniform,coordinated), (empirical,coordinated).

## Coverage

- Empirical-radii rows present: **180/180** (100.0%)

## Per-cell means for DRO (α, radii_mode, coordinated)

| α | radii_mode | coordinated | n | DP | IF | acc |
|---|---|---|---|---|---|---|
| 0.0 | uniform | False | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.0 | uniform | True | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.0 | empirical | True | 6 | 0.1426 | 0.0933 | 0.8147 |
| 0.1 | uniform | False | 6 | 0.1999 | 0.0466 | 0.8177 |
| 0.1 | uniform | True | 6 | 0.1784 | 0.0523 | 0.8205 |
| 0.1 | empirical | True | 6 | 0.1784 | 0.0523 | 0.8205 |
| 0.2 | uniform | False | 6 | 0.2334 | 0.0475 | 0.7586 |
| 0.2 | uniform | True | 6 | 0.0143 | 0.0156 | 0.7566 |
| 0.2 | empirical | True | 6 | 0.0143 | 0.0156 | 0.7566 |
| 0.3 | uniform | False | 6 | 0.2614 | 0.0638 | 0.6755 |
| 0.3 | uniform | True | 6 | 0.0600 | 0.0167 | 0.7296 |
| 0.3 | empirical | True | 6 | 0.0600 | 0.0167 | 0.7296 |
| 0.4 | uniform | False | 6 | 0.2855 | 0.0717 | 0.5607 |
| 0.4 | uniform | True | 6 | 0.1287 | 0.0251 | 0.6598 |
| 0.4 | empirical | True | 6 | 0.1287 | 0.0251 | 0.6598 |

## Clean comparison: empirical+coordinated vs uniform+coordinated (DRO)

Same corrupted data (coordinated=True), different radius calibration. H1 (Wilcoxon on DP): dp_uniform > dp_empirical (empirical radii lower DP = better). * marks p<0.05.

| α | n | DP_uni | DP_emp | ΔDP(uni-emp) | wins_emp | p_DP | IF_uni | IF_emp | ΔIF | acc_uni | acc_emp | Δacc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 6 | 0.1426 | 0.1426 | +0.0000 | 0/6 | 1.0000  | 0.0933 | 0.0933 | +0.0000 | 0.8147 | 0.8147 | +0.0000 |
| 0.1 | 6 | 0.1784 | 0.1784 | -0.0000 | 2/6 | 0.7812  | 0.0523 | 0.0523 | -0.0000 | 0.8205 | 0.8205 | +0.0000 |
| 0.2 | 6 | 0.0143 | 0.0143 | -0.0000 | 2/6 | 0.8906  | 0.0156 | 0.0156 | +0.0000 | 0.7566 | 0.7566 | +0.0000 |
| 0.3 | 6 | 0.0600 | 0.0600 | -0.0000 | 2/6 | 0.7812  | 0.0167 | 0.0167 | -0.0000 | 0.7296 | 0.7296 | +0.0000 |
| 0.4 | 6 | 0.1287 | 0.1287 | -0.0000 | 3/6 | 0.6562  | 0.0251 | 0.0251 | -0.0000 | 0.6598 | 0.6598 | +0.0000 |

## Verdict — does attack-aware radius calibration improve DRO?

No: empirical+coordinated does NOT lower DP in any cell (0/5); attack-aware radius calibration does not help under this coordinated attack.
