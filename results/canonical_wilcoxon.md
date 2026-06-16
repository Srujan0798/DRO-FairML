# Wilcoxon signed-rank tests (one-sided) — canonical_tau1 data

Source: fallback tau_ablation_tau1 (109 rows, n<=3 seeds)
PRELIMINARY: n<=3 (tau_ablation fallback); min attainable p=0.125. Regenerate after canonical_tau1.json (6 seeds) lands.

H_a: Naive_DP > DRO_DP  (i.e., DRO yields strictly lower DP violation)
Paired by seed. * marks p<0.05.

Columns: n_seeds, means, diff=naive-dro (positive good for DRO), wins_dro = #seeds DRO strictly better, p, sig.

| dataset | attack | α | n | DP_naive | DP_dro | ΔDP | wins | p | sig | IF_Δ | IF_p | IF_sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | combined | 0.0 | 3 | 0.1520 | 0.1457 | +0.0063 | 3/3 | 0.1250 |  | -0.0000 | 1.0000 |  |
| adult | combined | 0.1 | 3 | 0.1540 | 0.1476 | +0.0064 | 3/3 | 0.1250 |  | +0.0000 | 0.2500 |  |
| adult | combined | 0.2 | 3 | 0.1985 | 0.1826 | +0.0159 | 3/3 | 0.1250 |  | +0.0000 | 0.1250 |  |
| adult | combined | 0.3 | 3 | 0.2188 | 0.1951 | +0.0237 | 3/3 | 0.1250 |  | +0.0000 | 0.3750 |  |
| adult | combined | 0.4 | 3 | 0.2134 | 0.1848 | +0.0286 | 3/3 | 0.1250 |  | +0.0000 | 0.6250 |  |
| adult | dp | 0.0 | 3 | 0.1520 | 0.1457 | +0.0063 | 3/3 | 0.1250 |  | -0.0000 | 1.0000 |  |
| adult | dp | 0.1 | 3 | 0.2068 | 0.2046 | +0.0022 | 2/3 | 0.2500 |  | +0.0000 | 0.1250 |  |
| adult | dp | 0.2 | 3 | 0.2480 | 0.2371 | +0.0109 | 3/3 | 0.1250 |  | -0.0000 | 0.8750 |  |
| adult | dp | 0.3 | 3 | 0.2855 | 0.2640 | +0.0215 | 3/3 | 0.1250 |  | -0.0000 | 0.8750 |  |
| adult | dp | 0.4 | 3 | 0.3101 | 0.2834 | +0.0267 | 3/3 | 0.1250 |  | +0.0000 | 0.6250 |  |
| adult | if | 0.0 | 3 | 0.1520 | 0.1457 | +0.0063 | 3/3 | 0.1250 |  | -0.0000 | 1.0000 |  |
| adult | if | 0.1 | 3 | 0.0759 | 0.0729 | +0.0030 | 2/3 | 0.2500 |  | +0.0000 | 0.2500 |  |
| adult | if | 0.2 | 3 | 0.0450 | 0.0448 | +0.0002 | 2/3 | 0.3750 |  | +0.0000 | 0.2500 |  |
| adult | if | 0.3 | 3 | 0.0190 | 0.0218 | -0.0028 | 0/3 | 1.0000 |  | +0.0000 | 0.3750 |  |
| adult | if | 0.4 | 3 | 0.0071 | 0.0052 | +0.0019 | 2/3 | 0.3750 |  | -0.0000 | 0.6250 |  |
| credit | combined | 0.0 | 3 | 0.0118 | 0.0109 | +0.0009 | 3/3 | 0.1250 |  | +0.0000 | 0.5000 |  |
| credit | dp | 0.0 | 3 | 0.0118 | 0.0109 | +0.0009 | 3/3 | 0.1250 |  | +0.0000 | 0.5000 |  |
| credit | if | 0.0 | 3 | 0.0118 | 0.0109 | +0.0009 | 3/3 | 0.1250 |  | +0.0000 | 0.5000 |  |
