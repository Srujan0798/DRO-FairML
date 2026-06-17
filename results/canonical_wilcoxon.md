# Wilcoxon signed-rank tests (one-sided) — canonical_tau1 data

Source: canonical (57 rows, 6 seeds expected)
n=6 (canonical) — p<0.05 achievable for consistent effects. Previously n=3 limited min p~0.125.

H_a: Naive_DP > DRO_DP  (i.e., DRO yields strictly lower DP violation)
Paired by seed. * marks p<0.05.

Columns: n_seeds, means, diff=naive-dro (positive good for DRO), wins_dro = #seeds DRO strictly better, p, sig.

| dataset | attack | α | n | DP_naive | DP_dro | ΔDP | wins | p | sig | IF_Δ | IF_p | IF_sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | combined | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
| adult | combined | 0.1 | 3 | 0.1540 | 0.1476 | +0.0064 | 3/3 | 0.1250 |  | +0.0000 | 0.1250 |  |
| adult | dp | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
| adult | dp | 0.1 | 4 | 0.2039 | 0.2005 | +0.0034 | 3/4 | 0.1250 |  | +0.0000 | 0.1875 |  |
| adult | if | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
| adult | if | 0.1 | 3 | 0.0759 | 0.0730 | +0.0029 | 2/3 | 0.2500 |  | +0.0000 | 0.2500 |  |
