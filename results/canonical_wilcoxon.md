# Wilcoxon signed-rank tests (one-sided) — canonical_tau1 data

Source: canonical (39 rows, 6 seeds expected)
n=6 (canonical) — p<0.05 achievable for consistent effects. Previously n=3 limited min p~0.125.

H_a: Naive_DP > DRO_DP  (i.e., DRO yields strictly lower DP violation)
Paired by seed. * marks p<0.05.

Columns: n_seeds, means, diff=naive-dro (positive good for DRO), wins_dro = #seeds DRO strictly better, p, sig.

| dataset | attack | α | n | DP_naive | DP_dro | ΔDP | wins | p | sig | IF_Δ | IF_p | IF_sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | combined | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
| adult | dp | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
| adult | if | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
