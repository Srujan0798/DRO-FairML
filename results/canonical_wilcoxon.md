# Wilcoxon signed-rank tests (one-sided) — canonical_tau1 data

Source: canonical (69 rows, 6 seeds expected)
n=6 (canonical) — p<0.05 achievable for consistent effects. Previously n=3 limited min p~0.125.

H_a: Naive_DP > DRO_DP  (i.e., DRO yields strictly lower DP violation)
Paired by seed. * marks p<0.05.

Columns: n_seeds, means, diff=naive-dro (positive good for DRO), wins_dro = #seeds DRO strictly better, p, sig.

| dataset | attack | α | n | DP_naive | DP_dro | ΔDP | wins | p | sig | IF_Δ | IF_p | IF_sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | combined | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
| adult | combined | 0.1 | 5 | 0.1523 | 0.1445 | +0.0078 | 5/5 | 0.0312 | * | +0.0000 | 0.0312 | * |
| adult | dp | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
| adult | dp | 0.1 | 6 | 0.2026 | 0.1999 | +0.0027 | 5/6 | 0.0312 | * | +0.0000 | 0.2812 |  |
| adult | if | 0.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 6/6 | 0.0156 | * | -0.0000 | 0.7812 |  |
| adult | if | 0.1 | 5 | 0.0744 | 0.0703 | +0.0041 | 4/5 | 0.0625 |  | +0.0000 | 0.0938 |  |
