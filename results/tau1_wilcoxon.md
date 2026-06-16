# Wilcoxon tests — current tau=1 data (n=3 seeds)

> **Caveat.** With n=3 paired samples the minimum attainable
> one-sided p-value is 0.125. The 6-seed re-run is in progress
> and will be appended here when it lands. The table below
> reports descriptive wins / losses and p-values for the record.

Pairs: Naive DP − DRO DP. H_a: Naive > DRO (DRO is fairer).

| Dataset | Attack | α | n | DP naive | DP dro | Δ (n−d) | DRO wins / n | p |
|---|---|---|---|---|---|---|---|
| adult | combined | 0.0 | 3 | 0.1520 | 0.1457 | +0.0063 | 3/3 | 0.125 |
| adult | combined | 0.1 | 3 | 0.1540 | 0.1476 | +0.0064 | 3/3 | 0.125 |
| adult | combined | 0.2 | 3 | 0.1985 | 0.1826 | +0.0159 | 3/3 | 0.125 |
| adult | combined | 0.3 | 3 | 0.2188 | 0.1951 | +0.0237 | 3/3 | 0.125 |
| adult | combined | 0.4 | 3 | 0.2134 | 0.1848 | +0.0286 | 3/3 | 0.125 |
| adult | dp | 0.0 | 3 | 0.1520 | 0.1457 | +0.0063 | 3/3 | 0.125 |
| adult | dp | 0.1 | 3 | 0.2068 | 0.2046 | +0.0022 | 2/3 | 0.250 |
| adult | dp | 0.2 | 3 | 0.2480 | 0.2371 | +0.0109 | 3/3 | 0.125 |
| adult | dp | 0.3 | 3 | 0.2855 | 0.2640 | +0.0215 | 3/3 | 0.125 |
| adult | dp | 0.4 | 3 | 0.3101 | 0.2834 | +0.0267 | 3/3 | 0.125 |
| adult | if | 0.0 | 3 | 0.1520 | 0.1457 | +0.0063 | 3/3 | 0.125 |
| adult | if | 0.1 | 3 | 0.0759 | 0.0729 | +0.0030 | 2/3 | 0.250 |
| adult | if | 0.2 | 3 | 0.0450 | 0.0448 | +0.0002 | 2/3 | 0.375 |
| adult | if | 0.3 | 3 | 0.0190 | 0.0218 | -0.0028 | 0/3 | 1.000 |
| adult | if | 0.4 | 3 | 0.0071 | 0.0052 | +0.0019 | 2/3 | 0.375 |
| credit | combined | 0.0 | 3 | 0.0118 | 0.0109 | +0.0009 | 3/3 | 0.125 |
| credit | dp | 0.0 | 3 | 0.0118 | 0.0109 | +0.0009 | 3/3 | 0.125 |
| credit | if | 0.0 | 3 | 0.0118 | 0.0109 | +0.0009 | 3/3 | 0.125 |
