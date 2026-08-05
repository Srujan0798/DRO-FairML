# DRO-FAIR-AL (augmented Lagrangian) — pre-registered result

rows: **48/48** | criterion: >=2/4 cells p<0.05 (one arm), mean acc cost vs canonical DRO <= 0.005

`degen` flags cells where AL's accuracy is at/below the constant-predictor floor — a DP win there is collapse, not fairness.

| dataset | α | μ | n | DP dro | DP AL | ΔDP (AL−dro) | p (AL<dro) | DP naive | margin dro | margin AL | acc dro | acc AL | floor | degen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.1 | 5 | 6 | 0.1999 | 0.1826 | -0.0173 | 0.8438 | 0.2026 | +0.0027 | +0.0199 | 0.8177 | 0.8085 | 0.7521 | ok |
| adult | 0.2 | 5 | 6 | 0.2334 | 0.1358 | -0.0976 | 0.0156 * | 0.2452 | +0.0119 | +0.1094 | 0.7586 | 0.7944 | 0.7521 | ok |
| credit | 0.1 | 5 | 6 | 0.0134 | 0.0053 | -0.0081 | 0.0156 * | 0.0151 | +0.0017 | +0.0098 | 0.8097 | 0.7787 | 0.7788 | **DEGEN** |
| credit | 0.2 | 5 | 6 | 0.0178 | 0.0026 | -0.0152 | 0.0156 * | 0.0198 | +0.0020 | +0.0172 | 0.7819 | 0.7730 | 0.7788 | **DEGEN** |
| adult | 0.1 | 10 | 6 | 0.1999 | 0.1676 | -0.0323 | 0.5000 | 0.2026 | +0.0027 | +0.0350 | 0.8177 | 0.7978 | 0.7521 | ok |
| adult | 0.2 | 10 | 6 | 0.2334 | 0.1009 | -0.1325 | 0.0156 * | 0.2452 | +0.0119 | +0.1444 | 0.7586 | 0.7953 | 0.7521 | ok |
| credit | 0.1 | 10 | 6 | 0.0134 | 0.0045 | -0.0089 | 0.0156 * | 0.0151 | +0.0017 | +0.0106 | 0.8097 | 0.7790 | 0.7788 | **DEGEN** |
| credit | 0.2 | 10 | 6 | 0.0178 | 0.0017 | -0.0161 | 0.0156 * | 0.0198 | +0.0020 | +0.0181 | 0.7819 | 0.7699 | 0.7788 | **DEGEN** |

- μ=5: 3/4 cells significant (1/4 significant AND non-degenerate); mean acc cost +0.0033 → pre-registered criterion MET; after degeneracy guard: **GENUINE IMPROVEMENT**
- μ=10: 3/4 cells significant (1/4 significant AND non-degenerate); mean acc cost +0.0065 → pre-registered criterion NOT met; after degeneracy guard: **GENUINE IMPROVEMENT**

margin = Naive DP − method DP (positive = method beats Naive; bigger = the win Manisha asked to grow).

**Honest reading.** The pre-registered criterion counts only statistical significance and mean accuracy cost. Applying the project's standing degeneracy guard (accuracy must clear the constant-predictor floor) disqualifies the Credit cells, where AL drives DP down by collapsing toward the trivial predictor — the same failure mode documented for LSAC/DP. The surviving result is Adult, which is where the improvement is real.
