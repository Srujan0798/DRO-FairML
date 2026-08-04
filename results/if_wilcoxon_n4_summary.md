# Agent N4 — IF-violation paired Wilcoxon under IF attack (D4)

Analysis-only. No new training. Source of truth: `results/canonical_tau1.json` (IF-attack rows only).

Source: canonical_tau1.json (540 total, 180 IF rows)

Test: per (dataset, α), paired Wilcoxon one-sided on the **IF-violation** metric itself (`if_clean`), H1: `naive_if > dro_if` (DRO has **strictly lower** IF violation = better individual fairness). Paired by seed. Significance level α=0.05.

## Protocol-mean verification

Expected (D4): Adult α=0.3 DRO 0.0258 vs Naive 0.0334; Credit 0.1011 vs 0.1212.

  [OK] adult α=0.3: DRO got=0.0258 expected=0.0258 | Naive got=0.0334 expected=0.0334
  [OK] credit α=0.3: DRO got=0.1011 expected=0.1011 | Naive got=0.1212 expected=0.1212

## All (dataset, α) cells — IF violation Wilcoxon

| dataset | α | n_pairs | IF_naive | IF_dro | ΔIF(naive-dro) | wins_dro | p_value | sig |
|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | 6 | 0.0942 | 0.0933 | +0.0010 | 4/6 | 0.1562 |  |
| adult | 0.1 | 6 | 0.0407 | 0.0328 | +0.0079 | 6/6 | 0.0156 | * |
| adult | 0.2 | 6 | 0.0276 | 0.0222 | +0.0053 | 6/6 | 0.0156 | * |
| adult | 0.3 | 6 | 0.0334 | 0.0258 | +0.0076 | 6/6 | 0.0156 | * |
| adult | 0.4 | 6 | 0.0326 | 0.0233 | +0.0092 | 6/6 | 0.0156 | * |
| credit | 0.0 | 6 | 0.0236 | 0.0234 | +0.0002 | 4/6 | 0.1562 |  |
| credit | 0.1 | 6 | 0.0293 | 0.0257 | +0.0036 | 6/6 | 0.0156 | * |
| credit | 0.2 | 6 | 0.0779 | 0.0648 | +0.0131 | 6/6 | 0.0156 | * |
| credit | 0.3 | 6 | 0.1212 | 0.1011 | +0.0201 | 6/6 | 0.0156 | * |
| credit | 0.4 | 6 | 0.1454 | 0.1232 | +0.0222 | 6/6 | 0.0156 | * |
| lsac | 0.0 | 6 | 0.0156 | 0.0228 | -0.0072 | 0/6 | 1.0000 |  |
| lsac | 0.1 | 6 | 0.0174 | 0.0183 | -0.0009 | 2/6 | 0.7812 |  |
| lsac | 0.2 | 6 | 0.0408 | 0.0425 | -0.0016 | 1/6 | 0.9688 |  |
| lsac | 0.3 | 6 | 0.0997 | 0.0909 | +0.0088 | 6/6 | 0.0156 | * |
| lsac | 0.4 | 6 | 0.1405 | 0.1246 | +0.0159 | 6/6 | 0.0156 | * |

* marks p<0.05 (DRO IF violation significantly lower than naive).

## Headline — cells where DRO is significantly lower on IF (p<0.05)

- **adult α=0.1** — IF naive=0.0407 vs dro=0.0328, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **adult α=0.2** — IF naive=0.0276 vs dro=0.0222, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **adult α=0.3** — IF naive=0.0334 vs dro=0.0258, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **adult α=0.4** — IF naive=0.0326 vs dro=0.0233, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **credit α=0.1** — IF naive=0.0293 vs dro=0.0257, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **credit α=0.2** — IF naive=0.0779 vs dro=0.0648, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **credit α=0.3** — IF naive=0.1212 vs dro=0.1011, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **credit α=0.4** — IF naive=0.1454 vs dro=0.1232, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **lsac α=0.3** — IF naive=0.0997 vs dro=0.0909, n=6, wins_dro=6/6, p=0.0156 (<0.05)
- **lsac α=0.4** — IF naive=0.1405 vs dro=0.1246, n=6, wins_dro=6/6, p=0.0156 (<0.05)

## Kuldeep α=0.3 claim — explicit

Kuldeep claim — "if individual fairness is good for α=0.3, then we can state this clearly":
- Adult α=0.3 (IF attack, IF metric): DRO wins 6/6 seeds, dro_if=0.0258 < naive_if=0.0334, p=0.0156 → SIGNIFICANT (DRO IF strictly lower).
- Credit α=0.3 (IF attack, IF metric): DRO wins 6/6 seeds, dro_if=0.1011 < naive_if=0.1212, p=0.0156 → SIGNIFICANT (DRO IF strictly lower).
- Kuldeep claim SUPPORTED on Adult and Credit at α=0.3: YES.

## Coupling caveat — Adult α=0.3 DP under IF attack

Protocol (D4): "Adult α=0.3 still DP loss under IF (coupling)". We verify directly: under IF attack at Adult α=0.3, is `dro_dp > naive_dp`?

Coupling caveat (D4): Adult α=0.3 under IF attack — DP metric under IF attack.
- dp_naive_mean=0.0227, dp_dro_mean=0.0241, Δ(naive-dro)=-0.0014, n=6, dro_dp_wins=1/6, p(H1 naive>dro)=0.8906
- DP loss for DRO at Adult α=0.3 under IF attack: YES — dro_dp > naive_dp (DRO loses on DP), confirming the IF↔DP coupling caveat.

## Interpretation

DRO delivers significantly lower IF violation at α=0.3 on both Adult and Credit (Kuldeep claim supported). However, on Adult at α=0.3 the DP metric under the same IF attack is WORSE for DRO (dro_dp > naive_dp): the IF gain is coupled with a DP loss. State the IF result clearly, but pair it with this DP-coupling caveat in the paper's IF section.
