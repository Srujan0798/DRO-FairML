# Agent A1 — kNN ablation summary (attack_k ∈ {5,15}, IF attack)

Analysis-only. No new training. Source: `results/knn_ablation.json` 
(48/360 rows) + canonical IF-attack rows as the k=5 reference.

## Canonical attack_k question (resolved)

- `canonical_tau1.json` has an `attack_k` field on IF rows: **False**
- implicit canonical attack_k = **5** (per `run_fairness_pgd.run_single_experiment` default)
- canonical_tau1.json has NO attack_k field on IF rows; per run_fairness_pgd.run_single_experiment default attack_k=5, the canonical IF-attack grid IS the k=5 reference. The A1 driver's missing_configs() should skip attack_k=5 rows that overlap canonical.

## Coverage

- Ablation rows present: **48/360** (13.3%)
- Complete (dataset,α,attack_k) cells with n≥6 both methods: **2/30**
- **INCOMPLETE** — table below reflects partial data; re-run as more rows land (idempotent).

## Per-cell means (DP / IF / acc)

| dataset | α | k | method | n | DP | IF | acc |
|---|---|---|---|---|---|---|---|
| adult | 0.0 | 5 | dro | 6 | 0.1426 | 0.0933 | 0.8147 |
| adult | 0.0 | 5 | naive | 6 | 0.1491 | 0.0942 | 0.8135 |
| adult | 0.0 | 15 | dro | 6 | 0.1426 | 0.2900 | 0.8147 |
| adult | 0.0 | 15 | naive | 6 | 0.1491 | 0.2972 | 0.8135 |
| adult | 0.1 | 5 | dro | 4 | 0.0768 | 0.0338 | 0.8154 |
| adult | 0.1 | 5 | naive | 6 | 0.0820 | 0.0406 | 0.8124 |
| adult | 0.1 | 15 | dro | 4 | 0.0693 | 0.0961 | 0.8136 |
| adult | 0.1 | 15 | naive | 6 | 0.0729 | 0.1172 | 0.8118 |
| adult | 0.2 | 5 | naive | 2 | 0.0509 | 0.0290 | 0.7882 |
| adult | 0.2 | 15 | naive | 2 | 0.0472 | 0.0966 | 0.7816 |

## Paired Wilcoxon: k=15 vs k=5 (seed-paired, one-sided H1: k15 > k5)

Positive ΔIF / ΔDP ⇒ larger k raises the metric (stronger attack). * marks p<0.05.

| dataset | α | method | n | IF_k5 | IF_k15 | ΔIF | wins_k15 | p_IF | | DP_k5 | DP_k15 | ΔDP | wins_k15 | p_DP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | dro | 6 | 0.0933 | 0.2900 | +0.1968 | 6/6 | 0.0156 * | | 0.1426 | 0.1426 | +0.0000 | 0/6 | 1.0000  |
| adult | 0.0 | naive | 6 | 0.0942 | 0.2972 | +0.2030 | 6/6 | 0.0156 * | | 0.1491 | 0.1491 | -0.0000 | 1/6 | 0.9844  |
| adult | 0.1 | dro | 4 | 0.0338 | 0.0961 | +0.0624 | 4/4 | 0.0625  | | 0.0768 | 0.0693 | -0.0076 | 0/4 | 1.0000  |
| adult | 0.1 | naive | 6 | 0.0406 | 0.1172 | +0.0765 | 6/6 | 0.0156 * | | 0.0820 | 0.0729 | -0.0092 | 0/6 | 1.0000  |
| adult | 0.2 | naive | 2 | 0.0290 | 0.0966 | +0.0676 | 2/2 | 0.2500  | | 0.0509 | 0.0472 | -0.0037 | 0/2 | 1.0000  |

## One-sentence answer — does the IF attack's strength depend on k?

Partial: across 5 cells, larger k rises IF-violation (mean ΔIF=+0.1213, 3/5 p<0.05) and falls DP (mean ΔDP=-0.0041, 0/5 p<0.05) — the attack's strength DOES depend on k.
