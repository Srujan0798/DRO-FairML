# Agent A1 — kNN ablation summary (attack_k ∈ {5,15}, IF attack)

Analysis-only. No new training. Source: `results/knn_ablation.json` 
(28/360 rows) + canonical IF-attack rows as the k=5 reference.

## Canonical attack_k question (resolved)

- `canonical_tau1.json` has an `attack_k` field on IF rows: **UNKNOWN**
- implicit canonical attack_k = **5** (per `run_fairness_pgd.run_single_experiment` default)
- canonical load failed: No module named 'experiments'

## Coverage

- Ablation rows present: **28/360** (7.8%)
- Complete (dataset,α,attack_k) cells with n≥6 both methods: **2/30**
- **INCOMPLETE** — table below reflects partial data; re-run as more rows land (idempotent).

## Per-cell means (DP / IF / acc)

| dataset | α | k | method | n | DP | IF | acc |
|---|---|---|---|---|---|---|---|
| adult | 0.0 | 5 | dro | 6 | 0.1426 | 0.0933 | 0.8147 |
| adult | 0.0 | 5 | naive | 6 | 0.1491 | 0.0942 | 0.8135 |
| adult | 0.0 | 15 | dro | 6 | 0.1426 | 0.2900 | 0.8147 |
| adult | 0.0 | 15 | naive | 6 | 0.1491 | 0.2972 | 0.8135 |
| adult | 0.1 | 5 | naive | 2 | 0.0857 | 0.0412 | 0.8166 |
| adult | 0.1 | 15 | naive | 2 | 0.0754 | 0.1290 | 0.8156 |

## Paired Wilcoxon: k=15 vs k=5 (seed-paired, one-sided H1: k15 > k5)

Positive ΔIF / ΔDP ⇒ larger k raises the metric (stronger attack). * marks p<0.05.

| dataset | α | method | n | IF_k5 | IF_k15 | ΔIF | wins_k15 | p_IF | | DP_k5 | DP_k15 | ΔDP | wins_k15 | p_DP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | dro | 6 | 0.0933 | 0.2900 | +0.1968 | 6/6 | 0.0156 * | | 0.1426 | 0.1426 | +0.0000 | 0/6 | 1.0000  |
| adult | 0.0 | naive | 6 | 0.0942 | 0.2972 | +0.2030 | 6/6 | 0.0156 * | | 0.1491 | 0.1491 | -0.0000 | 1/6 | 0.9844  |
| adult | 0.1 | naive | 2 | 0.0412 | 0.1290 | +0.0877 | 2/2 | 0.2500  | | 0.0857 | 0.0754 | -0.0102 | 0/2 | 1.0000  |

## One-sentence answer — does the IF attack's strength depend on k?

Partial: across 3 cells, larger k rises IF-violation (mean ΔIF=+0.1625, 2/3 p<0.05) and falls DP (mean ΔDP=-0.0034, 0/3 p<0.05) — the attack's strength DOES depend on k.
