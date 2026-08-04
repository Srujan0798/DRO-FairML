# Agent A2 — τ ablation summary (τ ∈ {1, 10, 100}, DP attack)

Analysis-only. No new training. Source: `results/tau_ablation.json` 
(36/360 rows, τ∈{10,100}) + canonical τ=1 IF-attack rows 
(0 rows) as the reference via `experiments.loaders`.

## Coverage

- τ-ablation rows present: **36/360** (10.0%)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## The τ=100 artifact — DP for Naive vs DRO at each τ

Per (dataset, α), DP at τ=1 (canonical), τ=10, τ=100. Bold = lower (better).

| dataset | α | n@τ1 | Naive τ=1 | DRO τ=1 | n@τ10 | Naive τ=10 | DRO τ=10 | n@τ100 | Naive τ=100 | DRO τ=100 |
|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | — | — | — | 6 | **0.1590** | 0.1693 | 6 | **0.1575** | 0.1694 |
| adult | 0.1 | — | — | — | 4 | **0.1819** | 0.2171 | 4 | **0.1754** | 0.2033 |
| adult | 0.2 | — | — | — | — | — | — | — | — | — |
| adult | 0.3 | — | — | — | — | — | — | — | — | — |
| adult | 0.4 | — | — | — | — | — | — | — | — | — |
| credit | 0.0 | — | — | — | — | — | — | — | — | — |
| credit | 0.1 | — | — | — | — | — | — | — | — | — |
| credit | 0.2 | — | — | — | — | — | — | — | — | — |
| credit | 0.3 | — | — | — | — | — | — | — | — | — |
| credit | 0.4 | — | — | — | — | — | — | — | — | — |
| lsac | 0.0 | — | — | — | — | — | — | — | — | — |
| lsac | 0.1 | — | — | — | — | — | — | — | — | — |
| lsac | 0.2 | — | — | — | — | — | — | — | — | — |
| lsac | 0.3 | — | — | — | — | — | — | — | — | — |
| lsac | 0.4 | — | — | — | — | — | — | — | — | — |

## Seed-paired Wilcoxon (DRO vs Naive) at each τ

H1: naive_dp > dro_dp (DRO strictly lower DP). * marks p<0.05.

### τ = 1

_(no paired rows yet)_

### τ = 10

| dataset | α | n | Naive DP | DRO DP | ΔDP(naive-dro) | wins_dro | p | sig |
|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | 6 | 0.1590 | 0.1693 | -0.0103 | 0/6 | 1.0000 |  |

### τ = 100

| dataset | α | n | Naive DP | DRO DP | ΔDP(naive-dro) | wins_dro | p | sig |
|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | 6 | 0.1575 | 0.1694 | -0.0120 | 0/6 | 1.0000 |  |
| adult | 0.1 | 3 | 0.1790 | 0.2033 | -0.0242 | 0/3 | 1.0000 |  |

## The flip: τ=100 → DRO loses; τ=1 → DRO wins

_(need both τ=1 and τ=100 paired rows to demonstrate the flip — currently incomplete)_

## Comparison to historical pilot (paper Table~tab:tau-comparison)

Paper Table~tab:tau-comparison lists τ=10/100 numbers from a HISTORICAL stepped-schedule pilot (n=3). The new n=6 fixed-τ runs are a fresh, provenance-clean re-test. CONFIRM = sign + magnitude match; CONTRADICT = sign flip.

| α | τ | hist Naive | hist DRO | hist wins | new Naive | new DRO | new n | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 0.2026 | 0.1999 | 5/6 | — | — | 0 | NO NEW DATA |
| 0.1 | 10 | 0.1850 | 0.2231 | 0/3 | 0.1819 | 0.2171 | 4 | CONFIRM |
| 0.1 | 100 | 0.1801 | 0.2033 | 0/3 | 0.1754 | 0.2033 | 4 | CONFIRM |
| 0.2 | 1 | 0.2452 | 0.2334 | 6/6 | — | — | 0 | NO NEW DATA |
| 0.2 | 10 | 0.3382 | 0.4634 | 0/3 | — | — | 0 | NO NEW DATA |
| 0.2 | 100 | 0.3271 | 0.5030 | 0/3 | — | — | 0 | NO NEW DATA |
| 0.3 | 1 | 0.2848 | 0.2614 | 6/6 | — | — | 0 | NO NEW DATA |
| 0.3 | 10 | 0.5253 | 0.5532 | 0/3 | — | — | 0 | NO NEW DATA |
| 0.3 | 100 | 0.5313 | 0.5622 | 0/3 | — | — | 0 | NO NEW DATA |
| 0.4 | 1 | 0.3140 | 0.2855 | 6/6 | — | — | 0 | NO NEW DATA |
| 0.4 | 10 | 0.5158 | 0.5231 | 0/3 | — | — | 0 | NO NEW DATA |
| 0.4 | 100 | 0.5129 | 0.5260 | 0/3 | — | — | 0 | NO NEW DATA |
