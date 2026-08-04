# Agent N5 — K_inner ablation summary (K ∈ {5, 10, 20}, DRO only, DP attack)

Analysis-only. No new training. Source: `results/kinner_ablation.json` 
(16/180 rows, K∈{5,20}) + canonical K=10 DRO/DP rows 
(0 rows) as the reference via `experiments.loaders`.

## Coverage

- K_inner ablation rows present: **16/180** (8.9%)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## Per-cell means for DRO (dataset, α, K_inner)

| dataset | α | K | n | DP | IF | acc | wall/config (s) |
|---|---|---|---|---|---|---|---|
| adult | 0.0 | 5 | 6 | 0.1426 | 0.0933 | 0.8147 | 101.4 |
| adult | 0.0 | 10 | 0 | — | — | — | — |
| adult | 0.0 | 20 | 6 | 0.1426 | 0.0933 | 0.8147 | 103.4 |
| adult | 0.1 | 5 | 4 | 0.2004 | 0.0470 | 0.8195 | 812.4 |
| adult | 0.1 | 10 | 0 | — | — | — | — |
| adult | 0.1 | 20 | 0 | — | — | — | — |
| adult | 0.2 | 5 | 0 | — | — | — | — |
| adult | 0.2 | 10 | 0 | — | — | — | — |
| adult | 0.2 | 20 | 0 | — | — | — | — |
| adult | 0.3 | 5 | 0 | — | — | — | — |
| adult | 0.3 | 10 | 0 | — | — | — | — |
| adult | 0.3 | 20 | 0 | — | — | — | — |
| adult | 0.4 | 5 | 0 | — | — | — | — |
| adult | 0.4 | 10 | 0 | — | — | — | — |
| adult | 0.4 | 20 | 0 | — | — | — | — |
| credit | 0.0 | 5 | 0 | — | — | — | — |
| credit | 0.0 | 10 | 0 | — | — | — | — |
| credit | 0.0 | 20 | 0 | — | — | — | — |
| credit | 0.1 | 5 | 0 | — | — | — | — |
| credit | 0.1 | 10 | 0 | — | — | — | — |
| credit | 0.1 | 20 | 0 | — | — | — | — |
| credit | 0.2 | 5 | 0 | — | — | — | — |
| credit | 0.2 | 10 | 0 | — | — | — | — |
| credit | 0.2 | 20 | 0 | — | — | — | — |
| credit | 0.3 | 5 | 0 | — | — | — | — |
| credit | 0.3 | 10 | 0 | — | — | — | — |
| credit | 0.3 | 20 | 0 | — | — | — | — |
| credit | 0.4 | 5 | 0 | — | — | — | — |
| credit | 0.4 | 10 | 0 | — | — | — | — |
| credit | 0.4 | 20 | 0 | — | — | — | — |
| lsac | 0.0 | 5 | 0 | — | — | — | — |
| lsac | 0.0 | 10 | 0 | — | — | — | — |
| lsac | 0.0 | 20 | 0 | — | — | — | — |
| lsac | 0.1 | 5 | 0 | — | — | — | — |
| lsac | 0.1 | 10 | 0 | — | — | — | — |
| lsac | 0.1 | 20 | 0 | — | — | — | — |
| lsac | 0.2 | 5 | 0 | — | — | — | — |
| lsac | 0.2 | 10 | 0 | — | — | — | — |
| lsac | 0.2 | 20 | 0 | — | — | — | — |
| lsac | 0.3 | 5 | 0 | — | — | — | — |
| lsac | 0.3 | 10 | 0 | — | — | — | — |
| lsac | 0.3 | 20 | 0 | — | — | — | — |
| lsac | 0.4 | 5 | 0 | — | — | — | — |
| lsac | 0.4 | 10 | 0 | — | — | — | — |
| lsac | 0.4 | 20 | 0 | — | — | — | — |

## Paired Wilcoxon — K=5 vs K=10 (DRO, DP attack)

H1: k_alt > k_ref (alt K strictly raises the metric — DRO worse). * marks p<0.05.

_(no paired rows yet — need both k_alt and k_ref rows for the same seed)_

## Paired Wilcoxon — K=20 vs K=10 (DRO, DP attack)

H1: k_alt > k_ref (alt K strictly raises the metric — DRO worse). * marks p<0.05.

_(no paired rows yet — need both k_alt and k_ref rows for the same seed)_

## Verdict — does K_inner beyond 5 change anything materially?

Not yet answerable — need seed-paired K_alt and K=10 rows (currently INCOMPLETE).

## Wall-clock per config (DRO, mean over seeds)

| K_inner | mean wall/config (s) | n rows |
|---|---|---|
| 5 | 385.8 | 10 |
| 20 | 103.4 | 6 |
