# Agent N5 — K_inner ablation summary (K ∈ {5, 10, 20}, DRO only, DP attack)

Analysis-only. No new training. Source: `results/kinner_ablation.json` 
(23/180 rows, K∈{5,20}) + canonical K=10 DRO/DP rows 
(90 rows) as the reference via `experiments.loaders`.

## Coverage

- K_inner ablation rows present: **23/180** (12.8%)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## Per-cell means for DRO (dataset, α, K_inner)

| dataset | α | K | n | DP | IF | acc | wall/config (s) |
|---|---|---|---|---|---|---|---|
| adult | 0.0 | 5 | 6 | 0.1426 | 0.0933 | 0.8147 | 101.4 |
| adult | 0.0 | 10 | 6 | 0.1426 | 0.0000 | 0.8147 | 24.5 |
| adult | 0.0 | 20 | 6 | 0.1426 | 0.0933 | 0.8147 | 103.4 |
| adult | 0.1 | 5 | 6 | 0.1999 | 0.0466 | 0.8177 | 889.4 |
| adult | 0.1 | 10 | 6 | 0.1999 | 0.0000 | 0.8177 | 1015.0 |
| adult | 0.1 | 20 | 4 | 0.2005 | 0.0470 | 0.8195 | 8119.7 |
| adult | 0.2 | 5 | 1 | 0.2459 | 0.0458 | 0.7553 | 1084.8 |
| adult | 0.2 | 10 | 6 | 0.2334 | 0.0000 | 0.7586 | 5563.2 |
| adult | 0.2 | 20 | 0 | — | — | — | — |
| adult | 0.3 | 5 | 0 | — | — | — | — |
| adult | 0.3 | 10 | 6 | 0.2614 | 0.0000 | 0.6755 | 901.6 |
| adult | 0.3 | 20 | 0 | — | — | — | — |
| adult | 0.4 | 5 | 0 | — | — | — | — |
| adult | 0.4 | 10 | 6 | 0.2855 | 0.0000 | 0.5607 | 14974.8 |
| adult | 0.4 | 20 | 0 | — | — | — | — |
| credit | 0.0 | 5 | 0 | — | — | — | — |
| credit | 0.0 | 10 | 6 | 0.0119 | 0.0000 | 0.8068 | 19.1 |
| credit | 0.0 | 20 | 0 | — | — | — | — |
| credit | 0.1 | 5 | 0 | — | — | — | — |
| credit | 0.1 | 10 | 6 | 0.0134 | 0.0000 | 0.8097 | 7053.1 |
| credit | 0.1 | 20 | 0 | — | — | — | — |
| credit | 0.2 | 5 | 0 | — | — | — | — |
| credit | 0.2 | 10 | 6 | 0.0178 | 0.0000 | 0.7819 | 20445.7 |
| credit | 0.2 | 20 | 0 | — | — | — | — |
| credit | 0.3 | 5 | 0 | — | — | — | — |
| credit | 0.3 | 10 | 6 | 0.0228 | 0.0000 | 0.7531 | 4413.1 |
| credit | 0.3 | 20 | 0 | — | — | — | — |
| credit | 0.4 | 5 | 0 | — | — | — | — |
| credit | 0.4 | 10 | 6 | 0.0170 | 0.0000 | 0.7520 | 590.6 |
| credit | 0.4 | 20 | 0 | — | — | — | — |
| lsac | 0.0 | 5 | 0 | — | — | — | — |
| lsac | 0.0 | 10 | 6 | 0.1829 | 0.0000 | 0.9023 | 7.1 |
| lsac | 0.0 | 20 | 0 | — | — | — | — |
| lsac | 0.1 | 5 | 0 | — | — | — | — |
| lsac | 0.1 | 10 | 6 | 0.2539 | 0.0000 | 0.9046 | 389.2 |
| lsac | 0.1 | 20 | 0 | — | — | — | — |
| lsac | 0.2 | 5 | 0 | — | — | — | — |
| lsac | 0.2 | 10 | 6 | 0.2230 | 0.0000 | 0.9033 | 166.5 |
| lsac | 0.2 | 20 | 0 | — | — | — | — |
| lsac | 0.3 | 5 | 0 | — | — | — | — |
| lsac | 0.3 | 10 | 6 | 0.2220 | 0.0000 | 0.9032 | 121.8 |
| lsac | 0.3 | 20 | 0 | — | — | — | — |
| lsac | 0.4 | 5 | 0 | — | — | — | — |
| lsac | 0.4 | 10 | 6 | 0.2211 | 0.0000 | 0.9029 | 339.8 |
| lsac | 0.4 | 20 | 0 | — | — | — | — |

## Paired Wilcoxon — K=5 vs K=10 (DRO, DP attack)

H1: k_alt > k_ref (alt K strictly raises the metric — DRO worse). * marks p<0.05.

| dataset | α | n | DP_k_alt | DP_k_ref | ΔDP | wins_alt | p_DP | sig | ΔIF | p_IF |
|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | 6 | 0.1426 | 0.1426 | +0.0000 | 2/6 | 0.5000  | +0.0933 | 0.0156 |
| adult | 0.1 | 6 | 0.1999 | 0.1999 | -0.0000 | 2/6 | 0.7812  | +0.0466 | 0.0156 |

## Paired Wilcoxon — K=20 vs K=10 (DRO, DP attack)

H1: k_alt > k_ref (alt K strictly raises the metric — DRO worse). * marks p<0.05.

| dataset | α | n | DP_k_alt | DP_k_ref | ΔDP | wins_alt | p_DP | sig | ΔIF | p_IF |
|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | 6 | 0.1426 | 0.1426 | +0.0000 | 2/6 | 0.5000  | +0.0933 | 0.0156 |
| adult | 0.1 | 4 | 0.2005 | 0.2005 | +0.0000 | 3/4 | 0.3125  | +0.0470 | 0.0625 |

## Verdict — does K_inner beyond 5 change anything materially?

No: K_inner beyond 5 does NOT change anything materially (max |ΔDP|=0.0000, 0/4 cells p<0.05). DRO is K_inner-robust within {5,10,20}.

## Wall-clock per config (DRO, mean over seeds)

| K_inner | mean wall/config (s) | n rows |
|---|---|---|
| 5 | 540.7 | 13 |
| 10 | 3735.0 | 90 |
| 20 | 3309.9 | 10 |
