# flair2 U1/U2/U3 live progress (Grok lane)

_Last tick: 2026-08-05 ~05:58 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **53** | if α=0.3 s4 (s5 + α=0.4 next) | 3482009 | yes GPU0 |
| **U2** | 30 | **30** | **COMPLETE** | — | done |
| **U3** | 24 | **7** | s3 α=0.1 done; α=0.2 running | 3507049 | yes GPU1 |

Puller OK. ETA U1 ~1.5 h; U3 ~1.0 h.

## Signals
- Repro: **53** matched, **0 GAP**; by attack max|Δ| dp=0.0072 / if=0.0014
- U3: α=0.1 DP **3/4** DRO, IF **4/4** DRO; α=0.2 DP **1/3** DRO but IF **3/3** DRO (mixed DP, strong IF)

## This tick
- Pixel summary: separate **wins_DP** / **wins_IF** columns
- Repro: per-attack max/mean |ΔDP_clean|

## Open
- U1=90 (if remaining + combined 30); U3=24; U2 done
