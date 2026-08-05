# flair2 U1/U2/U3 live progress (Grok lane)

_Last tick: 2026-08-05 ~05:49 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **49** | if α=0.3 s0; if 0.0–0.2 done | 3482009 | yes GPU0 |
| **U2** | 30 | **30** | **COMPLETE** | — | done |
| **U3** | 24 | **4** | s0–1 both α; s2 α=0.1 running | 3507049 | yes GPU1 |

Puller OK (U3 tracking). ETA U1 ~1.6 h; U3 ~1.3 h @ ~3.5 min/cell.

## Signals
- Repro: **49** matched, **0 GAP**; IF-attack matched 19, max\|ΔIF\| **0.0014**
- U3 partial: α=0.1 n=2 multi-ish DRO 1/2; α=0.2 n=2 **Naive** both seeds (honest)
- Status: no false stall WARN on complete U2

## This tick
- Status script: COMPLETE line; include U3 json + process + log tail
- Pixel + repro summaries refreshed

## Open
- U1=90; U3=24; U2 done
