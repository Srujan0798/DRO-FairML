# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~04:10 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **20** | dp α=0.3 s0–1; **s=2 in progress** (~13 min) | 3482009 | yes (~4.1h) |
| **U2** | 30 | **20** | α=0.3 s0–1; **s=2 in log** (~10 min) | 3482442 | yes (~4.1h) |

Puller OK. ETA ~18.9 h (U1) / ~2.7 h (U2). U3 JPEGs ready.

## This tick
- **Atomic JSON writes** in U1/U2/U3 runners (`tmp` + `os.replace`) — disk on flair2 updated; **running U1/U2 still use old code in memory** (next resume/restart benefits)
- Status ETA prefers α≥0.3 timings; shows next-row ETA mid-cell
- Repro still 20 matched, 0 GAP (Mac partials unchanged this tick)

## Open
- U1=90 / U2=30; U3 after free GPU
