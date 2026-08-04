# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~04:00 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **20** | dp α=0.3 seeds 0–1 done; s=2 running | 3482009 | yes (~3.9h) |
| **U2** | 30 | **20** | α=0.3 seeds 0–1 done; log **RUN a=0.3 s=2** | 3482442 | yes (~3.9h) |

Puller OK. ETA ~18.9 h (U1) / ~2.7 h (U2). U3 JPEGs ready (23708).

## Signals
- Repro: **20** matched, **0 GAP**; α=0.3 s0–1 seed-matched |ΔDP clean| ≤ 0.0006
- U2 α=0.3 s1: multi win (N=0.247 / D=0.234)

## Hygiene this tick
- Untrack `logs/wave1_nohup.out` (~82 MB) + other `logs/*.out`; gitignore `logs/*.out`

## Open
- U1=90 / U2=30; U3 after free GPU
