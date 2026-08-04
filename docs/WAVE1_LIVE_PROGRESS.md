# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~04:20 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **21** | dp α=0.3 seeds 0–2; **s=3 running** | 3482009 | yes (~4.3h) |
| **U2** | 30 | **21** | α=0.3 seeds 0–2; log **RUN a=0.3 s=3** | 3482442 | yes (~4.2h) |

Puller OK. ETA ~18.8 h (U1) / ~2.5 h (U2). U3 JPEGs ready.

## Signals
- Repro: **21** matched, **0 GAP**; α=0.3 s0–2 max\|ΔDP clean\| **0.0006**
- U2 α=0.3 n=3: multi **3/3**, binary **3/3**
- max\|ΔDP corrupted\| overall **0.008** (still ≪0.02 thr)

## This tick
- Atomic writes also on `run_utkface.py` (base runner)
- Partials + summaries refreshed

## Open
- U1=90 / U2=30; U3 after free GPU
