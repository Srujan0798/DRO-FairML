# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~03:50 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **19** | dp **α=0.3 seed=0** done; s=1 running | 3482009 | yes (~3.8h) |
| **U2** | 30 | **19** | α=0.3 seed=0 done; log **RUN a=0.3 s=1** | 3482442 | yes (~3.7h) |

Puller OK (count-up summarize). ETA ~19.4 h (U1) / ~3.0 h (U2). U3 JPEGs ready (23708).

## Signals
- Repro: **19** matched, max\|ΔDP\|0.0072, **0 GAP**
- α=0.3 s0 seed-matched: clean ΔDP **−0.0005**, corr **−0.0024** (OK)
- U2 α=0.3 s0: multi win, binary win

## Fix this tick
- `summarize_utkface_repro.py`: partial cells used **all 6 Mac seeds** vs incomplete GPU → false large Δ (e.g. corr +0.024). Now **seed-matched** Mac only.

## Open
- U1=90 / U2=30; U3 after free GPU
