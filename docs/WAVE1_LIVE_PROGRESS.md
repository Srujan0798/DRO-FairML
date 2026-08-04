# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~04:50 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **23** | dp α=0.3 seed **4** (~1026s); s5 in flight | 3482009 | yes (~4.8h) |
| **U2** | 30 | **23** | α=0.3 seed **4**; log **RUN a=0.3 s=5** | 3482442 | yes (~4.8h) |

Puller OK. ETA ~18.6 h (U1) / ~1.9 h (U2). U3 JPEGs ready (23708).

## Signals
- Repro: **23** matched, **0 GAP**; max\|ΔDP clean\| **0.0072** (outlier: dp α=0.1 s1 gpu−mac); rest mostly ≤0.001
- U2 α=0.3 n=5 multi wins **4/1/5** (DRO/tie/n) — s3 still **tie**; s4 multi DRO but bin Naive
- U1 stdout log still **0 bytes** (redirect/buffering ghost); JSON is SoT — do not restart

## This tick
- Rsync partials U1=23 U2=23; re-summarize both
- Repro summarizer: no more `+nan` for missing cells (—); top seed-wise Δ list
- Status script notes empty U1 log honestly

## Open
- U1=90 / U2=30; U3 after free GPU (do not launch until U1/U2 done or one GPU free without killing)
