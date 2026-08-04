# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~05:04 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **25** | dp **α=0.4 s0** done (α=0.3 **6/6**); s1 in flight | 3482009 | yes (~5.0h) |
| **U2** | 30 | **24** | α=0.3 **6/6**; log **RUN a=0.4 s=0** | 3482442 | yes (~5.0h) |

Puller OK. ETA ~15.5 h (U1) / ~1.6 h (U2). U3 JPEGs ready (23708).

## Signals
- Repro: **25** matched, **0 GAP**; max\|ΔDP clean\| **0.0072** (dp α=0.1 s1); α=0.3 full OK (mean Δ +0.0004)
- U2 α=0.3 complete multi **5/1/6** (DRO/tie/n); bin↔multi agreement **4/6** (s3,s4 disagree)
- Jobs healthy: multi-core CPU ~3200%, GPU util bursts; U1 log still empty (JSON SoT)

## This tick
- Caught α=0.3 completion both jobs + U1 first α=0.4 cell
- Multigroup summarizer: bin↔multi winner agreement line
- Refreshed UTKFACE_STATUS + WAVE1

## Open
- U1=90 / U2=30; U3 after free GPU
