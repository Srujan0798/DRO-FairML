# flair2 U1/U2/U3 live progress (Grok lane)

_Last tick: 2026-08-05 ~05:20 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **35** | **dp 30/30 done**; if α=0.0 seeds 0–4; s5+ | 3482009 | yes (~5.2h) GPU0 |
| **U2** | 30 | **30** | **COMPLETE** α=0.0–0.4 × 6 seeds | — | done |
| **U3** | 24 | **0** (started) | seed=0 α=0.1 loading/training | 3504795 | yes GPU1 |

Puller: **U2 FINALIZED**. ETA U1 ~ remaining if+combined; α=0.0 if cells ~fast.

## Signals
- Repro: **35** matched, **0 GAP**; max\|ΔDP clean\| **0.0072** (dp α=0.1 s1)
- U2 multi wins by α: 6/4/5/5/6 of 6 (α=0.0–0.4); α=0.3 one multi **tie**; bin α=0.1 **0/6** DRO
- U3: 23704 JPEGs; concurrent on free GPU1 while U1 holds GPU0

## This tick
- **U2 COMPLETE** 30/30 — summary finalized (puller + Mac)
- Status ETA: last-k cell times; stall thr scales with pace
- Launch U3: allow free-GPU while U1 alive; CSV parse fix; started PID 3504795

## Open
- U1=90; U3=24; U2 done (human review before paper)
