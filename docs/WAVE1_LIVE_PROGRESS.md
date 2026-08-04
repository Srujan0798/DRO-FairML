# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~02:40 IST_

## Counts (do not pkill)

| Job | Target | Count | Last |
|-----|--------|------:|------|
| **U1** | 90 | **15** | dp α=0.2 seed=2 done; mid seed=3 |
| **U2** | 30 | **15** | α=0.2 seed=2 done; mid seed=3 |

Jobs alive GPU0/1. Puller restarted with durable multigroup summarizer.

## Signals
- Repro: 15 matched, max|ΔDP clean|≈0.007, **0 GAP**
- U2 α=0.2 n=3: multi wins **3/3**, mean multi ~0.22 (vs ~0.13 at α≤0.1)

## Docs
- Finding 3 cosine≠Euclidean now in **paper + report**
- `experiments/summarize_utkface_multigroup.py` for U2 finalize

## Open
- U1=90 / U2=30; U3 after free GPU
