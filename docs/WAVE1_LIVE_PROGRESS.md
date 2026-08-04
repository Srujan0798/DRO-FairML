# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~03:10 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **17** | dp α=0.2 seed=4 done; seed=5 running | 3482009 | yes (~3.1h elapsed) |
| **U2** | 30 | **16** | α=0.2 seeds 0–3 done; seed=4 in log | 3482442 | yes (~3.1h elapsed) |

GPU: both L40S ~637 MiB each (feature-space PGD; CPU-heavy kNN stretches). Do not kill.

Puller PID alive (poll 120s). ETA ~20 h (U1, ~73 left × ~16 min) / ~3.6 h (U2, ~14 left).

## Signals
- Repro: **17** matched, max\|ΔDP_dro clean\| **0.0072**, **0 GAP**
- α=0.2 seeds 0–4: seed-wise |ΔDP| ≤ 0.0013 (s0 −0.0002 … s4 −0.0003)
- U2 α=0.2 n=4: multi wins **4/4** (summary unchanged until seed 4 lands)

## Ops note
- `logs/u1_utkface_flair2.log` is **empty** (stdout redirected but 0 B) — trust JSON mtime / puller counts, not that log. U2 log is fine.
- Monitor: `ssh flair2 'python3 -c "…len(json)…"'` or Mac `logs/u12_puller.log`

## Open
- Finish U1=90 / U2=30; U3 after free GPU (needs JPEGs + launch_u3)
