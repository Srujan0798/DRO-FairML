# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~03:22 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **17** | dp α=0.2 seed=4 done; **seed=5 in progress** (~10+ min) | 3482009 | yes (~3.3h, ~3200% CPU) |
| **U2** | 30 | **17** | α=0.2 seed=4 done; **seed=5 in log** | 3482442 | yes (~3.3h, ~3150% CPU) |

GPU: both L40S ~637 MiB, util often 0% during CPU-heavy kNN/attack stretches (CUDA still reserved). Do not kill.

Puller: restarted with last-cell logging (PID on Mac). ETA ~19.6 h (U1) / ~3.5 h (U2).

## Signals
- Repro: **17** matched, max\|ΔDP\| **0.0072**, **0 GAP**
- U2 α=0.2 n=5: multi wins **4/5**

## Ops
- U1 log empty: flair2 copy of `run_utkface_server.py` had **no `flush=True`** (stdout fully buffered under nohup). Disk file re-synced with Mac flush fix for **future** runs; **current U1 process unchanged** — monitor via JSON mtime / puller.
- One-shot status: `bash scripts/flair2_u12_status.sh`

## Open
- Finish U1=90 / U2=30; U3 after free GPU (+ JPEGs); launch_u3 refuses while U1/U2 alive
