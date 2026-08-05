# flair2 U1/U2/U3 live progress (Grok lane)

_Last tick: 2026-08-05 ~05:34 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **42** | dp **30/30**; if α=0.0–0.1 **12/12**; if α=0.2 next | 3482009 | yes GPU0 |
| **U2** | 30 | **30** | **COMPLETE** | — | done |
| **U3** | 24 | **0** (running) | seed=0 α=0.1 (post-OOM fix) | 3507049 | yes GPU1 ~693MiB |

Puller OK (U2 finalized). U1 if cells ~2 min → if+combined remaining ~48 cells ~1–2 h at current pace.

## Signals
- Repro: **42** matched, **0 GAP**; max\|ΔDP clean\| **0.0072**
- U3 first launch **OOM** (full train pixels on GPU ~40 GiB) — fixed: CPU pixels + batched GPU features/PGD; smoke 512 OK; relaunched

## This tick
- Diagnosed U3 CUDA OOM at `normalize(X_attacked_pix)` full-tensor
- Fixed `run_utkface_pixel_pgd.py` memory path; BCE batch-1 shape fix
- Relaunched U3 on GPU1 concurrent with U1

## Open
- U1=90; U3=24; U2 done
