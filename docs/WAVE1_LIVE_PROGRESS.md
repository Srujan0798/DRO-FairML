# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~03:29 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **18** | **dp α=0.2 complete 6/6**; next α=0.3 | 3482009 | yes (~3.4h) |
| **U2** | 30 | **18** | **α=0.2 complete 6/6**; log shows **α=0.3 s=0** | 3482442 | yes (~3.4h) |

GPU: ~650 MiB each L40S. Puller OK (count-up summarize). ETA ~19.5 h (U1) / ~3.2 h (U2).

## Milestone — first full α tier on CUDA
- **U1 dp α∈{0.0,0.1,0.2}**: all 18 cells matched Mac; α=0.2 seed-wise max\|ΔDP_dro\| = **0.0013** (mean +0.0003)
- Overall matched max\|ΔDP\| still 0.0072 (from earlier α); **0 GAP** (thr=0.02)
- **U2 α=0.2**: multi wins **5/6** (seed 4 multi loss); binary DP wins only **2/6** (honest mixed)

## Signals
- Repro summary: `results/utkface_reproducibility_summary.md` (18/90)
- Multi summary: `results/utkface_multigroup_summary.md` (18/30)

## Ops
- U1 log still empty (running process pre-flush fix); U2 log fine
- Status: `bash scripts/flair2_u12_status.sh`

## Open
- U1 remaining: dp α=0.3–0.4 (12) + if (30) + combined (30) = 72
- U2 remaining: α=0.3–0.4 (12)
- U3 after free GPU + JPEGs
