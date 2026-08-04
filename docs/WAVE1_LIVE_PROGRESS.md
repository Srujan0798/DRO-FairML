# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~03:41 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | PID | alive |
|-----|--------|------:|------|-----|-------|
| **U1** | 90 | **18** | dp α=0.2 done; **α=0.3 s=0 in progress** (json_age ~15 min) | 3482009 | yes (~3.6h, ~3200% CPU) |
| **U2** | 30 | **18** | α=0.2 done; log **RUN a=0.3 s=0** (json_age ~13 min) | 3482442 | yes (~3.6h, ~3150% CPU) |

Puller OK. ETA ~19.5 h (U1) / ~3.2 h (U2). Typical cell ~16 min — not stalled yet.

## Milestone (prior)
- CUDA dp α∈{0.0,0.1,0.2} full 18/18; α=0.2 max seed-wise |ΔDP| **0.0013**, **0 GAP**
- U2 α=0.2 multi **5/6**

## U3 prep (this tick)
- **JPEGs linked** (symlink only, no job start):
  - `/data/srujan.sai/UTKFace` → `/data/kshitish.madbhavi/UTKFace` (**23708** jpg)
  - run tree `data/raw/utkface/UTKFace` same target
- Launch still **blocked** while U1/U2 hold both GPUs (`launch_u3` refuses)
- Status script now reports `json_age`, stall WARN >25 min, U3 image readiness

## Open
- Finish U1=90 / U2=30; then U3 via `bash scripts/launch_u3_pixel_pgd.sh` on flair2
