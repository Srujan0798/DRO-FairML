# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~02:10 IST_

## Counts (do not pkill)

| Job | Target | Count | Status |
|-----|--------|------:|--------|
| **U1** `utkface_flair2.json` | 90 | **13** | alive PID 3482009 GPU0; last **dp α=0.2 seed=0**; mid seed=1 |
| **U2** `utkface_multigroup.json` | 30 | **13** | alive PID 3482442 GPU1; last **α=0.2 seed=0**; mid seed=1 |

Puller `scripts/u12_puller.sh` OK. ETA ~21 h (U1) / ~4.7 h (U2).

## Signals
- **U1 repro:** 13 matched cells, max|ΔDP_dro|=0.0072, **0 GAP** (incl. first α=0.2 seed).
- **U2:** α≤0.1 complete; α=0.2 seed0 multi DP ~0.22 (jump from ~0.13) — stronger attack, not a stall.
  - α=0.0 multi wins 6/6; α=0.1 multi 4/6, binary 0/6.

## U3 prep
- JPEG source dry-run: `scripts/flair2_link_utkface_images.sh` (CONFIRM=1 to symlink)
- Selected: `/data/kshitish.madbhavi/UTKFace` (23708 jpg, world-readable)
- Do **not** start U3 until a GPU free after U1/U2; shared-user data — symlink only after OK

## Open
- U1=90 / U2=30; then U3 optional; Finding 3 paper integration
