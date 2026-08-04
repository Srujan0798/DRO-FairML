# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~02:19 IST_

## Counts (do not pkill)

| Job | Target | Count | Last | Alive |
|-----|--------|------:|------|-------|
| **U1** | 90 | **14** | dp α=0.2 seed=1 (t≈538s) | PID 3482009 GPU0 |
| **U2** | 30 | **14** | α=0.2 seed=1; mid seed=2 | PID 3482442 GPU1 |

Puller OK. ETA ~20h (U1) / ~4.2h (U2). Seed time variance ~540–1100s normal.

## Signals
- U1 repro: **14** matched, max|ΔDP|=0.0072, **0 GAP**
- U2 α=0.2 n=2: multi wins 2/2; multi DP ~0.22 (vs ~0.13 at α≤0.1)

## U3 prep (not started)
- `experiments/run_utkface_pixel_pgd.py` restored (τ=1, race-binary, resume-safe)
- Image link: `CONFIRM=1 bash scripts/flair2_link_utkface_images.sh` when GPU free
- Do not launch until U1 or U2 frees a GPU

## Open
- U1=90 / U2=30 → finalize summaries
- U3 after GPU free
