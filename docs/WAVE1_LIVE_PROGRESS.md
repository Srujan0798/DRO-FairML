# flair2 U1/U2 live progress (Grok lane)

_Last tick: 2026-08-05 ~01:59 IST_

## Counts (do not pkill these jobs)

| Job | File (flair2 `DRO-FairML-run/results/`) | Target | Count | Process | GPU |
|-----|------------------------------------------|--------|-------|---------|-----|
| **U1** CUDA repro | `utkface_flair2.json` | **90** | **12** | PID 3482009 **alive** | GPU0 ~637 MiB |
| **U2** 5-race | `utkface_multigroup.json` | **30** | **12** | PID 3482442 **alive** | GPU1 ~637 MiB |

- Puller: `scripts/u12_puller.sh` healthy (independent finalize + 10‑min partial rsync)
- ~15–18 min / attacked seed; GPU util often 0% (CPU-heavy features)

### Position
- **U1:** dp α=0.0+0.1 complete (12). Mid **dp α=0.2 seed=0** (~12 min into ~18 min cell).
- **U2:** α=0.0+0.1 complete (12). Running **α=0.2 seed=0**.

### ETA
- U1 remaining 78 → ~22 h
- U2 remaining 18 → ~5 h

## Signals

### U1 CUDA↔MPS
`results/utkface_repro_checkpoint_a01.md`: 12 cells, max|ΔDP_dro|=0.0072, **0 GAP**.

### U2 multi-group (`results/utkface_multigroup_summary.md`)
| α | n | wins_bin | wins_multi | mean Δmulti (N−D) |
|---:|--:|---------:|-----------:|------------------:|
| 0.0 | 6 | 3/6 | **6/6** | +0.0073 |
| 0.1 | 6 | **0/6** | **4/6** | +0.0037 |

Honest early read: multi-group advantage shrinks under attack (6/6→4/6); binary DP favors Naive at α=0.1. Gap still **Other-high** vs White/Black/Indian-low.

## U3 pixel PGD — path discovery (this tick)

Previously “no JPEGs under `/data/srujan.sai`”. **World-readable** UTKFace trees exist on flair2:

| Path | # JPEGs | Notes |
|------|--------:|-------|
| `/data/kshitish.madbhavi/UTKFace` | 23708 | `.chip.jpg` aligned; **readable** |
| `/data/kshitish.madbhavi/utkface_aligned_cropped/UTKFace` | 23708 | same |
| `/data/kshitish.madbhavi/kshitish/fl_fairness/data/UTKFace` | 23708 | same |

- Own dir `/data/srujan.sai/UTKFace` still **missing**; free space ~7 TB.
- Do **not** start U3 until U1/U2 free a GPU; prefer symlink/read-only use of shared path or local copy after advisor OK (shared-user data).
- Code ready: `src/corruption/image_pgd.py`.

## Tick history
- 01:30 U1=11 U2=10; provenance/flush
- 01:40 U1=11 U2=11; independent puller
- 01:49 U1=12 U2=11; repro checkpoint α≤0.1
- 01:59 U1=12 U2=12; U2 α=0.1 complete analysis; JPEG path discovery

## Open
- Finish U1=90 / U2=30
- U3: unblocked for *data access*, still blocked by GPU contention + greenlight/symlink
- Finding 3 cosine → paper integration
