# UTKFace Status

**Updated:** 2026-08-05 ~05:34 IST (Grok lane)  

## Mac MPS canonical (LOCKED — paper pilot)
**Grid: COMPLETE 90/90 REAL.** `results/utkface_canonical.json` (**do not rewrite**).

## flair2 CUDA (LIVE — Grok lane)
| Job | File | Target | Status |
|-----|------|--------|--------|
| **U1** CUDA repro | `results/utkface_flair2.json` | 90 | **42/90** — dp done; if α=0.0–0.1 done |
| **U2** 5-race multi | `results/utkface_multigroup.json` | 30 | **COMPLETE 30/30** |
| **U3** pixel PGD | `results/utkface_pixel_pgd.json` | 24 | **RUNNING** GPU1 after OOM fix |

### U3 bug (fixed this tick)
Full train-set pixel tensors were moved to CUDA → **OOM ~40 GiB**. Runner now keeps pixels on **CPU** and batches normalize/features/PGD on GPU. Relaunched PID 3507049.

### U2 headline (complete)
Multi DRO wins 6/4/5/5/6 by α; α=0.3 multi 5/1/6 (one tie). Binary weaker at α=0.1 (0/6).  
Summary: `results/utkface_multigroup_summary.md`

- Repro: **42** matched, **0 GAP**. `results/utkface_reproducibility_summary.md`
- Live: `docs/WAVE1_LIVE_PROGRESS.md`
