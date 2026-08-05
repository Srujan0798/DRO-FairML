# UTKFace Status

**Updated:** 2026-08-05 ~05:41 IST (Grok lane)  

## Mac MPS canonical (LOCKED — paper pilot)
**Grid: COMPLETE 90/90 REAL.** `results/utkface_canonical.json` (**do not rewrite**).

## flair2 CUDA (LIVE — Grok lane)
| Job | File | Target | Status |
|-----|------|--------|--------|
| **U1** CUDA repro | `results/utkface_flair2.json` | 90 | **~45/90** — dp done; if mid α=0.2 |
| **U2** 5-race multi | `results/utkface_multigroup.json` | 30 | **COMPLETE 30/30** |
| **U3** pixel PGD | `results/utkface_pixel_pgd.json` | 24 | **2/24** running (~4 min/cell) after OOM fix |

### U3 notes
- OOM fixed: pixels on CPU; batched GPU features/PGD
- Summary: `results/pixel_pgd_summary.md` (partial; not for paper until 24/24)
- Clean-test DP after pixel train attack is **not** comparable 1:1 to U1 feature FairnessTargetedPGD corrupted-test DP

### U2 headline (complete)
Multi DRO wins 6/4/5/5/6 by α. Summary: `results/utkface_multigroup_summary.md`

- Live: `docs/WAVE1_LIVE_PROGRESS.md`
