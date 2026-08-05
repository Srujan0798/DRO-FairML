# UTKFace Status

**Updated:** 2026-08-05 ~23:05 IST (Grok lane)

## Mac MPS canonical (LOCKED — paper pilot)
**Grid: COMPLETE 90/90 REAL.** `results/utkface_canonical.json` (**do not rewrite**).

## flair2 CUDA (Grok lane — COMPLETE)
| Job | File | Target | Status |
|-----|------|--------|--------|
| **U1** CUDA repro | `results/utkface_flair2.json` | 90 | **COMPLETE 90/90** — repro summary OK (max ΔDP_clean 0.0072) |
| **U2** 5-race multi | `results/utkface_multigroup.json` | 30 | **COMPLETE 30/30** |
| **U3** pixel PGD | `results/utkface_pixel_pgd.json` | 12 | **COMPLETE 12/12** (α∈{0.1,0.2} × 6 seeds; intentional scope) |

### U3 notes
- Protocol: pixel PGD ε=4/255, steps=10 on raw JPEGs; clean-test DP/IF/acc
- OOM fixed earlier: pixels on CPU; batched GPU features/PGD
- Summary: `results/pixel_pgd_summary.md` — DP mixed; IF 6/6 DRO both α
- Clean-test DP after pixel train attack is **not** comparable 1:1 to U1 feature FairnessTargetedPGD corrupted-test DP

### U2 headline (complete)
Multi DRO wins 6/4/5/5/6 by α. Summary: `results/utkface_multigroup_summary.md`

### U1 headline (complete)
Mac MPS vs flair2 CUDA seed-matched; all cells OK. Summary: `results/utkface_reproducibility_summary.md`

- Live log: `docs/WAVE1_LIVE_PROGRESS.md`
