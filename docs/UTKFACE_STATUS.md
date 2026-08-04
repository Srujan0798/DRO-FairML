# UTKFace Status

**Updated:** 2026-08-05 ~05:20 IST (Grok lane)  

## Mac MPS canonical (LOCKED — paper pilot)
**Grid: COMPLETE 90/90 REAL.** Summary: `results/utkface_summary.md`.

| Attack | Rows |
|--------|------|
| dp | 30/30 |
| if | 30/30 |
| combined | 30/30 |

- Features: `data/raw/utkface_features.npz` (N=23705, provenance REAL)  
- Output: `results/utkface_canonical.json` (**do not rewrite**)  
- Protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, 6 seeds, α∈{0,0.1,0.2,0.3,0.4}  
- **Paper claim:** honest **mixed clean-test pilot** (significant DP wins mainly at high α; not Adult copy).

## flair2 CUDA (LIVE — Grok lane)
| Job | File | Target | Status |
|-----|------|--------|--------|
| **U1** CUDA repro | `results/utkface_flair2.json` | 90 | **35/90** — dp **30/30**; if α=0.0 in progress |
| **U2** 5-race multi | `results/utkface_multigroup.json` | 30 | **COMPLETE 30/30** — summary ready |
| **U3** pixel PGD | `results/utkface_pixel_pgd.json` | 24 | **RUNNING** GPU1 (PID 3504795); seed0 α=0.1 |

### U2 headline (complete — not yet paper-integrated)
| α | multi wins (D/tie/n) | mean Δmulti (N−D) | bin wins |
|---|---------------------:|------------------:|---------:|
| 0.0 | 6/0/6 | +0.0073 | 3/0/6 |
| 0.1 | 4/0/6 | +0.0037 | 0/0/6 |
| 0.2 | 5/0/6 | +0.0038 | 2/0/6 |
| 0.3 | 5/1/6 | +0.0049 | 4/0/6 |
| 0.4 | 6/0/6 | +0.0136 | 6/0/6 |

Multi max-min DRO advantage holds; binary White/non-White is weaker (especially α=0.1).  
At high α, gap driven by **White** (min ~0.32) vs **Asian/Other** (max ~0.58).

- Repro: **35** matched, **0 GAP** thr=0.02. Summary: `results/utkface_reproducibility_summary.md`
- Multi-group: `results/utkface_multigroup_summary.md` (**30/30**)
- Live table: `docs/WAVE1_LIVE_PROGRESS.md`
