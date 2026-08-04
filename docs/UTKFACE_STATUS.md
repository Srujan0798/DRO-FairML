# UTKFace Status

**Updated:** 2026-08-05 ~05:04 IST (Grok lane)  

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
| **U1** CUDA repro | `results/utkface_flair2.json` | 90 | **25/90** — dp α=0.0–0.3 complete; α=0.4 seed 0 done, s1+ running |
| **U2** 5-race multi | `results/utkface_multigroup.json` | 30 | **24/30** — α=0.0–0.3 complete; α=0.4 s0 running |
| **U3** pixel PGD | `results/utkface_pixel_pgd.json` | 24 (planned) | **JPEGs ready** (symlink 23708); launch after U1/U2 free GPU |

- Repro: **25** matched, max\|ΔDP clean\| **0.0072** (outlier dp α=0.1 s1); **0 GAP** thr=0.02. Summary: `results/utkface_reproducibility_summary.md`
- Multi-group: α=0.3 multi **5/1/6** DRO/tie/n; bin↔multi agreement 4/6. Summary: `results/utkface_multigroup_summary.md`
- U1 log empty (known); JSON SoT — **do not restart**
- U3: linked `/data/srujan.sai/UTKFace` → kshitish tree; do not launch until U1/U2 done
- Live table: `docs/WAVE1_LIVE_PROGRESS.md`
