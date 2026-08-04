# UTKFace Status (LOCKED)

**Updated:** 2026-08-04 (Aug 10 package)  
**Grid: COMPLETE 90/90 REAL** (local MPS). Summary: `results/utkface_summary.md`.

| Attack | Rows |
|--------|------|
| dp | 30/30 |
| if | 30/30 |
| combined | 30/30 |

- Features: `data/raw/utkface_features.npz` (N=23705, provenance REAL)  
- Output: `results/utkface_canonical.json`  
- Protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, 6 seeds, α∈{0,0.1,0.2,0.3,0.4}  
- **flair2:** PROVEN & PARKED (not used for this grid)  
- **Paper claim:** honest **mixed clean-test pilot** (significant DP wins mainly at high α; not Adult copy). See paper §results-utkface + `results/utkface_summary.md`.

See git history for earlier path notes (download, extract, runner).

