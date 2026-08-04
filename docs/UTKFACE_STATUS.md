# UTKFace Status

**Updated:** 2026-08-05 (Grok lane live)  

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
| **U1** CUDA repro | `results/utkface_flair2.json` | 90 | **in progress** — see `docs/WAVE1_LIVE_PROGRESS.md` |
| **U2** 5-race multi | `results/utkface_multigroup.json` | 30 | **in progress** |
| **U3** pixel PGD | `results/utkface_pixel_pgd.json` | 24 (planned) | prepared, not started |

- Repro partial: matched cells so far max\|ΔDP_dro\|≈0.007 (0 GAP). Summary: `results/utkface_reproducibility_summary.md`
- Multi-group partial: `results/utkface_multigroup_summary.md`
- U3: images via `scripts/flair2_link_utkface_images.sh`; runner `experiments/run_utkface_pixel_pgd.py`

See git history for earlier path notes (download, extract, runner).

