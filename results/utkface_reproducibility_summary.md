# UTKFace reproducibility: Mac MPS vs flair2 CUDA

- Mac rows: **90/90** (`results/utkface_canonical.json`)
- flair2 rows: **20/90** (`results/utkface_flair2.json`)

Protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, n_seeds=6, REAL features.
Same seeds 0–5. Large gaps are bugs to investigate.
Cell means use **seed-matched** Mac rows only when GPU is partial.

## Clean test (primary)

| attack | α | n_mac | n_gpu | Δ mean DP_dro (gpu−mac) | Δ mean acc_dro | note |
|--------|---:|------:|------:|------------------------:|---------------:|------|
| combined | 0.0 | 6 | 0 | +nan | +nan | partial |
| combined | 0.1 | 6 | 0 | +nan | +nan | partial |
| combined | 0.2 | 6 | 0 | +nan | +nan | partial |
| combined | 0.3 | 6 | 0 | +nan | +nan | partial |
| combined | 0.4 | 6 | 0 | +nan | +nan | partial |
| dp | 0.0 | 6 | 6 | +0.0002 | +0.0001 | OK |
| dp | 0.1 | 6 | 6 | +0.0014 | -0.0001 | OK |
| dp | 0.2 | 6 | 6 | +0.0003 | -0.0005 | OK |
| dp | 0.3 | 6 | 2 | +0.0000 | -0.0015 | OK |
| dp | 0.4 | 6 | 0 | +nan | +nan | partial |
| if | 0.0 | 6 | 0 | +nan | +nan | partial |
| if | 0.1 | 6 | 0 | +nan | +nan | partial |
| if | 0.2 | 6 | 0 | +nan | +nan | partial |
| if | 0.3 | 6 | 0 | +nan | +nan | partial |
| if | 0.4 | 6 | 0 | +nan | +nan | partial |

## Corrupted (attacked) test

| attack | α | n_gpu | Δ mean DP_dro_corr | Δ mean acc_dro_corr | note |
|--------|---:|------:|-------------------:|--------------------:|------|
| dp | 0.0 | 6 | +0.0002 | +0.0001 | OK |
| dp | 0.1 | 6 | +0.0010 | +0.0001 | OK |
| dp | 0.2 | 6 | -0.0006 | -0.0007 | OK |
| dp | 0.3 | 2 | -0.0010 | -0.0020 | OK |

## Matched seed-wise (all completed GPU cells)
- Matched cells: **20**
- max\|Δ DP_dro clean\| = **0.0072**
- max\|Δ DP_dro corrupted\| = **0.0064**
- mean Δ DP_dro clean = +0.00058

### Verdict
- Grid complete on both: **False**
- GAP cells (clean, n≥6 both sides, thr=0.02): **0**
- If any cell is GAP with n=6 both sides, investigate device/nondeterminism before claiming CUDA repro.

