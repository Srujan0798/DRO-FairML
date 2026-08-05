# UTKFace reproducibility: Mac MPS vs flair2 CUDA

- Mac rows: **90/90** (`results/utkface_canonical.json`)
- flair2 rows: **90/90** (`results/utkface_flair2.json`)

Protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, n_seeds=6, REAL features.
Same seeds 0–5. Large gaps are bugs to investigate.
Cell means use **seed-matched** Mac rows only when GPU is partial.

## Clean test (primary)

| attack | α | n_mac | n_gpu | Δ mean DP_dro (gpu−mac) | Δ mean acc_dro | note |
|--------|---:|------:|------:|------------------------:|---------------:|------|
| combined | 0.0 | 6 | 6 | +0.0002 | +0.0001 | OK |
| combined | 0.1 | 6 | 6 | -0.0000 | -0.0006 | OK |
| combined | 0.2 | 6 | 6 | +0.0005 | -0.0007 | OK |
| combined | 0.3 | 6 | 6 | +0.0005 | -0.0005 | OK |
| combined | 0.4 | 6 | 6 | +0.0009 | +0.0000 | OK |
| dp | 0.0 | 6 | 6 | +0.0002 | +0.0001 | OK |
| dp | 0.1 | 6 | 6 | +0.0014 | -0.0001 | OK |
| dp | 0.2 | 6 | 6 | +0.0003 | -0.0005 | OK |
| dp | 0.3 | 6 | 6 | +0.0004 | -0.0005 | OK |
| dp | 0.4 | 6 | 6 | +0.0004 | -0.0005 | OK |
| if | 0.0 | 6 | 6 | +0.0002 | +0.0001 | OK |
| if | 0.1 | 6 | 6 | -0.0001 | +0.0000 | OK |
| if | 0.2 | 6 | 6 | +0.0002 | +0.0000 | OK |
| if | 0.3 | 6 | 6 | -0.0004 | +0.0007 | OK |
| if | 0.4 | 6 | 6 | -0.0005 | -0.0005 | OK |

## Corrupted (attacked) test

| attack | α | n_gpu | Δ mean DP_dro_corr | Δ mean acc_dro_corr | note |
|--------|---:|------:|-------------------:|--------------------:|------|
| combined | 0.0 | 6 | +0.0002 | +0.0001 | OK |
| combined | 0.1 | 6 | -0.0002 | -0.0012 | OK |
| combined | 0.2 | 6 | +0.0006 | +0.0000 | OK |
| combined | 0.3 | 6 | -0.0004 | -0.0009 | OK |
| combined | 0.4 | 6 | -0.0010 | -0.0004 | OK |
| dp | 0.0 | 6 | +0.0002 | +0.0001 | OK |
| dp | 0.1 | 6 | +0.0010 | +0.0001 | OK |
| dp | 0.2 | 6 | -0.0006 | -0.0007 | OK |
| dp | 0.3 | 6 | -0.0010 | -0.0016 | OK |
| dp | 0.4 | 6 | -0.0013 | -0.0009 | OK |
| if | 0.0 | 6 | +0.0002 | +0.0001 | OK |
| if | 0.1 | 6 | -0.0003 | +0.0001 | OK |
| if | 0.2 | 6 | +0.0003 | +0.0007 | OK |
| if | 0.3 | 6 | +0.0005 | +0.0011 | OK |
| if | 0.4 | 6 | +0.0006 | +0.0009 | OK |

## Matched seed-wise (all completed GPU cells)
- Matched cells: **90**
- max\|Δ DP_dro clean\| = **0.0072**
- max\|Δ DP_dro corrupted\| = **0.0122**
- mean Δ DP_dro clean = +0.00027
- By attack (clean DP_dro):
  - combined: n=30 max|ΔDP_clean|=0.0035 mean|Δ|=0.0009
  - dp: n=30 max|ΔDP_clean|=0.0072 mean|Δ|=0.0007
  - if: n=30 max|ΔDP_clean|=0.0044 mean|Δ|=0.0007
- Largest clean DP deltas (honest outliers, still OK if < thr):
  - dp α=0.1 s=1: gpu=0.0564 mac=0.0492 |Δ|=0.0072
  - if α=0.4 s=4: gpu=0.0117 mac=0.0162 |Δ|=0.0044
  - combined α=0.3 s=5: gpu=0.0850 mac=0.0815 |Δ|=0.0035
  - combined α=0.4 s=5: gpu=0.1283 mac=0.1251 |Δ|=0.0032
  - combined α=0.2 s=5: gpu=0.0523 mac=0.0501 |Δ|=0.0021

### Verdict
- Grid complete on both: **True**
- GAP cells (clean, n≥6 both sides, thr=0.02): **0**
- If any cell is GAP with n=6 both sides, investigate device/nondeterminism before claiming CUDA repro.

