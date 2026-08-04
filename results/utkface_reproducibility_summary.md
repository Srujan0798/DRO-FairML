# UTKFace reproducibility: Mac MPS vs flair2 CUDA

- Mac rows: **90/90** (`results/utkface_canonical.json`)
- flair2 rows: **12/90** (`results/utkface_flair2.json`)

Protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, n_seeds=6, REAL features.
Same seeds 0–5. Large gaps are bugs to investigate.

| attack | α | n_mac | n_gpu | Δ mean DP_dro (gpu−mac) | Δ mean acc_dro | note |
|--------|---:|------:|------:|------------------------:|---------------:|------|
| combined | 0.0 | 6 | 0 | +nan | +nan | partial |
| combined | 0.1 | 6 | 0 | +nan | +nan | partial |
| combined | 0.2 | 6 | 0 | +nan | +nan | partial |
| combined | 0.3 | 6 | 0 | +nan | +nan | partial |
| combined | 0.4 | 6 | 0 | +nan | +nan | partial |
| dp | 0.0 | 6 | 6 | +0.0002 | +0.0001 | OK |
| dp | 0.1 | 6 | 6 | +0.0014 | -0.0001 | OK |
| dp | 0.2 | 6 | 0 | +nan | +nan | partial |
| dp | 0.3 | 6 | 0 | +nan | +nan | partial |
| dp | 0.4 | 6 | 0 | +nan | +nan | partial |
| if | 0.0 | 6 | 0 | +nan | +nan | partial |
| if | 0.1 | 6 | 0 | +nan | +nan | partial |
| if | 0.2 | 6 | 0 | +nan | +nan | partial |
| if | 0.3 | 6 | 0 | +nan | +nan | partial |
| if | 0.4 | 6 | 0 | +nan | +nan | partial |

### Verdict
- Grid complete on both: **False**
- If any cell is GAP with n=6 both sides, investigate device/nondeterminism before claiming CUDA repro.
