# UTKFace CUDA vs MPS — intermediate checkpoint (dp α≤0.1 complete)

Matched cells: **12** (U1 grid still running toward 90).
Protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, REAL features.

| attack | α | seed | ΔDP_dro | Δacc_dro | ΔDP_naive | note |
|--------|---:|-----:|--------:|---------:|----------:|------|
| dp | 0.0 | 0 | -0.0006 | +0.0004 | -0.0002 | OK |
| dp | 0.0 | 1 | +0.0001 | -0.0006 | +0.0001 | OK |
| dp | 0.0 | 2 | +0.0001 | +0.0013 | +0.0001 | OK |
| dp | 0.0 | 3 | +0.0003 | -0.0004 | +0.0005 | OK |
| dp | 0.0 | 4 | +0.0010 | +0.0000 | -0.0001 | OK |
| dp | 0.0 | 5 | +0.0001 | +0.0000 | +0.0002 | OK |
| dp | 0.1 | 0 | -0.0006 | +0.0004 | +0.0001 | OK |
| dp | 0.1 | 1 | +0.0072 | -0.0011 | +0.0005 | OK |
| dp | 0.1 | 2 | +0.0006 | -0.0002 | +0.0002 | OK |
| dp | 0.1 | 3 | +0.0005 | +0.0025 | +0.0003 | OK |
| dp | 0.1 | 4 | +0.0003 | -0.0025 | -0.0000 | OK |
| dp | 0.1 | 5 | +0.0006 | +0.0002 | +0.0006 | OK |

- max\|ΔDP_dro\| = **0.0072**
- max\|Δacc_dro\| = **0.0025**
- mean ΔDP_dro = +0.00080, mean Δacc_dro = +0.00000
- GAP cells (threshold 0.02): **0**

### Verdict (partial)
CUDA and MPS agree within ~0.01 on DP/acc for completed dp cells at α∈{0.0,0.1}.
Do **not** claim full flair2 reproducibility until attacks if/combined and α∈{0.2,0.3,0.4} finish.
