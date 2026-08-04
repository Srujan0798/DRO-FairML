# Agent A3 — λ/lr grid summary (Adult, DP, α∈{0.2,0.3}, DRO)

Analysis-only. No new training. Source: `results/lambda_grid.json` 
(12/72 rows). Adult constant-predictor acc = **0.7521**.

## Coverage

- λ-grid rows present: **12/72** (16.7%)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## Per-cell table (α, λ_init, lr_λ)

Default cell (λ_init=0.0, lr_λ=0.005) marked **default**. ✓ = acc > 0.7521 (Adult constant predictor). ΔDP and Δacc are vs default at the same α (negative ΔDP = better DP).

| α | λ_init | lr_λ | n | DP | acc | acc>0.7521? | ΔDP vs default | Δacc vs default |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.00 | 0.001 | 2 | 0.2406 | 0.7549 | ✓ | +0.0004 | -0.0003 |
| 0.2 | 0.00 | 0.005 | 2 | 0.2402 **default** | 0.7552 | ✓ | +0.0000 | +0.0000 |
| 0.2 | 0.01 | 0.001 | 2 | 0.2405 | 0.7550 | ✓ | +0.0003 | -0.0002 |
| 0.2 | 0.01 | 0.005 | 2 | 0.2401 | 0.7546 | ✓ | -0.0001 | -0.0006 |
| 0.2 | 0.10 | 0.001 | 2 | 0.2346 | 0.7580 | ✓ | -0.0056 | +0.0029 |
| 0.2 | 0.10 | 0.005 | 2 | 0.2340 | 0.7583 | ✓ | -0.0062 | +0.0031 |
| 0.3 | 0.00 | 0.001 | 0 | — | — | — | — | — |
| 0.3 | 0.00 | 0.005 | 0 | — | — | — | — | — |
| 0.3 | 0.01 | 0.001 | 0 | — | — | — | — | — |
| 0.3 | 0.01 | 0.005 | 0 | — | — | — | — | — |
| 0.3 | 0.10 | 0.001 | 0 | — | — | — | — | — |
| 0.3 | 0.10 | 0.005 | 0 | — | — | — | — | — |

## (a) Does any (λ, lr) beat the default on DP without accuracy loss?

**Yes.** Cells beating the default on DP without acc loss:

- α=0.2, λ_init=0.10, lr_λ=0.005: ΔDP=-0.0062, Δacc=+0.0031 (n=2)
- α=0.2, λ_init=0.10, lr_λ=0.001: ΔDP=-0.0056, Δacc=+0.0029 (n=2)

## (b) Does ANY cell rescue α=0.3 accuracy above 0.7521?

**No.** No α=0.3 cell currently reaches acc > 0.7521. (File INCOMPLETE — re-run as more rows land.)
