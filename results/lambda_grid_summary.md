# Agent A3 — λ/lr grid summary (Adult, DP, α∈{0.2,0.3}, DRO)

Analysis-only. No new training. Source: `results/lambda_grid.json` 
(72/72 rows). Adult constant-predictor acc = **0.7521**.

## Coverage

- λ-grid rows present: **72/72** (100.0%)

## Per-cell table (α, λ_init, lr_λ)

Default cell (λ_init=0.0, lr_λ=0.005) marked **default**. ✓ = acc > 0.7521 (Adult constant predictor). ΔDP and Δacc are vs default at the same α (negative ΔDP = better DP).

| α | λ_init | lr_λ | n | DP | acc | acc>0.7521? | ΔDP vs default | Δacc vs default |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.00 | 0.001 | 6 | 0.2347 | 0.7574 | ✓ | +0.0013 | -0.0012 |
| 0.2 | 0.00 | 0.005 | 6 | 0.2334 **default** | 0.7586 | ✓ | +0.0000 | +0.0000 |
| 0.2 | 0.01 | 0.001 | 6 | 0.2341 | 0.7587 | ✓ | +0.0007 | +0.0000 |
| 0.2 | 0.01 | 0.005 | 6 | 0.2326 | 0.7594 | ✓ | -0.0007 | +0.0007 |
| 0.2 | 0.10 | 0.001 | 6 | 0.2219 | 0.7674 | ✓ | -0.0115 | +0.0088 |
| 0.2 | 0.10 | 0.005 | 6 | 0.2202 | 0.7679 | ✓ | -0.0132 | +0.0093 |
| 0.3 | 0.00 | 0.001 | 6 | 0.2638 | 0.6750 | ✗ | +0.0024 | -0.0006 |
| 0.3 | 0.00 | 0.005 | 6 | 0.2614 **default** | 0.6755 | ✗ | +0.0000 | +0.0000 |
| 0.3 | 0.01 | 0.001 | 6 | 0.2631 | 0.6759 | ✗ | +0.0017 | +0.0004 |
| 0.3 | 0.01 | 0.005 | 6 | 0.2604 | 0.6773 | ✗ | -0.0010 | +0.0017 |
| 0.3 | 0.10 | 0.001 | 6 | 0.2406 | 0.6844 | ✗ | -0.0207 | +0.0089 |
| 0.3 | 0.10 | 0.005 | 6 | 0.2367 | 0.6849 | ✗ | -0.0247 | +0.0093 |

## (a) Does any (λ, lr) beat the default on DP without accuracy loss?

**Yes.** Cells beating the default on DP without acc loss:

- α=0.2, λ_init=0.01, lr_λ=0.005: ΔDP=-0.0007, Δacc=+0.0007 (n=6)
- α=0.2, λ_init=0.10, lr_λ=0.005: ΔDP=-0.0132, Δacc=+0.0093 (n=6)
- α=0.2, λ_init=0.10, lr_λ=0.001: ΔDP=-0.0115, Δacc=+0.0088 (n=6)
- α=0.3, λ_init=0.10, lr_λ=0.001: ΔDP=-0.0207, Δacc=+0.0089 (n=6)
- α=0.3, λ_init=0.10, lr_λ=0.005: ΔDP=-0.0247, Δacc=+0.0093 (n=6)
- α=0.3, λ_init=0.01, lr_λ=0.005: ΔDP=-0.0010, Δacc=+0.0017 (n=6)

## (b) Does ANY cell rescue α=0.3 accuracy above 0.7521?

**No.** No α=0.3 cell currently reaches acc > 0.7521. High-α is not rescued by dual-step tuning alone — consistent with the locked paper claim.
