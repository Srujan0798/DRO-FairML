# TASK C — μ sensitivity (pre-registered)

rows: **90/90** · pre-reg in `docs/superpowers/specs/2026-08-05-mu-sensitivity-prereg.md`

## RULE C1 — μ_max_safe(α): where is AL SAFE?

| α | μ=0.5 | μ=1 | μ=2 | μ=20 | μ_max_safe |
|---|---|---|---|---|---|
| 0.0 | ✓ (0.814) | ✓ (0.814) | ✓ (0.811) | ✓ (0.797) | 20.0 |
| 0.2 | ✓ (0.769) | ✓ (0.770) | ✓ (0.773) | ✓ (0.778) | 20.0 |
| 0.4 | ✗ (0.540) | ✗ (0.498) | ✗ (0.446) | ✗ (0.274) | None |

**Pre-registered expectation:** μ_max_safe DECREASES as α increases. **Observed:** α=0.0→20.0, α=0.2→20.0, α=0.4→None. ✓ expectation CONFIRMED

## RULE C2 — recommended μ (largest both SAFE and EFFECTIVE)

| α | μ_recommend | DP reduction | p-value | accuracy | action |
|---|---|---|---|---|---|
| 0.0 | μ=20.0 | +81.7% | 0.0156 | 0.7966 | **use μ=20.0** |
| 0.2 | μ=20.0 | +70.8% | 0.0156 | 0.7783 | **use μ=20.0** |
| 0.4 | none | — | — | — | **do not use AL** |

## RULE C3 — Credit rescue: does a smaller μ work?

**Credit NOT rescued:** no μ ∈ {0.5, 1, 2} is both SAFE and EFFECTIVE. AL is **Adult-only among tested datasets**.

## RULE C4 — monotonicity sanity check

α=0.0: DP curve monotone ✓ ['0.1307', '0.1199', '0.1016', '0.0260']
α=0.2: DP curve monotone ✓ ['0.2164', '0.2025', '0.1792', '0.0682']
α=0.4: DP curve monotone ✓ ['0.2057', '0.1562', '0.1059', '0.0172']
