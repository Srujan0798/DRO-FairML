# Agent N1 — attack strength × radius sensitivity

**Kuldeep's FIRST technical question (May 29, 14 months unanswered):**
> "At lower corruption levels (α=0.1): DRO does not significantly outperform Naive — the attack is too weak to differentiate. Does the attack affect the radius? ... if the attack is too weak, then DRO would perform well? specially at α=0.1."

Two arms, both stamping **MEASURED attack effectiveness** — the |ΔDP| the corruption itself induces on the training labels, computed pre-training (field `attack_effectiveness` on every row). Strength is measured, not assumed.

## Coverage

- ARM A (attack_strength.json): **20/144** rows (13.9%)
  - **ARM A INCOMPLETE** — partial-data mode; re-run as more rows land.
- ARM B (radius_sensitivity.json): **20/180** rows (11.1%)
  - **ARM B INCOMPLETE** — partial-data mode; re-run as more rows land.
- Canonical DP rows (pgd_steps=20, radii_scale=1.0, read-only): 192

## ARM A — DRO advantage vs MEASURED attack strength

pgd_steps ∈ {5, 20(canonical), 50}; attack='dp'; α ∈ {0.1, 0.2}. DRO advantage = Naive_DP − DRO_DP (positive ⇒ DRO fairer). attack_effectiveness = |ΔDP_train| the corruption induces on training labels.

| dataset | α | pgd_steps | n | DP_naive | DP_dro | DRO_adv | wins_dro | p | attack_eff | n_eff |
|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.1 | 5 | 6 | 0.2150 | 0.2125 | +0.0025 | 5/6 | 0.0469 * | 0.0534 | 12 |
| adult | 0.1 | 20 | 6 | 0.2026 | 0.1999 | +0.0027 | 5/6 | 0.0312 * | 0.0534 | 4 |
| adult | 0.2 | 5 | 2 | 0.2673 | 0.2490 | +0.0183 | 2/2 | 0.2500  | 0.1761 | 8 |
| adult | 0.2 | 20 | 6 | 0.2452 | 0.2334 | +0.0119 | 6/6 | 0.0156 * | — | 0 |
| credit | 0.1 | 20 | 6 | 0.0151 | 0.0134 | +0.0017 | 6/6 | 0.0156 * | — | 0 |
| credit | 0.2 | 20 | 6 | 0.0198 | 0.0178 | +0.0020 | 6/6 | 0.0156 * | — | 0 |
| lsac | 0.1 | 20 | 6 | 0.2201 | 0.2539 | -0.0338 | 0/6 | 1.0000  | — | 0 |
| lsac | 0.2 | 20 | 6 | 0.1827 | 0.2230 | -0.0403 | 0/6 | 1.0000  | — | 0 |

**Spearman ρ (attack_eff vs DRO_advantage) = +0.866 (p=0.3333)** across 3 (ds,α,pgd_steps) cells.

→ Directional: DRO advantage trends up with attack strength but not significantly at this n.

## ARM B — DRO DP vs radii_scale (fixed attack)

radii_scale ∈ {0.5, 1.0(canonical), 2.0}; attack='dp'; DRO only. Naive is radii_scale-invariant (does not use ρ), so 'advantage' is DRO's DP reduction vs canonical Naive. The question: does DP peak (reach minimum) when the radius matches the true corruption?

| dataset | α | radii_scale | n | DP_naive(ref) | DP_dro | DRO_adv | attack_eff | n_eff |
|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | 0.5 | 6 | 0.1491 | 0.1426 | +0.0065 | 0.0000 | 6 |
| adult | 0.0 | 2.0 | 6 | 0.1491 | 0.1426 | +0.0065 | 0.0000 | 6 |
| adult | 0.1 | 0.5 | 4 | 0.2036 | 0.1993 | +0.0043 | 0.0534 | 4 |
| adult | 0.1 | 2.0 | 4 | 0.2036 | 0.1993 | +0.0043 | 0.0534 | 4 |

### ARM B — optimal radius per (dataset, α)

| dataset | α | n_radii | best_radii_scale | DP_dro@best | DP_dro@1.0 | attack_eff |
|---|---|---|---|---|---|---|
| adult | 0.0 | 2 | 0.5 | 0.1426 | — | 0.0000 |
| adult | 0.1 | 2 | 2.0 | 0.1993 | — | 0.0534 |

Best-radius counts across 2 (ds,α) cells: {0.5: 1, 2.0: 1}.

## Verdict — Kuldeep's question answered

ARM A: directional but not significant (ρ=+0.866, p=0.3333). DRO advantage trends up with attack strength but is not significant at this n. ARM B: optimal-radius table present but correlation not computed (insufficient cells).

Source files: `results/attack_strength.json`, `results/radius_sensitivity.json`, canonical (read-only).
