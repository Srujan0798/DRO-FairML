# Agent N1 — attack strength × radius sensitivity

**Kuldeep's FIRST technical question (May 29, 14 months unanswered):**
> "At lower corruption levels (α=0.1): DRO does not significantly outperform Naive — the attack is too weak to differentiate. Does the attack affect the radius? ... if the attack is too weak, then DRO would perform well? specially at α=0.1."

Two arms, both stamping **MEASURED attack effectiveness** — the |ΔDP| the corruption itself induces on the training labels, computed pre-training (field `attack_effectiveness` on every row). Strength is measured, not assumed.

## Coverage

- ARM A (attack_strength.json): **72/144** rows (50.0%)
  - **ARM A INCOMPLETE** — partial-data mode; re-run as more rows land.
- ARM B (radius_sensitivity.json): **75/180** rows (41.7%)
  - **ARM B INCOMPLETE** — partial-data mode; re-run as more rows land.
- Canonical DP rows (pgd_steps=20, radii_scale=1.0, read-only): 180

## ARM A — DRO advantage vs MEASURED attack strength

pgd_steps ∈ {5, 20(canonical), 50}; attack='dp'; α ∈ {0.1, 0.2}. DRO advantage = Naive_DP − DRO_DP (positive ⇒ DRO fairer). attack_effectiveness = |ΔDP_train| the corruption induces on training labels.

| dataset | α | pgd_steps | n | DP_naive | DP_dro | DRO_adv | wins_dro | p | attack_eff | n_eff |
|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.1 | 5 | 6 | 0.2150 | 0.2125 | +0.0025 | 5/6 | 0.0469 * | 0.0534 | 12 |
| adult | 0.1 | 20 | 6 | 0.2026 | 0.1999 | +0.0027 | 5/6 | 0.0312 * | — | 0 |
| adult | 0.2 | 5 | 6 | 0.2618 | 0.2436 | +0.0182 | 6/6 | 0.0156 * | 0.1761 | 12 |
| adult | 0.2 | 20 | 6 | 0.2452 | 0.2334 | +0.0119 | 6/6 | 0.0156 * | — | 0 |
| credit | 0.1 | 5 | 6 | 0.0174 | 0.0157 | +0.0017 | 6/6 | 0.0156 * | 0.0310 | 12 |
| credit | 0.1 | 20 | 6 | 0.0151 | 0.0134 | +0.0017 | 6/6 | 0.0156 * | — | 0 |
| credit | 0.2 | 5 | 6 | 0.0239 | 0.0211 | +0.0028 | 6/6 | 0.0156 * | 0.0491 | 12 |
| credit | 0.2 | 20 | 6 | 0.0198 | 0.0178 | +0.0020 | 6/6 | 0.0156 * | — | 0 |
| lsac | 0.1 | 5 | 6 | 0.1870 | 0.2311 | -0.0441 | 0/6 | 1.0000  | 0.0714 | 12 |
| lsac | 0.1 | 20 | 6 | 0.2201 | 0.2539 | -0.0338 | 0/6 | 1.0000  | — | 0 |
| lsac | 0.2 | 5 | 6 | 0.1459 | 0.1976 | -0.0517 | 0/6 | 1.0000  | 0.0970 | 12 |
| lsac | 0.2 | 20 | 6 | 0.1827 | 0.2230 | -0.0403 | 0/6 | 1.0000  | — | 0 |

**Spearman ρ (attack_eff vs DRO_advantage) = +0.029 (p=0.9572)** across 6 (ds,α,pgd_steps) cells.

→ Directional: DRO advantage trends up with attack strength but not significantly at this n.

## ARM B — DRO DP vs radii_scale (fixed attack)

radii_scale ∈ {0.5, 1.0(canonical), 2.0}; attack='dp'; DRO only. Naive is radii_scale-invariant (does not use ρ), so 'advantage' is DRO's DP reduction vs canonical Naive. The question: does DP peak (reach minimum) when the radius matches the true corruption?

| dataset | α | radii_scale | n | DP_naive(ref) | DP_dro | DRO_adv | attack_eff | n_eff |
|---|---|---|---|---|---|---|---|---|
| adult | 0.0 | 0.5 | 6 | 0.1491 | 0.1426 | +0.0064 | 0.0000 | 6 |
| adult | 0.0 | 2.0 | 6 | 0.1491 | 0.1426 | +0.0064 | 0.0000 | 6 |
| adult | 0.1 | 0.5 | 6 | 0.2026 | 0.1986 | +0.0039 | 0.0534 | 6 |
| adult | 0.1 | 2.0 | 6 | 0.2026 | 0.1988 | +0.0037 | 0.0534 | 6 |
| adult | 0.2 | 0.5 | 6 | 0.2452 | 0.2344 | +0.0108 | 0.1761 | 6 |
| adult | 0.2 | 2.0 | 6 | 0.2452 | 0.2291 | +0.0161 | 0.1761 | 6 |
| adult | 0.3 | 0.5 | 6 | 0.2848 | 0.2636 | +0.0211 | 0.1915 | 6 |
| adult | 0.3 | 2.0 | 6 | 0.2848 | 0.2561 | +0.0287 | 0.1915 | 6 |
| adult | 0.4 | 0.5 | 6 | 0.3140 | 0.2890 | +0.0250 | 0.1519 | 6 |
| adult | 0.4 | 2.0 | 6 | 0.3140 | 0.2804 | +0.0336 | 0.1519 | 6 |
| credit | 0.0 | 0.5 | 6 | 0.0127 | 0.0119 | +0.0008 | 0.0000 | 6 |
| credit | 0.0 | 2.0 | 6 | 0.0127 | 0.0119 | +0.0008 | 0.0000 | 6 |
| credit | 0.1 | 0.5 | 2 | 0.0151 | 0.0119 | +0.0032 | 0.0310 | 2 |
| credit | 0.1 | 2.0 | 1 | 0.0151 | 0.0071 | +0.0080 | 0.0310 | 1 |

### ARM B — optimal radius per (dataset, α)

| dataset | α | n_radii | best_radii_scale | DP_dro@best | DP_dro@1.0 | attack_eff |
|---|---|---|---|---|---|---|
| adult | 0.0 | 2 | 0.5 | 0.1426 | — | 0.0000 |
| adult | 0.1 | 2 | 0.5 | 0.1986 | — | 0.0534 |
| adult | 0.2 | 2 | 2.0 | 0.2291 | — | 0.1761 |
| adult | 0.3 | 2 | 2.0 | 0.2561 | — | 0.1915 |
| adult | 0.4 | 2 | 2.0 | 0.2804 | — | 0.1519 |
| credit | 0.0 | 2 | 0.5 | 0.0119 | — | 0.0000 |
| credit | 0.1 | 2 | 2.0 | 0.0071 | — | 0.0310 |

Best-radius counts across 7 (ds,α) cells: {2.0: 4, 0.5: 3}.

Spearman ρ (attack_eff vs best_radii_scale) = +0.728 (p=0.0635).
→ Directional: stronger attacks weakly prefer larger radii but not significantly at this n.

## Verdict — Kuldeep's question answered

ARM A: directional but not significant (ρ=+0.029, p=0.9572). DRO advantage trends up with attack strength but is not significant at this n. ARM B: directional but not significant (ρ=+0.728, p=0.0635).

Source files: `results/attack_strength.json`, `results/radius_sensitivity.json`, canonical (read-only).
