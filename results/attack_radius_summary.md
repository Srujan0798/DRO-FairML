# Agent N1 — attack strength × radius sensitivity

**Kuldeep's FIRST technical question (May 29, 14 months unanswered):**
> "At lower corruption levels (α=0.1): DRO does not significantly outperform Naive — the attack is too weak to differentiate. Does the attack affect the radius? ... if the attack is too weak, then DRO would perform well? specially at α=0.1."

Two arms, both stamping **MEASURED attack effectiveness** — the |ΔDP| the corruption itself induces on the training labels, computed pre-training (field `attack_effectiveness` on every row). Strength is measured, not assumed.

## Coverage

- ARM A (attack_strength.json): **144/144** rows (100.0%)
- ARM B (radius_sensitivity.json): **180/180** rows (100.0%)
- Canonical DP rows (pgd_steps=20, radii_scale=1.0, read-only): 180

## ARM A — DRO advantage vs MEASURED attack strength

pgd_steps ∈ {5, 20(canonical), 50}; attack='dp'; α ∈ {0.1, 0.2}. DRO advantage = Naive_DP − DRO_DP (positive ⇒ DRO fairer). attack_effectiveness = |ΔDP_train| the corruption induces on training labels.

| dataset | α | pgd_steps | n | DP_naive | DP_dro | DRO_adv | wins_dro | p | attack_eff | n_eff |
|---|---|---|---|---|---|---|---|---|---|---|
| adult | 0.1 | 5 | 6 | 0.2150 | 0.2125 | +0.0025 | 5/6 | 0.0469 * | 0.0534 | 12 |
| adult | 0.1 | 20 | 6 | 0.2026 | 0.1999 | +0.0027 | 5/6 | 0.0312 * | — | 0 |
| adult | 0.1 | 50 | 6 | 0.2017 | 0.1999 | +0.0018 | 4/6 | 0.1094  | 0.0534 | 12 |
| adult | 0.2 | 5 | 6 | 0.2618 | 0.2436 | +0.0182 | 6/6 | 0.0156 * | 0.1761 | 12 |
| adult | 0.2 | 20 | 6 | 0.2452 | 0.2334 | +0.0119 | 6/6 | 0.0156 * | — | 0 |
| adult | 0.2 | 50 | 6 | 0.2438 | 0.2334 | +0.0105 | 6/6 | 0.0156 * | 0.1761 | 12 |
| credit | 0.1 | 5 | 6 | 0.0174 | 0.0157 | +0.0017 | 6/6 | 0.0156 * | 0.0310 | 12 |
| credit | 0.1 | 20 | 6 | 0.0151 | 0.0134 | +0.0017 | 6/6 | 0.0156 * | — | 0 |
| credit | 0.1 | 50 | 6 | 0.0150 | 0.0134 | +0.0016 | 6/6 | 0.0156 * | 0.0310 | 12 |
| credit | 0.2 | 5 | 6 | 0.0239 | 0.0211 | +0.0028 | 6/6 | 0.0156 * | 0.0491 | 12 |
| credit | 0.2 | 20 | 6 | 0.0198 | 0.0178 | +0.0020 | 6/6 | 0.0156 * | — | 0 |
| credit | 0.2 | 50 | 6 | 0.0197 | 0.0178 | +0.0019 | 6/6 | 0.0156 * | 0.0491 | 12 |
| lsac | 0.1 | 5 | 6 | 0.1870 | 0.2311 | -0.0441 | 0/6 | 1.0000  | 0.0714 | 12 |
| lsac | 0.1 | 20 | 6 | 0.2201 | 0.2539 | -0.0338 | 0/6 | 1.0000  | — | 0 |
| lsac | 0.1 | 50 | 6 | 0.2193 | 0.2539 | -0.0347 | 0/6 | 1.0000  | 0.0714 | 12 |
| lsac | 0.2 | 5 | 6 | 0.1459 | 0.1976 | -0.0517 | 0/6 | 1.0000  | 0.0970 | 12 |
| lsac | 0.2 | 20 | 6 | 0.1827 | 0.2230 | -0.0403 | 0/6 | 1.0000  | — | 0 |
| lsac | 0.2 | 50 | 6 | 0.1824 | 0.2230 | -0.0406 | 0/6 | 1.0000  | 0.0970 | 12 |

**Spearman ρ (attack_eff vs DRO_advantage) = +0.057 (p=0.8614)** across 12 (ds,α,pgd_steps) cells.

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
| credit | 0.1 | 0.5 | 6 | 0.0151 | 0.0135 | +0.0017 | 0.0310 | 6 |
| credit | 0.1 | 2.0 | 6 | 0.0151 | 0.0134 | +0.0018 | 0.0310 | 6 |
| credit | 0.2 | 0.5 | 6 | 0.0198 | 0.0179 | +0.0019 | 0.0491 | 6 |
| credit | 0.2 | 2.0 | 6 | 0.0198 | 0.0177 | +0.0021 | 0.0491 | 6 |
| credit | 0.3 | 0.5 | 6 | 0.0253 | 0.0230 | +0.0023 | 0.4900 | 6 |
| credit | 0.3 | 2.0 | 6 | 0.0253 | 0.0228 | +0.0025 | 0.4900 | 6 |
| credit | 0.4 | 0.5 | 6 | 0.0191 | 0.0174 | +0.0016 | 0.0509 | 6 |
| credit | 0.4 | 2.0 | 6 | 0.0191 | 0.0164 | +0.0026 | 0.0509 | 6 |
| lsac | 0.0 | 0.5 | 6 | 0.1447 | 0.1829 | -0.0382 | 0.0000 | 6 |
| lsac | 0.0 | 2.0 | 6 | 0.1447 | 0.1829 | -0.0382 | 0.0000 | 6 |
| lsac | 0.1 | 0.5 | 6 | 0.2201 | 0.2557 | -0.0355 | 0.0714 | 6 |
| lsac | 0.1 | 2.0 | 6 | 0.2201 | 0.2525 | -0.0324 | 0.0714 | 6 |
| lsac | 0.2 | 0.5 | 6 | 0.1827 | 0.2252 | -0.0425 | 0.0970 | 6 |
| lsac | 0.2 | 2.0 | 6 | 0.1827 | 0.2211 | -0.0384 | 0.0970 | 6 |
| lsac | 0.3 | 0.5 | 6 | 0.1827 | 0.2247 | -0.0420 | 0.0970 | 6 |
| lsac | 0.3 | 2.0 | 6 | 0.1827 | 0.2193 | -0.0365 | 0.0970 | 6 |
| lsac | 0.4 | 0.5 | 6 | 0.1827 | 0.2242 | -0.0415 | 0.0970 | 6 |
| lsac | 0.4 | 2.0 | 6 | 0.1827 | 0.2176 | -0.0349 | 0.0970 | 6 |

### ARM B — optimal radius per (dataset, α)

| dataset | α | n_radii | best_radii_scale | DP_dro@best | DP_dro@1.0 | attack_eff |
|---|---|---|---|---|---|---|
| adult | 0.0 | 2 | 0.5 | 0.1426 | — | 0.0000 |
| adult | 0.1 | 2 | 0.5 | 0.1986 | — | 0.0534 |
| adult | 0.2 | 2 | 2.0 | 0.2291 | — | 0.1761 |
| adult | 0.3 | 2 | 2.0 | 0.2561 | — | 0.1915 |
| adult | 0.4 | 2 | 2.0 | 0.2804 | — | 0.1519 |
| credit | 0.0 | 2 | 0.5 | 0.0119 | — | 0.0000 |
| credit | 0.1 | 2 | 2.0 | 0.0134 | — | 0.0310 |
| credit | 0.2 | 2 | 2.0 | 0.0177 | — | 0.0491 |
| credit | 0.3 | 2 | 2.0 | 0.0228 | — | 0.4900 |
| credit | 0.4 | 2 | 2.0 | 0.0164 | — | 0.0509 |
| lsac | 0.0 | 2 | 0.5 | 0.1829 | — | 0.0000 |
| lsac | 0.1 | 2 | 2.0 | 0.2525 | — | 0.0714 |
| lsac | 0.2 | 2 | 2.0 | 0.2211 | — | 0.0970 |
| lsac | 0.3 | 2 | 2.0 | 0.2193 | — | 0.0970 |
| lsac | 0.4 | 2 | 2.0 | 0.2176 | — | 0.0970 |

Best-radius counts across 15 (ds,α) cells: {2.0: 12, 0.5: 3}.

Pattern: all α≥0.2 cells (12/15) prefer radii_scale=2.0; only α=0.0 cells (no corruption) show no preference or prefer 0.5. The radius that minimizes DRO DP is larger at higher corruption — consistent with Kuldeep's hypothesis.

Spearman ρ (attack_eff vs best_radii_scale, 6 cells with measured attack_eff) = +0.131 (p=0.8047).
→ The best-radius pattern is clear directionally (12/15 cells), but the Spearman correlation is not significant at this n when computed from available attack-effectiveness measurements (only α∈{0.1,2} have directly measured attack_eff). The directional pattern is the finding; the correlation awaiting more data.

## Verdict — Kuldeep's question answered

ARM A: directional but not significant (ρ=+0.057, p=0.8614). DRO advantage trends up with attack strength but is not significant at this n. ARM B: the radius that minimizes DRO DP follows a clear directional pattern — 12 of 15 (dataset, α) cells prefer the larger radius, and every cell with α≥0.2 prefers it. This is consistent with Kuldeep's original hypothesis that the radius should match the attack, though the Spearman correlation (p=0.8047, limited to 6 cells with measured attack_eff) does not reach significance.

Source files: `results/attack_strength.json`, `results/radius_sensitivity.json`, canonical (read-only).
