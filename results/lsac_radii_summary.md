# Agent L2 — LSAC degeneracy fix: hypothesis test summary

HYPOTHESIS: LSAC/DP is degenerate because the DRO radii formula `rho_dp[j] = alpha / ((1-alpha)*pi_clean[j] + alpha)` blows up on the ~90/10 imbalanced minority group (pi_clean[minority]=0.1 -> rho_min = 0.53..0.87, i.e. 2.0..4.8x the majority radius 0.11..0.43). The minority group is over-weighted, the classifier collapses to the majority class, accuracy pins at the 0.9016 constant-predictor baseline, and DP freezes at ~0.222 for alpha in {0.2,0.3,0.4}.

FIX UNDER TEST: `radii_clamp=0.3` (chosen on PRINCIPLE before running). 0.3 caps the minority radius at the majority-group radius level (majority radius is 0.11..0.43; 0.3 sits near the majority radius at alpha=0.3 = 0.32). It is the smallest cap that brings the minority radius into the same order of magnitude as the majority radius. NOT tuned — derived from the formula on the diagnosed imbalance, before any L2 result.

ARMS (LSAC, attack='dp', 5 alpha x 6 seeds):
- (a) uniform, clamp=None — CANONICAL reference (read-only from canonical_tau1.json)
- (b) uniform, clamp=0.3 (coordinated=False, same corruption as canonical)
- (c) empirical, coordinated=True, clamp=None
- (d) empirical, coordinated=True, clamp=0.3

Naive baseline: arms (c,d) use L2 naive (coordinated=True, same corrupted data as DRO); arm (b) uses canonical naive (coordinated=False). Naive does not use radii.

## Coverage

- L2 rows present: **6/120** (5.0%)
- Canonical LSAC/dp rows (arm a + naive-b): **60** (read-only; should be 60 = 5 alpha x 6 seeds x 2 methods)
- **INCOMPLETE** — partial-data mode; re-run as more rows land (idempotent).

## Per (alpha, arm): DRO accuracy & DP

Constant-predictor baseline (LSAC): **0.9016**. Canonical DP freezes at ~0.222 for alpha in {0.2,0.3,0.4}.

| alpha | (a) canonical acc | (a) canonical DP | (b) clamp=0.3 acc | (b) clamp=0.3 DP | (c) empirical acc | (c) empirical DP | (d) emp+clamp acc | (d) emp+clamp DP |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.9023 (n=6) | 0.1829 | 0.9023 (n=6) | 0.1829 | — | — | — | — |
| 0.1 | 0.9046 (n=6) | 0.2539 | — | — | — | — | — | — |
| 0.2 | 0.9033 (n=6) | 0.2230 | — | — | — | — | — | — |
| 0.3 | 0.9032 (n=6) | 0.2220 | — | — | — | — | — | — |
| 0.4 | 0.9029 (n=6) | 0.2211 | — | — | — | — | — | — |

## Degeneracy metrics per arm (DRO)

Thresholds: accuracy is 'off pin' if mean |acc - 0.9016| over alpha>=0.1 > 0.010; DP is 'unfrozen' if spread over alpha={0.2,0.3,0.4} > 0.020 (canonical spread ~0.002).

| arm | acc_off_pin (mean abs pp) | dp_spread {0.2,0.3,0.4} | acc off pin? | DP unfrozen? |
|---|---|---|---|---|
| a | 0.0019 | 0.0019 | no | no |
| b | — | — | no | no |
| c | — | — | no | no |
| d | — | — | no | no |

## Verdict per arm — does this arm un-degenerate LSAC?

An arm UN-DEGENERATES LSAC if BOTH: accuracy moves off the 0.9016 pin AND DP unfreezes across alpha in {0.2,0.3,0.4}.

| arm | un-degenerates? | evidence |
|---|---|---|
| a | no | acc_off_pin=0.0019 (>0.01? False); dp_spread_degen_band=0.0019 (>0.02? False) |
| b | — | insufficient data (need >=2 of alpha={0.2,0.3,0.4}; have 0) |
| c | — | insufficient data (need >=2 of alpha={0.2,0.3,0.4}; have 0) |
| d | — | insufficient data (need >=2 of alpha={0.2,0.3,0.4}; have 0) |

## Overall verdict

**INCOMPLETE.** Not all fix arms have enough data to evaluate yet. Re-run this summary as more L2 rows land (idempotent). No verdict can be honestly stated against an empty band.

## Arm-by-arm reading

### (a) uniform, clamp=None [CANONICAL]

- accuracy: alpha=0.0: 0.9023, alpha=0.1: 0.9046, alpha=0.2: 0.9033, alpha=0.3: 0.9032, alpha=0.4: 0.9029
- DP: alpha=0.0: 0.1829, alpha=0.1: 0.2539, alpha=0.2: 0.2230, alpha=0.3: 0.2220, alpha=0.4: 0.2211
- verdict: still degenerate (acc_off_pin=0.0019 (>0.01? False); dp_spread_degen_band=0.0019 (>0.02? False))

### (b) uniform, clamp=0.3

- accuracy: alpha=0.0: 0.9023
- DP: alpha=0.0: 0.1829
- verdict: INCOMPLETE (insufficient data (need >=2 of alpha={0.2,0.3,0.4}; have 0))

### (c) empirical, coord=True, clamp=None

- No data yet (INCOMPLETE).

### (d) empirical, coord=True, clamp=0.3

- No data yet (INCOMPLETE).

## Provenance

- Source (L2): `/Users/srujansai/Desktop/DRO-FairML/results/lsac_radii_fix.json` (6 rows)
- Source (canonical reference): `/Users/srujansai/Desktop/DRO-FairML/results/canonical_tau1.json` (60 LSAC/dp rows)
- clamp=0.3 justification: LSAC minority radius blows up to 0.53..0.87 (2.0..4.8x the majority radius 0.11..0.43) because pi_clean[minority]=0.1 shrinks the denominator. 0.3 caps the minority radius at the majority-group radius level, preventing minority over-weighting. Chosen on principle before running; not tuned-until-it-wins.
- All arms: tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0, lr_lambda=5e-3, attack_k=5, 6 seeds.