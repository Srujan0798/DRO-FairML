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

- L2 rows present: **120/120** (100.0%)
- Canonical LSAC/dp rows (arm a + naive-b): **60** (read-only; should be 60 = 5 alpha x 6 seeds x 2 methods)

## Per (alpha, arm): DRO accuracy & DP

Constant-predictor baseline (LSAC): **0.9016**. Canonical DP freezes at ~0.222 for alpha in {0.2,0.3,0.4}.

| alpha | (a) canonical acc | (a) canonical DP | (b) clamp=0.3 acc | (b) clamp=0.3 DP | (c) empirical acc | (c) empirical DP | (d) emp+clamp acc | (d) emp+clamp DP |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.9023 (n=6) | 0.1829 | 0.9023 (n=6) | 0.1829 | 0.9023 (n=6) | 0.1829 | 0.9023 (n=6) | 0.1829 |
| 0.1 | 0.9046 (n=6) | 0.2539 | 0.9048 (n=6) | 0.2554 | 0.9016 (n=6) | 0.1528 | 0.9016 (n=6) | 0.1533 |
| 0.2 | 0.9033 (n=6) | 0.2230 | 0.9032 (n=6) | 0.2250 | 0.9016 (n=6) | 0.0849 | 0.9016 (n=6) | 0.0851 |
| 0.3 | 0.9032 (n=6) | 0.2220 | 0.9032 (n=6) | 0.2244 | 0.9016 (n=6) | 0.0846 | 0.9016 (n=6) | 0.0850 |
| 0.4 | 0.9029 (n=6) | 0.2211 | 0.9032 (n=6) | 0.2244 | 0.9016 (n=6) | 0.0843 | 0.9016 (n=6) | 0.0850 |

## Degeneracy metrics per arm (DRO)

Thresholds: accuracy is 'off pin' if mean |acc - 0.9016| over alpha>=0.1 > 0.010; DP is 'unfrozen' if spread over alpha={0.2,0.3,0.4} > 0.020 (canonical spread ~0.002).

| arm | acc_off_pin (mean abs pp) | dp_spread {0.2,0.3,0.4} | acc off pin? | DP unfrozen? |
|---|---|---|---|---|
| a | 0.0019 | 0.0019 | no | no |
| b | 0.0020 | 0.0005 | no | no |
| c | 0.0000 | 0.0005 | no | no |
| d | 0.0000 | 0.0001 | no | no |

## Verdict per arm — does this arm un-degenerate LSAC?

An arm UN-DEGENERATES LSAC if BOTH: accuracy moves off the 0.9016 pin AND DP unfreezes across alpha in {0.2,0.3,0.4}.

| arm | un-degenerates? | evidence |
|---|---|---|
| a | no | acc_off_pin=0.0019 (>0.01? False); dp_spread_degen_band=0.0019 (>0.02? False) |
| b | no | acc_off_pin=0.0020 (>0.01? False); dp_spread_degen_band=0.0005 (>0.02? False) |
| c | no | acc_off_pin=0.0000 (>0.01? False); dp_spread_degen_band=0.0005 (>0.02? False) |
| d | no | acc_off_pin=0.0000 (>0.01? False); dp_spread_degen_band=0.0001 (>0.02? False) |

## Overall verdict

**NO — the limitation stands WITH EVIDENCE.** None of the fix arms (uniform+clamp=0.3, empirical, empirical+clamp=0.3) move accuracy off the 0.9016 constant-predictor pin AND unfreeze DP across alpha. The LSAC/DP degeneracy is NOT an artifact of the un-clamped minority radius that principled radius calibration fixes. The diagnosis (docs/LSAC_DEGENERACY.md) is confirmed, and the limitation is now supported by a tested fix rather than an untested hypothesis. LSAC/DP remains a degenerate/diagnostic result, NOT a DRO win or loss.

## Arm-by-arm reading

### (a) uniform, clamp=None [CANONICAL]

- accuracy: alpha=0.0: 0.9023, alpha=0.1: 0.9046, alpha=0.2: 0.9033, alpha=0.3: 0.9032, alpha=0.4: 0.9029
- DP: alpha=0.0: 0.1829, alpha=0.1: 0.2539, alpha=0.2: 0.2230, alpha=0.3: 0.2220, alpha=0.4: 0.2211
- verdict: still degenerate (acc_off_pin=0.0019 (>0.01? False); dp_spread_degen_band=0.0019 (>0.02? False))

### (b) uniform, clamp=0.3

- accuracy: alpha=0.0: 0.9023, alpha=0.1: 0.9048, alpha=0.2: 0.9032, alpha=0.3: 0.9032, alpha=0.4: 0.9032
- DP: alpha=0.0: 0.1829, alpha=0.1: 0.2554, alpha=0.2: 0.2250, alpha=0.3: 0.2244, alpha=0.4: 0.2244
- verdict: still degenerate (acc_off_pin=0.0020 (>0.01? False); dp_spread_degen_band=0.0005 (>0.02? False))

### (c) empirical, coord=True, clamp=None

- accuracy: alpha=0.0: 0.9023, alpha=0.1: 0.9016, alpha=0.2: 0.9016, alpha=0.3: 0.9016, alpha=0.4: 0.9016
- DP: alpha=0.0: 0.1829, alpha=0.1: 0.1528, alpha=0.2: 0.0849, alpha=0.3: 0.0846, alpha=0.4: 0.0843
- verdict: still degenerate (acc_off_pin=0.0000 (>0.01? False); dp_spread_degen_band=0.0005 (>0.02? False))

### (d) empirical, coord=True, clamp=0.3

- accuracy: alpha=0.0: 0.9023, alpha=0.1: 0.9016, alpha=0.2: 0.9016, alpha=0.3: 0.9016, alpha=0.4: 0.9016
- DP: alpha=0.0: 0.1829, alpha=0.1: 0.1533, alpha=0.2: 0.0851, alpha=0.3: 0.0850, alpha=0.4: 0.0850
- verdict: still degenerate (acc_off_pin=0.0000 (>0.01? False); dp_spread_degen_band=0.0001 (>0.02? False))

## Provenance

- Source (L2): `/Users/srujansai/Desktop/DRO-FairML/results/lsac_radii_fix.json` (120 rows)
- Source (canonical reference): `/Users/srujansai/Desktop/DRO-FairML/results/canonical_tau1.json` (60 LSAC/dp rows)
- clamp=0.3 justification: LSAC minority radius blows up to 0.53..0.87 (2.0..4.8x the majority radius 0.11..0.43) because pi_clean[minority]=0.1 shrinks the denominator. 0.3 caps the minority radius at the majority-group radius level, preventing minority over-weighting. Chosen on principle before running; not tuned-until-it-wins.
- All arms: tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0, lr_lambda=5e-3, attack_k=5, 6 seeds.