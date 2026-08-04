# LSAC pi_shrinkage_k hypothesis test — result: FAILED to fix the degeneracy

**30/30 configs complete.** LSAC only, attack='dp', DRO only, 5 alphas × 3 seeds × 2
shrinkage strengths (k=700, k=2100), chosen on principle before running (see
`experiments/run_lsac_pi_shrinkage.py` docstring for the derivation from
n_val=2804, minority=6.4%).

## Result

| Arm | n | acc mean | acc range | DP mean |
|---|---|---|---|---|
| k=700 (moderate shrinkage) | 15 | 0.9029 | 0.9016–0.9059 | 0.2142 |
| k=2100 (strong shrinkage) | 15 | 0.9030 | 0.9016–0.9064 | 0.2143 |
| **Canonical (k=0, unshrunk)** — same seeds 0-2, read from `canonical_tau1.json` | 15 | ~0.902–0.905 | — | 0.184–0.245 |

**Shrinkage produced results statistically indistinguishable from the unshrunk
baseline**, at both a moderate and a strong shrinkage strength. Accuracy stays
pinned near the constant-predictor baseline (0.9016) exactly as in the canonical
degenerate case. DP does not move outside the canonical range. Doubling the
shrinkage strength (700→2100) changed the outcome by less than 0.0001 on accuracy
and DP — the mechanism has essentially no effect at either tested strength.

## Interpretation

This is the **second independent, principled mechanism to fail at fixing LSAC/DP**,
after L2's `radii_clamp` (see `results/lsac_radii_summary.md`). Both attempts were
honestly pre-registered (values chosen before seeing results) and both failed
cleanly rather than ambiguously.

This matters for the paper's framing: LSAC's validation-set minority proportion is
**not noisy** (n_val=2804, a large and reliable sample) — so shrinkage's usual
justification (denoising a small-sample estimate) never applied here in the first
place, as flagged before this experiment ran. The fact that neither a hard clamp nor
a soft shrinkage recovers LSAC is consistent with the degeneracy being **structural**:
the ~90/10 imbalance itself, not an estimation artifact, drives the TV-radius formula
into a regime that induces majority-class collapse.

**Recommendation:** report LSAC/DP as a genuine, structural limitation of TV-ball
radii calibration under severe protected-group imbalance — not as an unfixed bug.
Two independent fix attempts failed under honest, pre-registered testing, which is
stronger evidence for "structural" than either attempt alone would have been.
