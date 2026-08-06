# TASK C2 — does AL (μ=20) compound with the radius fix (radii_scale=2.0)?

rows: **18 new** in `results/al_radius_compound.json` + existing canonical / mu_sensitivity / radius_sensitivity arms reused read-only · pre-reg in `docs/superpowers/specs/2026-08-07-al-radius-compound-prereg.md` · floor Adult **0.7521**, DEGEN threshold **0.7571**

## Arm means (Adult, DP attack, 6 seeds)

| arm | config | α=0.2 DP | α=0.2 acc | α=0.3 DP | α=0.3 acc |
|---|---|---|---|---|---|
| C | r=1.0 μ=0.0 | 0.2334 | 0.7586 | 0.2614 (DEGEN) | 0.6755 |
| A | r=1.0 μ=20.0 | 0.0682 | 0.7783 | 0.0259 (DEGEN) | 0.7542 |
| R | r=2.0 μ=0.0 | 0.2291 | 0.7609 | 0.2561 (DEGEN) | 0.6774 |
| B | r=2.0 μ=20.0 | 0.0139 (DEGEN) | 0.7561 | 0.0083 (DEGEN) | 0.7071 |

## Verdict per α (pre-registered Rules 1–5)

**α=0.2: CONFLICT** — combined DEGENERATE (acc 0.7561 <= 0.7571); DP is collapse, not fairness

- C (r=1.0 μ=0.0): DP 0.2334 (R=+0.0%, p=1.0000), acc 0.7586
- A (r=1.0 μ=20.0): DP 0.0682 (R=+70.8%, p=0.0156), acc 0.7783
- R (r=2.0 μ=0.0): DP 0.2291 (R=+1.8%, p=0.0156), acc 0.7609
- B (r=2.0 μ=20.0): DP 0.0139 (R=+94.0%, p=0.0156), acc 0.7561 **DEGENERATE**
**α=0.3: CONFLICT** — combined DEGENERATE (acc 0.7071 <= 0.7571); DP is collapse, not fairness

- C (r=1.0 μ=0.0): DP 0.2614 (R=+0.0%, p=1.0000), acc 0.6755 **DEGENERATE**
- A (r=1.0 μ=20.0): DP 0.0259 (R=+90.1%, p=0.0156), acc 0.7542 **DEGENERATE**
- R (r=2.0 μ=0.0): DP 0.2561 (R=+2.0%, p=0.0156), acc 0.6774 **DEGENERATE**
- B (r=2.0 μ=20.0): DP 0.0083 (R=+96.8%, p=0.0156), acc 0.7071 **DEGENERATE**

## Overall verdict (Rule 5: α=0.2 is the scoped cell)

**CONFLICT** — see the prose paragraph below; α=0.3 is a stress test and is never used to rescue or over-claim.

**Verdict: CONFLICT.** The combined arm (radii_scale=2.0 + μ=20) at the scoped cell α=0.2 collapses: mean accuracy 0.7561 is at/below the degeneracy threshold 0.7571 (floor 0.7521 + 0.005), so its DP 0.0139 (R=+94.0%) is near-constant-predictor collapse, not fairness — the same failure mode TASK A already documented for Credit/LSAC. AL-only (r=1.0, μ=20) is the genuine Pareto win here (DP 0.0682, R=+70.8%, accuracy 0.7783 above the floor), while radius-only (r=2.0, μ=0) is essentially inert on Adult/DP (DP 0.2291, R=+1.8%). Adding the larger radius on top of the strong AL penalty destroys the accuracy margin AL preserves — the two levers do not compound into a usable result, they conflict. This DISAGREES with the pre-registered prediction (REDUNDANT): instead of the radius amplifying a near-zero g to no effect, the extra constraint pressure pushes an already-marginal regime (Adult α=0.2 canonical accuracy 0.7586 is only +0.0065 over the floor) over the edge. The degenerate counter-hypothesis flagged in the pre-registration is what occurred. At α=0.3 every arm is degenerate (canonical 0.6755 ≤ 0.7571 already, excluded regime), so nothing there is usable; it is reported as a stress test only.

R = (DP_canonical − DP_arm)/DP_canonical. `DEGENERATE` = mean accuracy ≤ 0.7571 (constant-predictor floor + 0.005): its DP is model collapse, not fairness, and is reported as such.
