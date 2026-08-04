# lambda_max / beta aggressiveness ablation — result: BOTH arms failed to widen the margin

**12/12 configs complete.** Adult only, attack='dp', DRO only, α∈{0.1,0.2}, 3 seeds,
two arms varied one at a time against the canonical baseline (λ_max=1.5, β=5.0).

## Arm A — raise λ_max to 2.0 (the paper's own original value)

| α | seed | acc | DP | Δ vs canonical (λ_max=1.5) |
|---|---|---|---|---|
| 0.1 | 0,1,2 | identical | identical | **0.0000** (all 3 seeds) |
| 0.2 | 0,1,2 | identical | identical | **0.0000** (all 3 seeds) |

**Result: complete no-op.** Every single one of 6 configs is byte-identical to the
canonical λ_max=1.5 run. The dual variable λ_DP simply never reaches 1.5 during
training at these α — the cap is not binding, so raising it changes nothing. This
answers the question cleanly: λ_max is **not** the reason DRO's margin is modest at
α≤0.2. (It may still matter at higher α, where λ is more likely to approach its
clamp — untested here, out of scope for this quick check.)

## Arm B — sharpen β to 10.0 (2× the tilted-risk concentration)

| α | seed | DP (β=10) | DP (canonical, β=5) | Δ (canonical − new) |
|---|---|---|---|---|
| 0.1 | 0 | 0.2203 | 0.2146 | **−0.0057** |
| 0.1 | 1 | 0.2104 | 0.2065 | **−0.0038** |
| 0.1 | 2 | 0.1951 | 0.1926 | **−0.0025** |
| 0.2 | 0 | 0.2544 | 0.2459 | **−0.0085** |
| 0.2 | 1 | 0.2428 | 0.2344 | **−0.0084** |
| 0.2 | 2 | 0.2382 | 0.2310 | **−0.0072** |

**Result: consistently worse, not better.** Sharper tilting (β=10.0) increased DP
violation on all 6/6 configs, no exceptions — a modest but completely consistent
effect (0.0025 to 0.0085 absolute DP increase). Accuracy was essentially unchanged
across both arms, so this is not an accuracy/fairness tradeoff story — sharper
tilting on this Lagrangian formulation simply concentrates the tilted risk on
worst-case *loss*, which is not the same objective as worst-case *fairness
violation*, and over-weighting it appears to work against DP rather than for it.

## Interpretation

Neither of the two proposed fairness-aggressiveness levers widened DRO's margin
over Naive; one was a no-op and the other made things modestly worse. This is a
genuine, useful negative result for the specific question Manisha raised ("can the
approach be changed to make the win bigger"): under this Lagrangian formulation, at
least at α≤0.2 on Adult, the current λ_max/β settings are not leaving an easy win on
the table. The current modest margin appears closer to what this method delivers
under the canonical protocol than to an artifact of overly conservative
hyperparameters.

**What this does NOT rule out:** (1) the third proposed lever (augmented-Lagrangian
dual stabilization, replacing the hard clamp with a soft penalty) is a different
mechanism, untested here; (2) the α-misspecification robustness idea (future work)
is a different kind of improvement — not "bigger margin," but "more reliable margin
under uncertainty about α" — and remains the strongest open direction; (3) this was
tested only on Adult/dp/α≤0.2 — Credit, LSAC, and higher α were not covered.

**Recommendation:** report both arms as tested-and-negative in the paper's Future
Work section rather than as an open suggestion — this is stronger than leaving it
unstated, and it closes off two specific reviewer questions ("did you try raising
lambda_max / sharpening beta") with real evidence rather than silence.
