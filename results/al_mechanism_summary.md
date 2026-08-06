# TASK B — mechanism: does AL suppress the corrupted points? (Adult, α=0.2, seed 0)

Hypothesis: AL penalises the DP gap hard enough to suppress the attacked 
points' influence on training, fitting them LESS than canonical DRO does. 
If true, acc_on_corrupted should be LOWER (or grow less) for AL than canonical, 
while acc_on_clean stays comparable or improves.

| trainer | μ | n_corrupted | n_clean | acc on corrupted subset | acc on clean subset | pos rate by group |
|---|---|---|---|---|---|---|
| canonical_DRO | 0.0 | 5878 | 23515 | 0.5126 | 0.8118 | {0: 0.16432881355285645, 1: 0.5387714505195618} |
| AL_mu5 | 5.0 | 5878 | 23515 | 0.1803 | 0.8421 | {0: 0.003393665188923478, 1: 0.36298272013664246} |
| AL_mu20 | 20.0 | 5878 | 23515 | 0.1776 | 0.7643 | {0: 0.0, 1: 0.04431909695267677} |

## Verdict
**Hypothesis SUPPORTED, with a caveat that matters for μ selection.** Both AL variants fit the corrupted subset far less than canonical DRO (corrupted-subset accuracy: canonical 0.5126 vs μ=5 0.1803 vs μ=20 0.1776) — the model is actively resisting the adversarial label pattern on those specific points, not just uniformly under-fitting. Clean-subset accuracy at μ=5 *improves* (0.8118→0.8421), which is clean support for denoising as the mechanism.

**At μ=20 the picture is more mixed than "denoising" alone explains.** Predicted-positive rate collapses to {group 0: 0.0%, group 1: 4.4%} — the model predicts negative almost everywhere. Adult's constant-negative-predictor accuracy is **0.7522** (computed directly from the training labels), essentially identical to μ=20's clean-subset accuracy here (**0.7643**). On this specific seed, μ=20 is sitting close to constant-predictor behavior, not a nuanced fairness-aware classifier — DP is trivially small when almost nobody is predicted positive in either group, independent of whether the classification is meaningful.

This does not contradict TASK C's finding that μ=20 is SAFE in aggregate (6-seed mean accuracy 0.7783 at α=0.2, comfortably above the 0.7526 floor+margin) — the other seeds must be pulling the mean well clear of collapse. But it means **the aggregate safety margin is thinner on individual seeds than the mean suggests**, and μ=20's mechanism is partly "suppress positive predictions toward the majority class," not purely "ignore the corrupted points while classifying normally elsewhere." This is exactly the kind of per-seed behavior TASK E's independent review should check directly rather than trusting the 6-seed mean alone.
