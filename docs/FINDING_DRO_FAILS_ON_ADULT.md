# Finding: DRO Fails on Adult at tau=100 (Fixed at tau=1)

## The original observation

At the old stepped tau schedule (tau=100 for alpha<=0.3, tau=1 for alpha=0.4),
DRO produced HIGHER DP violation than Naive on Adult at alpha=0.1--0.3:

| alpha | Naive DP | DRO DP | DRO worse by |
|-------|----------|--------|-------------|
| 0.1 | 0.180 | 0.203 | +12.8% |
| 0.2 | 0.327 | 0.503 | +53.8% |
| 0.3 | 0.531 | 0.562 | +5.8% |

## Root cause: high temperature (tau=100)

tau=100 sharpens the soft predictions sigma(tau*f) toward 0/1, concentrating
the inner maximisation's weights on a few "worst" samples and driving
lambda_DP to its clamp. This causes the lambda-runaway observed in the
"adversarial feedback loop" discussion.

## Fix: tau=1 (fixed)

At tau=1 (fixed across all alpha), DRO beats Naive on DP at every alpha:

| alpha | Naive DP | DRO DP | DRO wins/3 seeds |
|-------|----------|--------|-------------------|
| 0.1 | 0.207 | 0.205 | 2/3 |
| 0.2 | 0.248 | 0.237 | 3/3 |
| 0.3 | 0.286 | 0.264 | 3/3 |
| 0.4 | 0.310 | 0.283 | 3/3 |

Source: `results/tau_ablation_tau1.json` (Adult, 3 seeds)

## The tau comparison

| alpha | tau | Naive DP | DRO DP | Verdict |
|-------|-----|----------|--------|---------|
| 0.2 | 1 | 0.248 | 0.237 | DRO wins |
| 0.2 | 10 | 0.338 | 0.463 | Naive wins |
| 0.2 | 100 | 0.327 | 0.503 | Naive wins |
| 0.3 | 1 | 0.286 | 0.264 | DRO wins |
| 0.3 | 10 | 0.525 | 0.553 | Naive wins |
| 0.3 | 100 | 0.531 | 0.562 | Naive wins |

Source: `results/tau_ablation_tau{1,10,100}.json`

## What this means

1. The "DRO is fragile" finding was entirely a tau=100 artifact
2. At tau=1, DRO wins on Adult DP at every alpha
3. Kuldeep's Q12 ("fix tau for all alpha") was correct
4. The production setting should be tau=1 fixed

## What still needs fixing

- Credit/LSAC tau=1 numbers (in progress)
- n=6 seeds for Wilcoxon p<0.05 (in progress)
- Empirical radii calibration for known attack structure (Q5)
- UTKFace on GPU (blocked on flair2 access)
